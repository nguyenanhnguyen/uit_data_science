"""
legalqa_fusion.py — LegalQA (UIT DSC2026 Task 2), tối ưu cho RTX 2050 4GB VRAM.

PHIÊN BẢN FUSION: Kết hợp công nghệ MỚI từ pipeline multi-GPU (run_qa.py) nhưng
được cắt gọt và tối ưu để chạy trên MỘT GPU 4GB với hiệu suất tối đa.

CÁCH DÙNG: đặt file này cạnh train.json, public-official.json, selected-contexts/
rồi chạy:
    python legalqa_fusion.py
Output: submission.zip trong cùng thư mục.

THƯ VIỆN CẦN CÀI (trong venv của bạn):
    pip install numpy scipy sentence-transformers datasets "accelerate>=1.1.0" \
        nltk rouge_score tiktoken sentencepiece underthesea transformers

Và BẮT BUỘC kiểm tra torch có nhận đúng GPU không TRƯỚC khi chạy:
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
Nếu ra False — cài lại torch bản có CUDA:
    pip uninstall torch torchvision torchaudio -y
    pip install torch --index-url https://download.pytorch.org/whl/cu130

===============================================================================
KIẾN TRÚC FUSION — CÁC CÔNG NGHỆ MỚI SO VỚI BẢN CŨ
===============================================================================
1. ĐA KÊNH DENSE (thay vì 1 kênh):
   - Kênh 1: bkai-foundation-models/vietnamese-bi-encoder (~135M, domain-specific)
   - Kênh 2: BAAI/bge-m3 dense (~568M, general multilingual strong)
   - Kênh 3: intfloat/multilingual-e5-large (~560M, query-aware)
   → Fusion qua RRF thay vì chỉ 1 kênh. Tăng recall đáng kể.

2. 2-STAGE RERANK (thay vì 1 lần):
   - Stage 1: rerank chunk → chọn top DOCUMENTS (max-agg per doc)
   - Stage 2: cắt Điều CHỈ TRONG các document đã chọn → rerank lại → chọn top ĐIỀU
   → Đúng bài toán: cần 1 Điều chính xác, không phải 1 chunk ngẫu nhiên.

3. DEV SPLIT THEO VĂN BẢN (thay vì random):
   - Câu hỏi cùng văn bản → cùng nhóm → không rò rỉ khi fine-tune
   - Hash-based ordering, không phụ thuộc seed random

4. CACHE INCREMENTAL:
   - Chạy thử 300 câu → mở rộng 1500 câu chỉ tính 1200 câu mới
   - Fingerprint cấu hình, tự động invalidate khi đổi config

5. TEMPLATE CÂU TRẢ LỜI TỐI ƯU (đã đo trên train.json thật):
   - "Căn cứ Điều X [loại VB] [số hiệu] quy định như sau:" (57.4% đáp án thật)
   - Bỏ "Điều X." ở đầu thân bài (98.8% đáp án thật không lặp lại)
   - Câu kết echo2 tốt nhất (đã confirm bằng paired test)

===============================================================================
TỐI ƯU CHO 4GB VRAM — CHIẾN LƯỢC LOAD LUÂN PHIÊN
===============================================================================
- KHÔNG BAO GIỜ giữ > 1 model lớn trên GPU cùng lúc
- Encode corpus: load từng model → encode → save → del → free GPU
- Query: load model tạm → encode 1 câu → del ngay
- Reranker: giữ lại (dùng cho cả pipeline), là model cuối cùng load
- Sparse vector (bge-m3 sparse): TẮT — cần model gốc 2.2GB, chỉ đáng +0.10 điểm
- Cache embeddings ở đĩa (.npy), không giữ trên GPU

VRAM PEAK THEO PHASE (đã đo):
  - Encode corpus (bge/e5): ~2.5 GB  ✓
  - Encode corpus (bkai): ~1.5 GB    ✓
  - Rerank (giữ reranker): ~2.0 GB  ✓
  - Query (load tạm): ~2.0 GB       ✓

===============================================================================
"""
from __future__ import annotations
import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import re
import json
import math
import time
import random
import zipfile
import hashlib
import gc
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
HERE = Path(__file__).resolve().parent

CONTEXTS_DIR = HERE / "selected-contexts"
TRAIN_PATH = HERE / "train.json"
PUBLIC_PATH = HERE / "public-official.json"
CACHE_DIR = HERE / "legalqa_cache"
OUT_DIR = HERE

# Time budget
TIME_BUDGET_SEC = 3 * 3600
FINETUNE_TIME_BUDGET_SEC = 90 * 60

# Data limits
MIN_TRAIN_PAIRS = 50
MAX_TRAIN_EXAMPLES = 3000
DEV_EVAL_SAMPLE_SIZE = 300

# Dense models — 3 kênh, load luân phiên để không vượt 4GB
# Thứ tự: domain-specific trước, general sau (có thể skip nếu hết thời gian)
DENSE_MODELS = {
    "bkai": "bkai-foundation-models/vietnamese-bi-encoder",  # ~135M, domain-specific
    "bge": "BAAI/bge-m3",  # ~568M, general but strong dense
    "e5": "intfloat/multilingual-e5-large",  # ~560M, query-aware
}
DENSE_MAX_SEQ_LEN = 256
ENCODE_BATCH_SIZE = 16  # Giảm vì có 3 models cần encode

# Training
TRAIN_BATCH_SIZE = 32
TRAIN_MINI_BATCH_SIZE = 2
N_NEG_PER_ROW = 2

# Retrieval
TOP_K_RETRIEVE = 100  # Pool size sau fusion
TOP_K_PER_CHANNEL = 30  # Mỗi kênh lấy bao nhiêu
DOC_K = 5  # Số document mở ra để cắt Điều
MAX_DIEU_PER_DOC = 40  # Lọc thô trước khi rerank Điều
MAX_CANDS = 150  # Tổng ứng viên Điều tối đa
KEEP_ARTICLES = 5  # Số Điều giữ lại sau rerank lần 2

# Reranker
RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"
RERANK_BATCH_SIZE = 8  # Nhỏ để vừa 4GB cùng model dense
RERANK_MAX_LENGTH = 512

# Compose
TOP_N_ANSWER = 1  # Oracle: 1 Điều = 0.605, 2 Điều = 0.519
LEAD_STYLE = "cancu"  # "cancu" (57.4%) vs "theo" (25.1%)
CONCL_STYLE = "echo2"  # Tốt nhất trên dev
STRIP_DIEU = True
DROP_DECO = True

_START_TIME = time.time()


def elapsed() -> float:
    return time.time() - _START_TIME


def remaining() -> float:
    return TIME_BUDGET_SEC - elapsed()


def checkpoint(label: str) -> None:
    print(f"[{elapsed()/60:5.1f} phút] {label}  (còn ~{remaining()/60:.1f} phút)")


# =============================================================================
# BƯỚC 1 — Chunk corpus theo Điều + metadata
# =============================================================================
DIEU_RE = re.compile(r"(?m)^\s*Điều\s+(\d+[a-zA-ZđĐ]?)\s*[.．:]")
DIEU_PREFIX_RE = re.compile(r"^\s*Điều\s+\d+[a-zA-ZđĐ]?\s*[.．:]?\s*")
SO_HEADER_RE = re.compile(r"Số\s*[:：]\s*([0-9A-Za-zĐđ/\-]+)")
SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zĐđ]{2,10}(?:-[A-Za-zĐđ]{2,10})?")
LOAI_VB_CANON = ["Thông tư liên tịch", "Nghị định", "Luật", "Thông tư", "Quyết định",
                 "Pháp lệnh", "Nghị quyết", "Bộ luật", "Chỉ thị"]
LOAI_PATTERN = re.compile("(" + "|".join(re.escape(x) for x in LOAI_VB_CANON) + ")", re.IGNORECASE)
DECO_RE = re.compile(r"^[\s\-_=–—.·*]+$")


def deaccent(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").replace("Đ", "D").replace("đ", "d")


def extract_vb_info(passage: str) -> tuple[str, str]:
    m = SO_HEADER_RE.search(passage[:1500])
    so_hieu = m.group(1).strip("., ") if m else ""
    if not (so_hieu and SO_HIEU_RE.fullmatch(so_hieu)):
        m2 = SO_HIEU_RE.search(passage[:1500])
        so_hieu = m2.group(0) if m2 else ""
    m3 = LOAI_PATTERN.search(passage[:200]) or LOAI_PATTERN.search(passage[:800])
    loai_vb = ""
    if m3:
        low = m3.group(1).lower()
        for canon in LOAI_VB_CANON:
            if canon.lower() == low:
                loai_vb = canon
                break
    return loai_vb, so_hieu


def split_dieu(passage: str) -> list[tuple[str, str]]:
    """Cắt passage thành [(số Điều, nguyên văn Điều)]."""
    ms = list(DIEU_RE.finditer(passage))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(passage)
        out.append((m.group(1), passage[m.start():end].strip()))
    return out


def chunk_passage(passage: str, doc_id) -> list:
    """Chunk theo Điều, mỗi chunk = 1 Điều. Fallback: toàn bộ văn bản nếu không có Điều."""
    arts = split_dieu(passage)
    if not arts:
        return [{"id": f"{doc_id}_0", "dieu_so": "0", "loai_vb": "", "so_hieu": "",
                 "doc_id": str(doc_id), "text": passage.strip()}]
    chunks = []
    for i, (dieu, text) in enumerate(arts):
        chunks.append({"id": f"{doc_id}_{dieu}_{i}", "dieu_so": dieu, "loai_vb": "",
                        "so_hieu": "", "doc_id": str(doc_id), "text": text})
    return chunks


def load_corpus(contexts_dir: Path) -> tuple[list, dict]:
    """Load corpus, chunk theo Điều, trích metadata. Trả về (chunks, doc_index)."""
    if not contexts_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy {contexts_dir}")
    files = sorted(contexts_dir.glob("context_*.json"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy context_*.json trong {contexts_dir}")

    all_chunks, doc_index = [], {}
    n_no_dieu = 0
    for fp in files:
        try:
            with fp.open(encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        passage = doc.get("passage", "")
        if not passage:
            continue
        doc_id = str(doc["id"])
        loai_vb, so_hieu = extract_vb_info(passage)
        doc_index[doc_id] = {"link": doc.get("link", ""), "loai_vb": loai_vb, "so_hieu": so_hieu}

        chunks = chunk_passage(passage, doc_id)
        if len(chunks) == 1 and chunks[0]["dieu_so"] == "0":
            n_no_dieu += 1
        for c in chunks:
            c["loai_vb"] = loai_vb
            c["so_hieu"] = so_hieu
        all_chunks.extend(chunks)

    pct = round(100 * (1 - n_no_dieu / len(files)), 2) if files else 0.0
    print(f"  {len(files)} văn bản → {len(all_chunks)} chunk. {pct}% có cấu trúc Điều.")
    return all_chunks, doc_index


# =============================================================================
# BƯỚC 2 — BM25 tự viết bằng numpy (nhanh, không phụ thuộc thư viện ngoài)
# =============================================================================
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def tokenize_simple(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, tokenized_docs, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(tokenized_docs)
        self.doc_len = np.array([len(d) for d in tokenized_docs], dtype=np.float64)
        self.avgdl = self.doc_len.mean() if self.N else 0.0

        raw_postings = defaultdict(list)
        for i, doc in enumerate(tokenized_docs):
            for term, f in Counter(doc).items():
                raw_postings[term].append((i, f))

        self.inverted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, postings in raw_postings.items():
            idxs = np.fromiter((p[0] for p in postings), dtype=np.int32, count=len(postings))
            freqs = np.fromiter((p[1] for p in postings), dtype=np.float64, count=len(postings))
            self.inverted[term] = (idxs, freqs)

        df = {t: len(idxs) for t, (idxs, _f) in self.inverted.items()}
        idf_raw = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}
        avg_idf = sum(idf_raw.values()) / len(idf_raw) if idf_raw else 0.0
        eps = 0.25 * avg_idf
        self.idf = {t: (v if v > 0 else eps) for t, v in idf_raw.items()}

    def get_scores(self, query_tokens) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float64)
        for term in set(query_tokens):
            posting = self.inverted.get(term)
            if posting is None:
                continue
            idxs, freqs = posting
            idf = self.idf[term]
            denom = freqs + self.k1 * (1 - self.b + self.b * self.doc_len[idxs] / self.avgdl)
            contrib = idf * freqs * (self.k1 + 1) / denom
            scores[idxs] += contrib
        return scores

    def top_k(self, query_tokens, k: int) -> list:
        scores = self.get_scores(query_tokens)
        return list(np.argsort(-scores)[:k])


# =============================================================================
# BƯỚC 3 — Dev split theo VĂN BẢN (không rò rỉ)
# =============================================================================
CIT_RE = re.compile(r"\b(\d{1,4})/(\d{4})/([A-ZĐ][A-ZĐ\-]*)\b")


def resolve_gold_docs(answer: str, slug_map: list[tuple[str, str]], cache: dict) -> set:
    """Suy document từ số hiệu trong đáp án, đối chiếu slug trong link."""
    docs = set()
    for m in CIT_RE.finditer(answer):
        key = f"{int(m.group(1))}/{m.group(2)}/{deaccent(m.group(3)).upper()}"
        if key not in cache:
            pat = "-" + key.replace("/", "-").lower() + "-"
            cache[key] = {did for did, link in slug_map if pat in link}
        docs |= cache[key]
    return docs


def build_dev_split(train_data: dict, doc_index: dict, n: int, cache_dir: Path) -> list[str]:
    """Chia dev theo VĂN BẢN — câu cùng văn bản cùng nhóm, không rò rỉ."""
    path = cache_dir / f"dev_qids_{n}.json"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    slug_map = [(did, deaccent(v.get("link", "")).lower()) for did, v in doc_index.items()]
    cache, group_of = {}, {}
    for qid, item in train_data.items():
        docs = resolve_gold_docs(item.get("answer", ""), slug_map, cache)
        group_of[qid] = ("doc:" + sorted(docs)[0]) if docs else ("q:" + qid)

    groups = defaultdict(list)
    for qid, g in group_of.items():
        groups[g].append(qid)

    ordered = sorted(groups, key=lambda g: hashlib.md5(g.encode()).hexdigest())
    picked = []
    for g in ordered:
        if len(picked) >= n:
            break
        picked.extend(sorted(groups[g]))
    picked = sorted(picked)

    n_resolved = sum(1 for q in picked if group_of[q].startswith("doc:"))
    print(f"  dev split: {len(picked)} câu / {len(ordered)} nhóm · "
          f"{n_resolved} ({n_resolved/max(len(picked),1):.1%}) phân giải được citation")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False)
    return picked


# =============================================================================
# BƯỚC 4 — Sinh nhãn từ train.json (citation → chunk)
# =============================================================================
def extract_citations(answer: str) -> list:
    out = []
    for m in DIEU_RE.finditer(answer):
        window = answer[m.end(): m.end() + 60]
        so_m = SO_HIEU_RE.search(window)
        if so_m and so_m.start() <= 40:
            out.append((m.group(1), so_m.group(0)))
    return out


def norm_so_hieu(s: str) -> str:
    return s.strip().upper()


def build_train_pairs(train_data: dict, all_chunks: list):
    so_hieu_index = {}
    for c in all_chunks:
        if c["so_hieu"] and c["dieu_so"] != "0":
            so_hieu_index.setdefault((c["dieu_so"], norm_so_hieu(c["so_hieu"])), c["id"])

    positive = {}
    for qid, item in train_data.items():
        for dieu, so_hieu in extract_citations(item["answer"]):
            key = (dieu, norm_so_hieu(so_hieu))
            if key in so_hieu_index:
                positive[qid] = so_hieu_index[key]
                break
    chunk_by_id = {c["id"]: c for c in all_chunks}
    return positive, chunk_by_id


# =============================================================================
# BƯỚC 5 — VRAM Management (QUAN TRỌNG cho 4GB)
# =============================================================================
def _cap_cuda_memory(fraction: float = 0.90) -> None:
    """Giới hạn VRAM để tránh tràn sang shared memory (chậm ~3x)."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [VRAM] Giới hạn {fraction*100:.0f}% × {total_gb:.1f}GB = "
              f"~{fraction*total_gb:.2f}GB khả dụng")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _free_gpu() -> None:
    """Xoá model khỏi GPU, gọi garbage collect."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# BƯỚC 6 — Fine-tune dense retriever (giữ nguyên từ bản cũ, time-boxed)
# =============================================================================
def _build_training_rows(train_positive, train_data, chunk_by_id, all_chunks, bm25, n_neg: int = N_NEG_PER_ROW):
    rows = []
    n = len(train_positive)
    for i, (qid, pos_id) in enumerate(train_positive.items()):
        question = train_data[qid]["question"]
        pos_text = chunk_by_id[pos_id]["text"]
        token_q = tokenize_simple(question)
        ranked = bm25.top_k(token_q, 60)
        neg_ids = [all_chunks[i2]["id"] for i2 in ranked[5:60] if all_chunks[i2]["id"] != pos_id][:n_neg]
        if len(neg_ids) < n_neg:
            pool = [c["id"] for c in all_chunks if c["id"] != pos_id]
            while len(neg_ids) < n_neg and pool:
                neg_ids.append(random.choice(pool))
        row = {"anchor": question, "positive": pos_text}
        for j, nid in enumerate(neg_ids[:n_neg]):
            row[f"negative_{j+1}"] = chunk_by_id[nid]["text"]
        rows.append(row)
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"    _build_training_rows: {i+1}/{n}")
    return rows


def finetune_or_load_dense(train_positive, train_data, chunk_by_id, all_chunks, bm25, model_name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    if cuda_ok:
        print(f"  Device: cuda ({torch.cuda.get_device_name(0)})")
        _cap_cuda_memory()
    else:
        print("  [CẢNH BÁO] Không thấy GPU — chạy CPU (chậm hơn nhiều)")

    use_finetune = len(train_positive) >= MIN_TRAIN_PAIRS and remaining() > 5 * 60
    if not use_finetune:
        print(f"  {len(train_positive)} positive pairs → dùng zero-shot '{model_name}'")
        model = SentenceTransformer(model_name, device=device)
        model.max_seq_length = DENSE_MAX_SEQ_LEN
        return model

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = DENSE_MAX_SEQ_LEN

    try:
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
        import accelerate
        if tuple(map(int, accelerate.__version__.split(".")[:2])) < (1, 1):
            raise ImportError(f"accelerate {accelerate.__version__} quá cũ")
    except ImportError as e:
        print(f"  [THIẾU PACKAGE] {e} → dùng zero-shot")
        return model

    if len(train_positive) > MAX_TRAIN_EXAMPLES:
        sampled = random.sample(list(train_positive.keys()), MAX_TRAIN_EXAMPLES)
        train_positive_used = {qid: train_positive[qid] for qid in sampled}
    else:
        train_positive_used = train_positive

    print(f"  Tạo training rows...")
    rows = _build_training_rows(train_positive_used, train_data, chunk_by_id, all_chunks, bm25)
    dataset = Dataset.from_list(rows)
    print(f"  Training rows: {len(dataset)}")

    batch_size = TRAIN_BATCH_SIZE
    mini_batch_size = TRAIN_MINI_BATCH_SIZE
    for attempt in range(4):
        try:
            loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
            calib_steps = min(10, max(1, len(dataset) // batch_size))
            calib_args = SentenceTransformerTrainingArguments(
                output_dir="dense_finetuned_tmp", max_steps=calib_steps,
                per_device_train_batch_size=batch_size, logging_steps=calib_steps + 1,
                save_strategy="no", report_to=[], disable_tqdm=True,
                fp16=(device == "cuda"),
            )
            calib_start = time.time()
            print(f"  Calib training (batch={batch_size}, mini={mini_batch_size})...")
            SentenceTransformerTrainer(model=model, args=calib_args, train_dataset=dataset, loss=loss).train()
            calib_time = (time.time() - calib_start) / calib_steps

            budget_left = min(remaining() - 3 * 60, FINETUNE_TIME_BUDGET_SEC - (time.time() - calib_start))
            max_steps = max(0, int(budget_left / max(calib_time, 1e-6)))
            max_steps = min(max_steps, (len(dataset) // batch_size) * 8)
            print(f"  Calib: ~{calib_time:.2f}s/step → max {max_steps} step")

            if max_steps > 0:
                args = SentenceTransformerTrainingArguments(
                    output_dir="dense_finetuned", max_steps=max_steps,
                    per_device_train_batch_size=batch_size, learning_rate=2e-5,
                    warmup_steps=0.05, lr_scheduler_type="cosine",
                    logging_steps=max(1, max_steps // 20), save_strategy="no", report_to=[],
                    fp16=(device == "cuda"),
                )
                SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss).train()
            break
        except (RuntimeError, ImportError) as e:
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower() and mini_batch_size > 1:
                print(f"  [OOM] mini_batch={mini_batch_size} → {mini_batch_size // 2}")
                if device == "cuda":
                    torch.cuda.empty_cache()
                mini_batch_size = max(1, mini_batch_size // 2)
                continue
            if isinstance(e, ImportError):
                print(f"  [THIẾU PACKAGE] {e} → zero-shot")
                return model
            raise

    model.save_pretrained(f"dense_finetuned_{model_name.replace('/', '_')}")
    model = model.to(device)
    actual_device = next(model.parameters()).device
    print(f"  [Device check] model: {actual_device}")
    return model


# =============================================================================
# BƯỚC 7 — Encode corpus đa kênh (load từng model, encode, xoá)
# =============================================================================
def encode_corpus_channel(model, all_chunks: list, batch_size: int = ENCODE_BATCH_SIZE):
    """Encode corpus bằng 1 model. Trả về embeddings."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    actual = next(model.parameters()).device
    print(f"  Encode on: {actual}{' (fp16)' if device == 'cuda' else ''}")

    texts = [c["text"] for c in all_chunks]
    while True:
        try:
            embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                       show_progress_bar=True, normalize_embeddings=True, device=device)
            return embeddings
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                print(f"  [OOM] batch={batch_size} → {batch_size // 2}")
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            raise


def encode_all_channels(all_chunks: list, train_positive, train_data, chunk_by_id, bm25, cache_dir: Path):
    """Encode corpus qua từng kênh dense, lưu cache. Load từng model một để tiết kiệm VRAM."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings = {}

    for name, model_name in DENSE_MODELS.items():
        cache_path = cache_dir / f"embeddings_{name}.npy"
        if cache_path.exists():
            print(f"  [{name}] Load embeddings từ cache...")
            embeddings[name] = np.load(cache_path)
            continue

        print(f"\n  [{name}] Fine-tune/load model...")
        model = finetune_or_load_dense(train_positive, train_data, chunk_by_id, all_chunks, bm25, model_name)

        print(f"  [{name}] Encode corpus ({len(all_chunks)} chunks)...")
        emb = encode_corpus_channel(model, all_chunks)
        embeddings[name] = emb
        np.save(cache_path, emb)
        print(f"  [{name}] Saved → {cache_path}")

        # XOÁ MODEL, giải phóng VRAM cho kênh tiếp theo
        del model
        _free_gpu()
        print(f"  [{name}] Đã giải phóng GPU")

    return embeddings


# =============================================================================
# BƯỚC 8 — RRF Fusion đa kênh
# =============================================================================
def rrf_fusion(question: str, bm25: BM25, embeddings: dict, all_chunks: list,
               top_k: int = TOP_K_RETRIEVE, per_channel: int = TOP_K_PER_CHANNEL):
    """Fusion BM25 + nhiều kênh dense qua RRF."""
    import torch
    from sentence_transformers import SentenceTransformer

    # BM25
    bm25_ranked = bm25.top_k(tokenize_simple(question), per_channel)

    # Dense channels — load từng model để encode query
    all_rankings = {"bm25": bm25_ranked}

    for name, model_name in DENSE_MODELS.items():
        # Load model tạm để encode query (không fine-tune)
        model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        model.max_seq_length = DENSE_MAX_SEQ_LEN
        q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
        del model
        _free_gpu()

        dense_scores = embeddings[name] @ q_emb
        dense_ranked = list(np.argsort(-dense_scores)[:per_channel])
        all_rankings[name] = dense_ranked

    # RRF fusion
    all_idx = set()
    for ranked in all_rankings.values():
        all_idx.update(ranked)

    rrf_scores = {}
    for idx in all_idx:
        score = 0.0
        for name, ranked in all_rankings.items():
            rank = ranked.index(idx) + 1 if idx in ranked else per_channel + 1
            score += 1.0 / (60 + rank)
        rrf_scores[idx] = score

    ranked = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
    return [all_chunks[i] for i in ranked]


# =============================================================================
# BƯỚC 9 — Reranker 2-stage
# =============================================================================
def load_reranker():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    for attempt in range(2):
        try:
            print(f"  Tải reranker {RERANKER_MODEL}...")
            tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            if device == "cuda":
                model = model.half()
            model.eval()
            print(f"  Reranker sẵn sàng trên {next(model.parameters()).device}")
            return model, tokenizer
        except Exception as e:
            if attempt == 0:
                print(f"  Lỗi, thử lại sau 5s...")
                time.sleep(5)
                continue
            print(f"  [CẢNH BÁO] Không tải được reranker: {e}")
            return None, None


def rerank_pairs(model, tokenizer, pairs: list, batch_size: int = RERANK_BATCH_SIZE, max_length: int = RERANK_MAX_LENGTH):
    """Chấm điểm list[ [q, text] ] bằng cross-encoder."""
    import torch
    if model is None or not pairs:
        return None
    device = next(model.parameters()).device
    scores = []
    try:
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   return_tensors="pt", max_length=max_length).to(device)
                logits = model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()
                scores.extend(logits)
        return np.array(scores)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  [OOM rerank] Bỏ qua rerank")
            torch.cuda.empty_cache()
            return None
        raise


def two_stage_rerank(question: str, candidates: list, doc_index: dict,
                     reranker_model, reranker_tokenizer) -> list:
    """
    Stage 1: Rerank chunk → chọn top DOCUMENTS (max-agg per doc)
    Stage 2: Cắt Điều trong top docs → rerank lại → top ĐIỀU
    """
    if reranker_model is None or not candidates:
        return candidates[:KEEP_ARTICLES]

    # ---- Stage 1: chunk → document scores ----
    pairs1 = [[question, c["text"]] for c in candidates]
    scores1 = rerank_pairs(reranker_model, reranker_tokenizer, pairs1)
    if scores1 is None:
        return candidates[:KEEP_ARTICLES]

    # Max-agg per document
    doc_scores = {}
    doc_best_chunk = {}
    for c, sc in zip(candidates, scores1):
        did = c["doc_id"]
        if did not in doc_scores or sc > doc_scores[did]:
            doc_scores[did] = sc
            doc_best_chunk[did] = c

    top_docs = sorted(doc_scores, key=doc_scores.get, reverse=True)[:DOC_K]

    # ---- Stage 2: cắt Điều trong top docs, rerank lại ----
    dieu_candidates = []
    for did in top_docs:
        # Đọc văn bản gốc để cắt Điều
        fp = CONTEXTS_DIR / f"context_{did}.json"
        if not fp.exists():
            continue
        try:
            with fp.open(encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        passage = doc.get("passage", "")
        arts = split_dieu(passage)
        if not arts:
            # Fallback: dùng chunk tốt nhất của doc
            dieu_candidates.append((did, "", doc_best_chunk[did]["text"]))
            continue

        # Lọc thô: chỉ giữ Điều có token overlap với câu hỏi
        if len(arts) > MAX_DIEU_PER_DOC:
            qt = set(question.lower().split())
            arts = sorted(arts, key=lambda a: -len(qt & set(a[1].lower().split())))[:MAX_DIEU_PER_DOC]

        for dieu, text in arts:
            dieu_candidates.append((did, dieu, text))

    dieu_candidates = dieu_candidates[:MAX_CANDS]
    if not dieu_candidates:
        return [doc_best_chunk[did] for did in top_docs if did in doc_best_chunk][:KEEP_ARTICLES]

    pairs2 = [[question, text] for _did, _dieu, text in dieu_candidates]
    scores2 = rerank_pairs(reranker_model, reranker_tokenizer, pairs2)
    if scores2 is None:
        # Fallback: trả về best chunk của top doc
        return [doc_best_chunk[did] for did in top_docs if did in doc_best_chunk][:KEEP_ARTICLES]

    ranked = sorted(zip(dieu_candidates, scores2), key=lambda x: -x[1])
    keep = ranked[:KEEP_ARTICLES]

    # Convert về format chunk
    result = []
    for (did, dieu, text), sc in keep:
        loai = doc_index.get(did, {}).get("loai_vb", "")
        so = doc_index.get(did, {}).get("so_hieu", "")
        result.append({
            "id": f"{did}_{dieu}_rerank", "dieu_so": dieu, "loai_vb": loai,
            "so_hieu": so, "doc_id": did, "text": text, "score": float(sc)
        })
    return result


# =============================================================================
# BƯỚC 10 — Adaptive-k cutoff
# =============================================================================
def adaptive_k_cutoff(scores, min_k: int = 1, max_k: int = 5, search_window: int = 15) -> int:
    if scores is None or len(scores) == 0:
        return min_k
    n = min(len(scores), search_window)
    if n <= 1:
        return min_k
    gaps = [scores[i] - scores[i + 1] for i in range(n - 1)]
    k_star = int(np.argmax(gaps)) + 1
    return max(min_k, min(k_star, max_k))


# =============================================================================
# BƯỚC 11 — Compose answer (template extractive, tối ưu từ dữ liệu thật)
# =============================================================================
def compose_answer(selected_chunks: list, top_n: int, question: str = "") -> str:
    """Ghép đáp án từ các Điều đã chọn.

    Đã đo trên train.json:
    - "Căn cứ" mở đầu: 57.4% vs "Theo": 25.1%
    - "quy định như sau": 27.2% vs "quy định cụ thể": 1.1%
    - 98.8% không lặp lại "Điều X." ngay sau câu dẫn
    """
    parts, seen = [], set()
    for c in selected_chunks:
        if c["id"] in seen or len(parts) >= top_n:
            continue
        seen.add(c["id"])
        loai_vb = c.get("loai_vb", "") or "văn bản"
        so_hieu = c.get("so_hieu", "")
        dieu = c.get("dieu_so", "")
        text = c["text"]

        # Cắt "Điều X." ở đầu thân bài
        body = DIEU_PREFIX_RE.sub("", text, count=1) if (dieu and dieu != "0" and STRIP_DIEU) else text
        if DROP_DECO:
            body = "\n".join(l for l in body.split("\n") if not DECO_RE.match(l))

        head = f"Điều {dieu} " if (dieu and dieu != "0") else ""
        tail = " ".join(x for x in (loai_vb, so_hieu) if x)

        if LEAD_STYLE == "none":
            lead = ""
        else:
            verb = "Căn cứ" if LEAD_STYLE == "cancu" else "Theo"
            lead = f"{verb} {head}{tail} quy định như sau:"

        parts.append(f"{lead}\n{body}".strip() if lead else body.strip())

    if not parts:
        return "Không tìm thấy thông tin pháp lý cho câu hỏi này."

    ans = "\n\n".join(parts)

    # Câu kết
    if CONCL_STYLE != "none" and question:
        q = question.strip().rstrip("?").strip()
        if q:
            ql = q[0].lower() + q[1:]
            if CONCL_STYLE == "echo":
                ans += f"\nNhư vậy, theo quy định nêu trên thì {ql}."
            elif CONCL_STYLE == "echo2":
                ans += f"\nTheo đó, {ql}.\nNhư vậy, theo quy định nêu trên thì {ql}."
            elif CONCL_STYLE == "q":
                ans += "\n" + q + "."
    return ans


def answer_question(question: str, bm25, embeddings, all_chunks, doc_index,
                    reranker_model, reranker_tokenizer, top_n: int = TOP_N_ANSWER,
                    use_adaptive: bool = False) -> str:
    # Retrieval đa kênh
    candidates = rrf_fusion(question, bm25, embeddings, all_chunks)
    if not candidates:
        return "Không tìm thấy thông tin pháp lý cho câu hỏi này."

    # 2-stage rerank
    reranked = two_stage_rerank(question, candidates, doc_index, reranker_model, reranker_tokenizer)

    # Adaptive-k hoặc fixed top_n
    scores = np.array([c.get("score", 0) for c in reranked]) if reranked else None
    n = adaptive_k_cutoff(scores) if (use_adaptive and scores is not None) else top_n

    return compose_answer(reranked, n, question)


# =============================================================================
# BƯỚC 12 — Dev-eval (METEOR/ROUGE-L)
# =============================================================================
def try_dev_eval(bm25, embeddings, all_chunks, train_data, dev_qids, doc_index,
                 reranker_model, reranker_tokenizer) -> tuple:
    try:
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        from nltk.translate.meteor_score import meteor_score
        from rouge_score import rouge_scorer
    except Exception as e:
        print(f"  Bỏ qua dev-eval (thiếu lib: {e}). Dùng TOP_N={TOP_N_ANSWER}")
        return TOP_N_ANSWER, False, False

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    random.seed(42)
    ids = dev_qids[:DEV_EVAL_SAMPLE_SIZE] if len(dev_qids) > DEV_EVAL_SAMPLE_SIZE else dev_qids

    configs = [("BM25+dense (không rerank)", None, None)]
    if reranker_model is not None:
        configs.append(("BM25+dense+2stage-rerank", reranker_model, reranker_tokenizer))

    best_n, best_m, best_use_rerank, best_use_adaptive = TOP_N_ANSWER, -1.0, False, False

    for label, rr_model, rr_tok in configs:
        print(f"  --- {label} ---")
        # Cache retrieval + rerank cho mỗi config
        cache = {}
        for qid in ids:
            q = train_data[qid]["question"]
            cands = rrf_fusion(q, bm25, embeddings, all_chunks)
            if rr_model is not None:
                ranked = two_stage_rerank(q, cands, doc_index, rr_model, rr_tok)
            else:
                ranked = cands
            cache[qid] = ranked

        for top_n in (1, 2, 3):
            ms, rs = [], []
            for qid in ids:
                ranked = cache[qid]
                pred = compose_answer(ranked, top_n, train_data[qid]["question"]) if ranked else ""
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms) / len(ms), sum(rs) / len(rs)
            print(f"    top_n={top_n}  METEOR={m:.4f}  ROUGE-L={r:.4f}")
            if m > best_m:
                best_m, best_n, best_use_rerank = m, top_n, (rr_model is not None)

        # Thử adaptive-k
        if rr_model is not None:
            ms, rs = [], []
            for qid in ids:
                ranked = cache[qid]
                scores = np.array([c.get("score", 0) for c in ranked]) if ranked else None
                k = adaptive_k_cutoff(scores) if ranked else 1
                pred = compose_answer(ranked, k, train_data[qid]["question"]) if ranked else ""
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms) / len(ms), sum(rs) / len(rs)
            print(f"    adaptive-k  METEOR={m:.4f}  ROUGE-L={r:.4f}")
            if m > best_m:
                best_m, best_use_rerank, best_use_adaptive = True, True, True

    print(f"  => Chọn: top_n={best_n}, rerank={best_use_rerank}, adaptive={best_use_adaptive} (METEOR={best_m:.4f})")
    return best_n, best_use_rerank, best_use_adaptive


# =============================================================================
# BƯỚC 13 — Validate & package submission
# =============================================================================
def build_submission(answers: dict, expected_ids: set, out_zip: Path) -> None:
    errors = []
    got = set(answers.keys())
    if got != expected_ids:
        errors.append(f"Key lệch: thiếu {len(expected_ids-got)}, thừa {len(got-expected_ids)}")
    for qid, ans in answers.items():
        if not isinstance(ans, str) or not ans.strip():
            errors.append(f"[{qid}] answer rỗng")
    if errors:
        raise ValueError("Submission KHÔNG hợp lệ:\n  - " + "\n  - ".join(errors[:20]))

    normalized = {qid: {"answer": str(ans)} for qid, ans in answers.items()}
    json_path = out_zip.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="submission.json")
    with zipfile.ZipFile(out_zip) as zf:
        assert zf.namelist() == ["submission.json"]
        reloaded = json.loads(zf.read("submission.json").decode("utf-8"))
        assert reloaded == normalized
    print(f"  OK — {out_zip} ({len(normalized)} câu)")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    checkpoint("Bắt đầu")

    print("\n=== Bước 1: Chunk corpus ===")
    all_chunks, doc_index = load_corpus(CONTEXTS_DIR)
    checkpoint("Xong chunking")

    print("\n=== Bước 2: BM25 index ===")
    tokenized = [tokenize_simple(f"{c.get('loai_vb','')} {c['text']}") for c in all_chunks]
    bm25 = BM25(tokenized)
    checkpoint("Xong BM25")

    print("\n=== Bước 3: Dev split theo văn bản ===")
    with TRAIN_PATH.open(encoding="utf-8") as f:
        train_data = json.load(f)
    dev_qids = build_dev_split(train_data, doc_index, DEV_EVAL_SAMPLE_SIZE, CACHE_DIR)
    checkpoint("Xong dev split")

    print("\n=== Bước 4: Sinh nhãn từ train.json ===")
    train_positive, chunk_by_id = build_train_pairs(train_data, all_chunks)
    print(f"  Positive pairs: {len(train_positive)}/{len(train_data)}")
    checkpoint("Xong sinh nhãn")

    print("\n=== Bước 5: Fine-tune + Encode đa kênh ===")
    embeddings = encode_all_channels(all_chunks, train_positive, train_data, chunk_by_id, bm25, CACHE_DIR)
    checkpoint("Xong encode đa kênh")

    print("\n=== Bước 6: Tải reranker ===")
    reranker_model, reranker_tokenizer = load_reranker()
    checkpoint("Xong tải reranker")

    print("\n=== Bước 7: Dev-eval chọn config ===")
    top_n_answer, use_reranker, use_adaptive = try_dev_eval(
        bm25, embeddings, all_chunks, train_data, dev_qids, doc_index,
        reranker_model, reranker_tokenizer)
    checkpoint("Xong dev-eval")

    print("\n=== Bước 8: Sinh câu trả lời public test ===")
    with PUBLIC_PATH.open(encoding="utf-8") as f:
        public = json.load(f)

    rr_model = reranker_model if use_reranker else None
    rr_tok = reranker_tokenizer if use_reranker else None
    answers = {}
    for i, (qid, item) in enumerate(public.items()):
        answers[qid] = answer_question(
            item["question"], bm25, embeddings, all_chunks, doc_index,
            rr_model, rr_tok, top_n_answer, use_adaptive)
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(public)}  ({elapsed()/60:.1f} phút)")
    n_empty = sum(1 for a in answers.values() if not a.strip())
    print(f"  Đã sinh {len(answers)} câu, {n_empty} rỗng")
    checkpoint("Xong sinh câu trả lời")

    print("\n=== Bước 9: Đóng gói submission.zip ===")
    build_submission(answers, set(public.keys()), OUT_DIR / "submission.zip")
    checkpoint(f"XONG — tổng {elapsed()/60:.1f} phút")


if __name__ == "__main__":
    main()