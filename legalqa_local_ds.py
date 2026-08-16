"""
legalqa_local_v2.py -- LegalQA (UIT DSC2026 Task 2)
Patch tối thiểu trên legalqa_local.py gốc:
  1. Chunking: sliding window fallback cho văn bản không có "Điều"
  2. build_train_pairs: tìm theo số hiệu đơn lẻ
  3. Reranker: không .half(), bỏ token_type_ids, try/except CUDA assert
  4. LLM (optional): Qwen2.5-0.5B fp16, không cần bitsandbytes, tắt mặc định

CÁCH DÙNG:
    python legalqa_local_v2.py

Để bật LLM (sau khi test extractive ổn định):
    Sửa USE_LLM = True ở dòng CONFIG bên dưới
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
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


# ==============================================================================
# CONFIG
# ==============================================================================
HERE = Path(__file__).resolve().parent
CONTEXTS_DIR = HERE / "selected-contexts"
TRAIN_PATH = HERE / "train.json"
PUBLIC_PATH = HERE / "public-official.json"
OUT_DIR = HERE

TIME_BUDGET_SEC = 3 * 3600
FINETUNE_TIME_BUDGET_SEC = 90 * 60
MIN_TRAIN_PAIRS = 10               # Giảm xuống vì nhiều văn bản không có Điều
MAX_TRAIN_EXAMPLES = 3000

BASE_DENSE_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
DENSE_MAX_SEQ_LEN = 256
TRAIN_BATCH_SIZE = 8
ENCODE_BATCH_SIZE = 32

TOP_K_RETRIEVE = 100
DEV_EVAL_SAMPLE_SIZE = 300

# --- Reranker ---
RERANKER_PRIMARY = "AITeamVN/Vietnamese_Reranker"
RERANKER_FALLBACK = "itdainb/PhoRanker"
RERANKER_MAX_LENGTH = 384

# --- LLM (OPTIONAL, tắt mặc định để giữ ổn định) ---
USE_LLM = False                    # Đổi thành True nếu muốn thử LLM
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_MAX_NEW_TOKENS = 320
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.95

_START_TIME = time.time()

def elapsed() -> float:
    return time.time() - _START_TIME

def remaining() -> float:
    return TIME_BUDGET_SEC - elapsed()

def checkpoint(label: str) -> None:
    print(f"[{elapsed()/60:5.1f} phút] {label}  (còn lại ~{remaining()/60:.1f} phút trong ngân sách)")


# ==============================================================================
# BƯỚC 1 — Chunk corpus (PATCHED: thêm sliding window fallback)
# ==============================================================================
DIEU_RE = re.compile(r"^[ \t]*Điều\s+(\d+)[a-zđA-ZĐ]?[\.\s]", re.MULTILINE)
KHOAN_RE = re.compile(r"^[ \t]*Khoản\s+(\d+)[\.\s]", re.MULTILINE)
CHUONG_RE = re.compile(r"^[ \t]*Chương\s+[IVX\d]+", re.MULTILINE)
MUC_RE = re.compile(r"^[ \t]*Mục\s+\d+", re.MULTILINE)

SO_HEADER_RE = re.compile(r"Số\s*[:：]\s*([0-9A-Za-zĐđ/\-]+)")
SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zĐđ]{2,10}(?:-[A-Za-zĐđ]{2,10})?")
LOAI_VB_CANON = ["Thông tư liên tịch", "Nghị định", "Luật", "Thông tư", "Quyết định",
                 "Pháp lệnh", "Nghị quyết", "Bộ luật", "Chỉ thị"]
LOAI_PATTERN = re.compile("(" + "|".join(re.escape(x) for x in LOAI_VB_CANON) + ")", re.IGNORECASE)


def extract_vb_info(passage: str):
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


def chunk_passage(passage: str, doc_id) -> list:
    """PATCH: Thử Điều -> Khoản -> Chương/Mục -> Sliding window."""
    # 1. Thử tìm Điều
    matches = list(DIEU_RE.finditer(passage))
    if matches:
        chunks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(passage)
            dieu = m.group(1)
            chunks.append({"id": f"{doc_id}_{dieu}_{i}", "dieu_so": dieu, "loai_vb": "", "so_hieu": "",
                            "text": passage[start:end].strip()})
        return chunks

    # 2. Thử tìm Khoản
    matches = list(KHOAN_RE.finditer(passage))
    if matches:
        chunks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(passage)
            khoan = m.group(1)
            chunks.append({"id": f"{doc_id}_k{khoan}_{i}", "dieu_so": khoan, "loai_vb": "", "so_hieu": "",
                            "text": passage[start:end].strip()})
        return chunks

    # 3. Thử Chương/Mục
    matches = list(CHUONG_RE.finditer(passage)) or list(MUC_RE.finditer(passage))
    if matches:
        chunks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(passage)
            chunks.append({"id": f"{doc_id}_sec{i}", "dieu_so": str(i+1), "loai_vb": "", "so_hieu": "",
                            "text": passage[start:end].strip()})
        return chunks

    # 4. PATCH: Sliding window fallback cho văn bản hành chính không có cấu trúc
    text = passage.strip()
    if len(text) <= 2000:
        return [{"id": f"{doc_id}_0", "dieu_so": "0", "loai_vb": "", "so_hieu": "", "text": text}]

    chunks = []
    window_size = 1500
    overlap = 300
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        if end < len(text):
            for j in range(end, max(end-200, start), -1):
                if text[j-1] in '.\n':
                    end = j
                    break
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"id": f"{doc_id}_w{idx}", "dieu_so": str(idx), "loai_vb": "", "so_hieu": "",
                            "text": chunk_text})
        start = end - overlap
        idx += 1
        if start >= len(text) - overlap:
            break
    return chunks if chunks else [{"id": f"{doc_id}_0", "dieu_so": "0", "loai_vb": "", "so_hieu": "", "text": text}]


def load_corpus(contexts_dir: Path) -> list:
    if not contexts_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy {contexts_dir}")
    files = sorted(contexts_dir.glob("context_*.json"))
    if not files:
        nested = contexts_dir / "selected-contexts"
        if nested.exists():
            files = sorted(nested.glob("context_*.json"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy context_*.json trong {contexts_dir}")

    all_chunks, n_no_struct = [], 0
    for fp in files:
        try:
            with fp.open(encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        passage = doc.get("passage")
        if not passage:
            continue
        chunks = chunk_passage(passage, doc["id"])
        has_struct = any(c["dieu_so"] != "0" and not c["id"].endswith("_0") for c in chunks)
        if not has_struct:
            n_no_struct += 1
        loai_vb, so_hieu = extract_vb_info(passage)
        for c in chunks:
            c["loai_vb"], c["so_hieu"] = loai_vb, so_hieu
        all_chunks.extend(chunks)

    pct = round(100 * (1 - n_no_struct / len(files)), 2) if files else 0.0
    print(f"  {len(files)} văn bản -> {len(all_chunks)} chunk. {pct}% có cấu trúc pháp lý.")
    return all_chunks


# ==============================================================================
# BƯỚC 2 — BM25 (giữ nguyên)
# ==============================================================================
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


# ==============================================================================
# BƯỚC 3 — Sinh nhãn (PATCHED: tìm theo số hiệu đơn lẻ)
# ==============================================================================
DIEU_CITATION_RE = re.compile(r"Điều\s+(\d+)\s*[a-zđA-ZĐ]?\b")


def extract_citations(answer: str) -> list:
    out = []
    for m in DIEU_CITATION_RE.finditer(answer):
        window = answer[m.end(): m.end() + 80]
        so_m = SO_HIEU_RE.search(window)
        if so_m and so_m.start() <= 50:
            out.append((m.group(1), so_m.group(0)))
    return out


def norm_so_hieu(s: str) -> str:
    return s.strip().upper()


def build_train_pairs(train_data: dict, all_chunks: list):
    """PATCH: Index tất cả chunk có số hiệu, tìm theo số hiệu đơn lẻ nếu không match (Điều, số hiệu)."""
    so_hieu_index = {}
    so_hieu_only_index = {}

    for c in all_chunks:
        if c["so_hieu"]:
            key_full = (c["dieu_so"], norm_so_hieu(c["so_hieu"]))
            so_hieu_index.setdefault(key_full, c["id"])
            key_only = norm_so_hieu(c["so_hieu"])
            so_hieu_only_index.setdefault(key_only, c["id"])

    positive = {}
    for qid, item in train_data.items():
        found = False
        # Thử (Điều, số hiệu)
        for dieu, so_hieu in extract_citations(item["answer"]):
            key = (dieu, norm_so_hieu(so_hieu))
            if key in so_hieu_index:
                positive[qid] = so_hieu_index[key]
                found = True
                break
        if found:
            continue
        # Thử số hiệu đơn lẻ
        for _dieu, so_hieu in extract_citations(item["answer"]):
            key_only = norm_so_hieu(so_hieu)
            if key_only in so_hieu_only_index:
                positive[qid] = so_hieu_only_index[key_only]
                found = True
                break
        if found:
            continue
        # Thử tất cả số hiệu trong answer
        all_so = SO_HIEU_RE.findall(item["answer"])
        for so in all_so:
            key_only = norm_so_hieu(so)
            if key_only in so_hieu_only_index:
                positive[qid] = so_hieu_only_index[key_only]
                break

    chunk_by_id = {c["id"]: c for c in all_chunks}
    return positive, chunk_by_id


# ==============================================================================
# BƯỚC 4 — Fine-tune dense retriever (giữ nguyên)
# ==============================================================================
def _build_training_rows(train_positive, train_data, chunk_by_id, all_chunks, bm25, n_neg: int = 4):
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
            print(f"    _build_training_rows: {i+1}/{n}  ({elapsed()/60:.1f} phút)")
    return rows


def _cap_cuda_memory(fraction: float = 0.92) -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [Giới hạn VRAM] Ép trần {fraction*100:.0f}% x {total_gb:.1f}GB = "
              f"~{fraction*total_gb:.2f}GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def finetune_or_load_dense(train_positive, train_data, chunk_by_id, all_chunks, bm25):
    import torch
    from sentence_transformers import SentenceTransformer

    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    if cuda_ok:
        print(f"  Device: cuda ({torch.cuda.get_device_name(0)})")
        _cap_cuda_memory()
    else:
        print("  [CẢNH BÁO] torch.cuda.is_available()=False")

    use_finetune = len(train_positive) >= MIN_TRAIN_PAIRS and remaining() > 5 * 60
    if not use_finetune:
        print(f"  {len(train_positive)} positive pairs -> zero-shot '{BASE_DENSE_MODEL}'.")
        model = SentenceTransformer(BASE_DENSE_MODEL, device=device)
        model.max_seq_length = DENSE_MAX_SEQ_LEN
        return model

    model = SentenceTransformer(BASE_DENSE_MODEL, device=device)
    model.max_seq_length = DENSE_MAX_SEQ_LEN

    try:
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        import accelerate
        if tuple(map(int, accelerate.__version__.split(".")[:2])) < (1, 1):
            raise ImportError(f"accelerate {accelerate.__version__} quá cũ")
    except ImportError as e:
        print(f"  [THIẾU PACKAGE] {e} -> zero-shot.")
        return model

    if len(train_positive) > MAX_TRAIN_EXAMPLES:
        sampled_qids = random.sample(list(train_positive.keys()), MAX_TRAIN_EXAMPLES)
        train_positive_used = {qid: train_positive[qid] for qid in sampled_qids}
        print(f"  Sampled {MAX_TRAIN_EXAMPLES}/{len(train_positive)} pairs.")
    else:
        train_positive_used = train_positive

    print(f"  Building training rows ...")
    rows = _build_training_rows(train_positive_used, train_data, chunk_by_id, all_chunks, bm25)
    dataset = Dataset.from_list(rows)
    print(f"  Training rows: {len(dataset)}")

    batch_size = TRAIN_BATCH_SIZE
    for attempt in range(3):
        try:
            loss = MultipleNegativesRankingLoss(model)
            calib_steps = min(10, max(1, len(dataset) // batch_size))
            calib_args = SentenceTransformerTrainingArguments(
                output_dir="dense_finetuned_tmp", max_steps=calib_steps,
                per_device_train_batch_size=batch_size, logging_steps=calib_steps + 1,
                save_strategy="no", report_to=[], disable_tqdm=True,
                fp16=(device == "cuda"),
            )
            calib_start = time.time()
            print("  Calibrating ...")
            SentenceTransformerTrainer(model=model, args=calib_args, train_dataset=dataset, loss=loss).train()
            calib_time = (time.time() - calib_start) / calib_steps

            budget_left = min(remaining() - 3 * 60, FINETUNE_TIME_BUDGET_SEC - (time.time() - calib_start))
            max_steps = max(0, int(budget_left / max(calib_time, 1e-6)))
            max_steps = min(max_steps, (len(dataset) // batch_size) * 8)
            print(f"  Calib: ~{calib_time:.2f}s/step -> max_steps={max_steps}")

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
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower() and batch_size > 1:
                print(f"  [OOM] batch={batch_size} -> {batch_size//2}")
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            raise

    model.save_pretrained("dense_finetuned")
    model = model.to(device)
    actual_device = next(model.parameters()).device
    print(f"  [Device check] model: {actual_device}")
    return model


# ==============================================================================
# BƯỚC 5 — Encode corpus + RRF (giữ nguyên)
# ==============================================================================
def encode_corpus(model, all_chunks: list):
    import torch
    if torch.cuda.is_available():
        _cap_cuda_memory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    actual_device = next(model.parameters()).device
    print(f"  [Encode] device: {actual_device}{' fp16' if device=='cuda' else ''}")
    if actual_device.type != device and device == "cuda":
        raise RuntimeError(f"Model không ở GPU sau .to('cuda')")

    texts = [c["text"] for c in all_chunks]
    batch_size = ENCODE_BATCH_SIZE
    while True:
        try:
            embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                       show_progress_bar=True, normalize_embeddings=True,
                                       device=device)
            return embeddings
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                print(f"  [OOM] encode batch={batch_size} -> {batch_size//2}")
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            raise


def rrf_retrieve(question: str, bm25: BM25, dense_model, dense_embeddings, all_chunks, top_k: int = TOP_K_RETRIEVE):
    bm25_ranked = bm25.top_k(tokenize_simple(question), top_k)
    q_emb = dense_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    dense_scores = dense_embeddings @ q_emb
    dense_ranked = list(np.argsort(-dense_scores)[:top_k])

    bm25_rank_map = {idx: r for r, idx in enumerate(bm25_ranked)}
    dense_rank_map = {idx: r for r, idx in enumerate(dense_ranked)}
    all_idx = set(bm25_ranked) | set(dense_ranked)
    rrf = {i: 1 / (60 + bm25_rank_map.get(i, top_k + 1)) + 1 / (60 + dense_rank_map.get(i, top_k + 1))
           for i in all_idx}
    ranked = sorted(rrf, key=rrf.get, reverse=True)
    return [all_chunks[i] for i in ranked]


# ==============================================================================
# BƯỚC 5b — Tải reranker (PATCHED: không .half(), bỏ token_type_ids, try/except)
# ==============================================================================
def load_reranker():
    """PATCH: Thử AITeamVN trước, fallback PhoRanker. Không .half(). Bắt lỗi CUDA assert."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    for model_name in [RERANKER_PRIMARY, RERANKER_FALLBACK]:
        for attempt in range(2):
            try:
                print(f"  Loading reranker {model_name} ...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                # PATCH: KHÔNG .half() cho RoBERTa-based -> tránh CUDA assert
                print(f"  Reranker ready on {next(model.parameters()).device} (fp32)")
                model.eval()
                return model, tokenizer
            except Exception as e:
                if attempt == 0:
                    print(f"    Lỗi lần 1: {e}, thử lại ...")
                    time.sleep(3)
                    continue
                print(f"    Bỏ qua {model_name}: {e}")
                break

    print("  [CẢNH BÁO] Không tải được reranker -> dùng thẳng RRF.")
    return None, None


def rerank(question: str, candidates: list, reranker_model, reranker_tokenizer,
           max_candidates: int = 30, max_length: int = RERANKER_MAX_LENGTH) -> list:
    """PATCH: Bỏ token_type_ids, try/except bắt CUDA assert."""
    import torch
    if reranker_model is None or not candidates:
        return candidates
    subset = candidates[:max_candidates]
    device = next(reranker_model.parameters()).device
    pairs = [[question, c["text"]] for c in subset]
    try:
        with torch.no_grad():
            inputs = reranker_tokenizer(pairs, padding=True, truncation=True,
                                         return_tensors="pt", max_length=max_length)
            # PATCH: RoBERTa không dùng token_type_ids -> bỏ để tránh CUDA assert
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            inputs = inputs.to(device)
            scores = reranker_model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()
        order = np.argsort(-scores)
        reranked = [subset[i] for i in order]
        return reranked + candidates[max_candidates:]
    except Exception as e:
        # PATCH: Bắt mọi lỗi CUDA/assert/OOM -> skip rerank cho câu này
        err = str(e).lower()
        if "assert" in err or "out of memory" in err or "cuda" in err:
            print(f"  [Rerank skip] Lỗi CUDA/assert cho câu này -> dùng RRF.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return candidates
        raise


# ==============================================================================
# BƯỚC 5c — LLM Loader (OPTIONAL, tắt mặc định)
# ==============================================================================
def load_llm():
    """Tải Qwen 0.5B fp16. Không cần bitsandbytes. Tự động fallback nếu lỗi."""
    if not USE_LLM:
        return None, None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"  Loading LLM {LLM_MODEL} (fp16) ...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        model.eval()
        print(f"  LLM ready on {next(model.parameters()).device}")
        return model, tokenizer
    except Exception as e:
        print(f"  [WARNING] Không tải được LLM ({e}) -> dùng extractive.")
        return None, None


def build_prompt(question: str, chunks: list) -> str:
    context_parts = []
    for c in chunks:
        header = f"Điều {c['dieu_so']}"
        if c["loai_vb"]:
            header += f" {c['loai_vb']}"
        if c["so_hieu"]:
            header += f" số {c['so_hieu']}"
        context_parts.append(f"{header}:\n{c['text']}")
    context = "\n\n".join(context_parts)
    return (
        "Bạn là chuyên gia pháp luật Việt Nam. Dựa trên các điều luật được cung cấp, "
        "hãy trả lời câu hỏi chính xác, đầy đủ, súc tích. Luôn nêu rõ căn cứ pháp lý.\n\n"
        f"Các điều luật tham khảo:\n{context}\n\n"
        f"Câu hỏi: {question}\n\nTrả lời:"
    )


def generate_answer_llm(question: str, chunks: list, llm_model, llm_tokenizer) -> str:
    import torch
    prompt = build_prompt(question, chunks)
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            do_sample=(LLM_TEMPERATURE > 0),
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = llm_tokenizer.decode(generated, skip_special_tokens=True).strip()
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer


# ==============================================================================
# BƯỚC 6 — Sinh câu trả lời (giữ template + thêm LLM optional)
# ==============================================================================
_DIEU_PREFIX_STRIP_RE = re.compile(r"^\s*Điều\s+\d+[a-zđA-ZĐ]?\.?\s*", re.IGNORECASE)


def render_answer(selected_chunks: list, top_n: int) -> str:
    parts, seen = [], set()
    for c in selected_chunks:
        if c["id"] in seen or len(parts) >= top_n:
            continue
        seen.add(c["id"])
        loai_vb = c["loai_vb"] or "văn bản"
        so_hieu = c["so_hieu"] or ""
        dieu = c["dieu_so"]
        lead = (f"Căn cứ Điều {dieu} {loai_vb} {so_hieu} quy định như sau:"
                if dieu != "0" else f"Căn cứ {loai_vb} {so_hieu} quy định như sau:")
        body = _DIEU_PREFIX_STRIP_RE.sub("", c["text"], count=1) if dieu != "0" else c["text"]
        parts.append(f"{lead}\n{body}")
    return "\n\n".join(parts)


def answer_question(question: str, bm25, dense_model, dense_embeddings, all_chunks, top_n: int,
                     reranker_model=None, reranker_tokenizer=None,
                     llm_model=None, llm_tokenizer=None, use_llm: bool = False) -> str:
    ranked = rrf_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks)
    if not ranked:
        return "Không tìm thấy thông tin pháp lý cho câu hỏi này."
    if reranker_model is not None:
        ranked = rerank(question, ranked, reranker_model, reranker_tokenizer)

    if use_llm and llm_model is not None:
        try:
            return generate_answer_llm(question, ranked[:top_n], llm_model, llm_tokenizer)
        except Exception as e:
            print(f"    [LLM error] fallback extractive: {e}")
            return render_answer(ranked, top_n)
    return render_answer(ranked, top_n)


# ==============================================================================
# BƯỚC 7 — Dev-eval (giữ nguyên + thêm LLM)
# ==============================================================================
def try_dev_eval(bm25, dense_model, dense_embeddings, all_chunks, train_data,
                  reranker_model=None, reranker_tokenizer=None,
                  llm_model=None, llm_tokenizer=None) -> tuple:
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
        print(f"  Bỏ qua dev-eval (thiếu lib: {e}). -> top_n=3, no rerank.")
        return 3, False, False

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    random.seed(42)
    n_sample = min(DEV_EVAL_SAMPLE_SIZE, len(train_data))
    ids = random.sample(list(train_data.keys()), n_sample)

    configs = [("Extractive", False, False)]
    if reranker_model is not None:
        configs.append(("Extractive+Rerank", True, False))
    if llm_model is not None and USE_LLM:
        configs.append(("LLM+Rerank", True, True))

    best_cfg = ("Extractive", 3, False, False, -1.0)
    for label, use_rerank, use_llm in configs:
        print(f"  --- {label} ---")
        rr_m = reranker_model if use_rerank else None
        rr_t = reranker_tokenizer if use_rerank else None

        ranked_cache = {}
        for qid in ids:
            ranked = rrf_retrieve(train_data[qid]["question"], bm25, dense_model, dense_embeddings, all_chunks)
            if use_rerank and rr_m is not None and ranked:
                ranked = rerank(train_data[qid]["question"], ranked, rr_m, rr_t)
            ranked_cache[qid] = ranked

        for top_n in (1, 2, 3, 4, 5):
            ms, rs = [], []
            for qid in ids:
                ranked = ranked_cache[qid]
                if use_llm and llm_model is not None:
                    pred = generate_answer_llm(train_data[qid]["question"], ranked[:top_n], llm_model, llm_tokenizer)
                else:
                    pred = render_answer(ranked, top_n) if ranked else "Không tìm thấy thông tin."
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms) / len(ms), sum(rs) / len(rs)
            print(f"    top_n={top_n}  METEOR={m:.4f}  ROUGE-L={r:.4f}")
            if m > best_cfg[4]:
                best_cfg = (label, top_n, use_rerank, use_llm, m)

    label, best_n, best_use_rerank, best_use_llm, best_m = best_cfg
    print(f"  => chọn TOP_N_ANSWER={best_n}, rerank={best_use_rerank}, llm={best_use_llm} (METEOR={best_m:.4f})")
    return best_n, best_use_rerank, best_use_llm


def measure_retrieval_recall(bm25, dense_model, dense_embeddings, all_chunks, train_data,
                              train_positive, reranker_model=None, reranker_tokenizer=None,
                              sample_size: int = 300) -> None:
    if not train_positive:
        print("  [WARNING] Không có train_positive -> skip Recall@k.")
        return
    ids = [qid for qid in random.sample(list(train_positive.keys()),
                                          min(sample_size, len(train_positive)))]
    print(f"  Đo Recall@k trên {len(ids)} câu hỏi ...")
    ks = [1, 3, 5, 10, 30, 100]
    hits_rrf = {k: 0 for k in ks}
    hits_rerank = {k: 0 for k in ks} if reranker_model is not None else None

    for qid in ids:
        question = train_data[qid]["question"]
        pos_id = train_positive[qid]
        ranked = rrf_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks)
        ranked_ids = [c["id"] for c in ranked]
        for k in ks:
            if pos_id in ranked_ids[:k]:
                hits_rrf[k] += 1
        if reranker_model is not None and ranked:
            reranked = rerank(question, ranked, reranker_model, reranker_tokenizer)
            reranked_ids = [c["id"] for c in reranked]
            for k in ks:
                if pos_id in reranked_ids[:k]:
                    hits_rerank[k] += 1

    n = len(ids)
    print("  --- RRF (chưa rerank) ---")
    for k in ks:
        print(f"    Recall@{k:<3d} = {hits_rrf[k]}/{n} = {100*hits_rrf[k]/n:.1f}%")
    if hits_rerank is not None:
        print("  --- Sau rerank ---")
        for k in ks:
            print(f"    Recall@{k:<3d} = {hits_rerank[k]}/{n} = {100*hits_rerank[k]/n:.1f}%")


# ==============================================================================
# BƯỚC 8 — Đóng gói (giữ nguyên)
# ==============================================================================
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
    print(f"  OK — {out_zip} ({len(normalized)} câu trả lời)")


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    checkpoint("Bắt đầu")

    print("\n=== Bước 1: Chunk corpus ===")
    all_chunks = load_corpus(CONTEXTS_DIR)
    checkpoint("Xong chunking")

    print("\n=== Bước 2: BM25 index ===")
    tokenized = [tokenize_simple(f"{c.get('loai_vb','')} {c['text']}") for c in all_chunks]
    bm25 = BM25(tokenized)
    checkpoint("Xong BM25")

    print("\n=== Bước 3: Sinh nhãn từ train.json ===")
    with TRAIN_PATH.open(encoding="utf-8") as f:
        train_data = json.load(f)
    train_positive, chunk_by_id = build_train_pairs(train_data, all_chunks)
    print(f"  Positive pairs: {len(train_positive)}/{len(train_data)}")
    checkpoint("Xong nhãn")

    print("\n=== Bước 4: Fine-tune dense retriever ===")
    dense_model = finetune_or_load_dense(train_positive, train_data, chunk_by_id, all_chunks, bm25)
    checkpoint("Xong dense retriever")

    print("\n=== Bước 5: Encode corpus ===")
    dense_embeddings = encode_corpus(dense_model, all_chunks)
    checkpoint("Xong encode corpus")

    print("\n=== Bước 5b: Tải reranker ===")
    reranker_model, reranker_tokenizer = load_reranker()
    checkpoint("Xong tải reranker")

    print("\n=== Bước 5c: Tải LLM (optional) ===")
    llm_model, llm_tokenizer = load_llm()
    checkpoint("Xong tải LLM")

    print("\n=== Bước 5d: Đo Recall@k ===")
    measure_retrieval_recall(bm25, dense_model, dense_embeddings, all_chunks, train_data,
                              train_positive, reranker_model, reranker_tokenizer)
    checkpoint("Xong Recall@k")

    print("\n=== Bước 6: Dev-eval chọn config ===")
    top_n_answer, use_reranker, use_llm = try_dev_eval(
        bm25, dense_model, dense_embeddings, all_chunks, train_data,
        reranker_model, reranker_tokenizer, llm_model, llm_tokenizer
    )
    checkpoint("Xong dev-eval")

    print("\n=== Bước 7: Sinh câu trả lời cho public-official.json ===")
    with PUBLIC_PATH.open(encoding="utf-8") as f:
        questions = json.load(f)
    rr_model = reranker_model if use_reranker else None
    rr_tok = reranker_tokenizer if use_reranker else None
    llm_m = llm_model if use_llm else None
    llm_t = llm_tokenizer if use_llm else None

    answers = {}
    for i, (qid, item) in enumerate(questions.items()):
        answers[qid] = answer_question(
            item["question"], bm25, dense_model, dense_embeddings, all_chunks, top_n_answer,
            reranker_model=rr_model, reranker_tokenizer=rr_tok,
            llm_model=llm_m, llm_tokenizer=llm_t, use_llm=use_llm
        )
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(questions)}  ({elapsed()/60:.1f} phút)")
    n_empty = sum(1 for a in answers.values() if not a.strip())
    print(f"  Đã sinh {len(answers)} câu trả lời, {n_empty} câu rỗng")
    checkpoint("Xong sinh câu trả lời")

    print("\n=== Bước 8: Đóng gói submission.zip ===")
    build_submission(answers, set(questions.keys()), OUT_DIR / "submission.zip")
    checkpoint(f"XONG — tổng thờigian {elapsed()/60:.1f} phút")


if __name__ == "__main__":
    main()