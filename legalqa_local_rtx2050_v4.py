"""
legalqa_local.py — LegalQA (UIT DSC2026 Task 2), tối ưu cho RTX 2050 4GB VRAM.

CÁCH DÙNG: đặt file này cạnh train.json, public-official.json, selected-contexts/ (đúng
layout thư mục của bạn) rồi chạy:
    python legalqa_local.py
Output: submission.zip trong cùng thư mục. Cache/model tải về nằm trong `cache/`, checkpoint
2 dense encoder đã fine-tune nằm trong `checkpoints/` (2 thư mục riêng biệt, xem BẢN SỬA #8).

THƯ VIỆN CẦN CÀI (trong venv "env" của bạn):
    pip install numpy sentence-transformers datasets "accelerate>=1.1.0" nltk rouge_score tiktoken sentencepiece
Và BẮT BUỘC kiểm tra torch có nhận đúng GPU không TRƯỚC khi chạy (chạy thử):
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only')"
Nếu ra False dù máy có RTX 2050 — cài lại torch bản có CUDA (driver mới thì dùng cu130,
driver cũ hơn thì cu126), KHÔNG dùng `pip install torch` trần trên Windows (mặc định tải
bản CPU-only).

===============================================================================
BẢN SỬA #8 (kiến trúc "mạnh nhất" — 2 dense encoder, đồng bộ với bản Kaggle T4x2, thu gọn
cho 1 GPU 4GB) — theo phân tích chênh lệch điểm số, xem PHAN_TICH_KY_THUAT.md
===============================================================================
Bản #7 (1 bi-encoder 135M tự train) đã CHẠM TRẦN chất lượng của chính nó (~0,52 METEOR /
~0,48 ROUGE-L, ổn định qua nhiều lần chạy có seed) — vá vặt thêm (seed, warmup ablation)
không còn kéo điểm lên được nữa. Không có nhãn document-level của Task 1 (đã xác nhận với
người dùng — không tồn tại trong dataset hiện có), nên nâng cấp bằng cách TỰ train retriever
mạnh hơn hẳn từ dữ liệu Task 2 sẵn có, cùng công thức với bản Kaggle:

1. THAY 1 encoder 135M bằng 2 encoder khác họ, mạnh hơn nhiều: BAAI/bge-m3 (~568M, không
   cần tiền tố) + intfloat/multilingual-e5-large (~560M, CẦN tiền tố "query: "/"passage: ").
   Fusion RRF 3 kênh (BM25 + bge-m3-ft + e5-ft) thay vì 2 kênh.

2. Khác bản Kaggle (2 GPU 16GB, train SONG SONG qua subprocess): ở đây CHỈ 1 GPU 4GB nên
   train TUẦN TỰ (encoder A xong mới tới encoder B, trong CÙNG tiến trình — không cần
   subprocess vì không có gì để chia song song).

3. Model to hơn ~4 lần (568M/560M so với 135M cũ) nên VRAM CĂNG THẲNG hơn nhiều:
   - Batch khởi điểm hạ xuống (TRAIN_MINI_BATCH_SIZE 4→2, ENCODE_BATCH_SIZE 64→16) —
     OOM-backoff vẫn xử lý nếu cần thấp hơn nữa.
   - Giữ CẢ 3 model (bge-m3 + e5-large + reranker, mỗi cái ~1,1GB fp16) cùng lúc trên VRAM
     4GB ở giai đoạn phục vụ truy vấn (Bước 6/7) gần như chắc chắn OOM. Giải pháp: SAU khi
     encode xong corpus (Bước 5, cần GPU cho tốc độ), CHUYỂN 1 encoder (mặc định e5-large)
     SANG CPU cho phần còn lại — encode 1 câu hỏi ngắn lúc truy vấn (không phải encode
     nguyên corpus) rẻ tới mức CPU cũng đủ nhanh, đổi lại nhường VRAM cho reranker.

4. LƯU checkpoint (khác bản #5/#7 — "không lưu checkpoint"): train 2 encoder lớn trên 4GB
   có thể mất nhiều giờ, mất trắng khi crash/tắt máy giữa chừng là quá đắt so với trước.
   Checkpoint nằm trong `checkpoints/` (thư mục RIÊNG, không lẫn vào `cache/` vốn được coi
   là có thể xoá an toàn — `checkpoints/` thì KHÔNG nên xoá tuỳ tiện).
   THÊM: `REUSE_CHECKPOINT_IF_EXISTS=True` — nếu đã có checkpoint hợp lệ từ lần chạy trước,
   TỰ ĐỘNG tải lại thay vì train lại từ đầu (tiết kiệm hàng giờ cho các lần chạy sau chỉ để
   thử nghiệm phần compose/dev-eval).

Giữ nguyên 100% từ bản #7: SEED toàn cục + set_all_seeds(), USE_WARMUP (ablation
warmup.json), EXPERIMENT_LOG_PATH (sổ thí nghiệm), toàn bộ Bước 1-3 (chunk, BM25, sinh
nhãn), reranker VẪN zero-shot (chưa fine-tune — việc tiếp theo, xem TODO ở main()).

===============================================================================
QUYẾT ĐỊNH THIẾT KẾ CÒN GIỮ NGUYÊN CHO RÀNG BUỘC 4GB VRAM
===============================================================================
1. KHÔNG fine-tune/chạy LLM sinh câu trả lời (LoRA SFT/DPO) — dùng template extractive.
2. BM25 tự viết bằng numpy, chạy CPU, không tốn VRAM.
3. Fine-tune time-boxed dựa trên tốc độ ĐO THẬT (calib), không phải ước lượng.
===============================================================================
"""
from __future__ import annotations
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent  # script chạy cùng thư mục với train.json, v.v.

CACHE_DIR = HERE / "cache"                 # model tải về/nltk — coi là XOÁ ĐƯỢC AN TOÀN
CHECKPOINT_DIR = HERE / "checkpoints"      # BẢN SỬA #8: encoder đã fine-tune — KHÔNG xoá tuỳ tiện
HF_CACHE_DIR = CACHE_DIR / "hf"
NLTK_CACHE_DIR = CACHE_DIR / "nltk_data"
TRAINER_TMP_DIR = CACHE_DIR / "trainer_tmp"
for _d in (HF_CACHE_DIR, NLTK_CACHE_DIR, TRAINER_TMP_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # tránh treo do fork trên Windows
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
import re
import json
import math
import time
import random
import zipfile
from collections import defaultdict, Counter

import numpy as np

# ==============================================================================
# CONFIG
# ==============================================================================
CONTEXTS_DIR = HERE / "selected-contexts"
TRAIN_PATH = HERE / "train.json"
WARMUP_PATH = HERE / "warmup.json"     # tuỳ chọn — gộp thêm nếu tồn tại VÀ USE_WARMUP=True.
PUBLIC_PATH = HERE / "public-official.json"
OUT_DIR = HERE
EXPERIMENT_LOG_PATH = OUT_DIR / "experiment_log.jsonl"  # "sổ thí nghiệm" — 1 dòng JSON/lần chạy

SEED = 42
USE_WARMUP = True                  # đặt False để ablation: chỉ dùng train.json
REUSE_CHECKPOINT_IF_EXISTS = True  # BẢN SỬA #8: có checkpoint hợp lệ từ trước thì tải lại,
                                    # KHÔNG train lại từ đầu (tiết kiệm hàng giờ). Đặt False
                                    # để ép train lại (vd đổi USE_WARMUP, muốn so sánh thật).


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


TIME_BUDGET_SEC = 6 * 3600          # BẢN SỬA #8: nâng từ 3h lên 6h — train 2 encoder lớn hơn
                                     # (568M/560M so với 135M) tuần tự trên 1 GPU 4GB CHẬM hơn
                                     # đáng kể so với bản #7, cần trần rộng hơn để không bị cắt
                                     # ngang Bước 4 giữa chừng.
FINETUNE_TIME_BUDGET_SEC = 150 * 60  # ngân sách MỖI encoder (không phải tổng cả 2)
MIN_TRAIN_PAIRS = 50
MAX_TRAIN_EXAMPLES = 3000

BASE_DENSE_MODEL_A = "BAAI/bge-m3"                        # không cần tiền tố
BASE_DENSE_MODEL_B = "intfloat/multilingual-e5-large"     # CẦN tiền tố "query: "/"passage: " —
                                                            # bẫy kinh điển: quên tiền tố thì
                                                            # recall tụt mà KHÔNG lỗi nào báo.
DENSE_MAX_SEQ_LEN = 256
TRAIN_BATCH_SIZE = 32               # batch HIỆU DỤNG — không đổi khi OOM, xem CachedMNRL.
TRAIN_MINI_BATCH_SIZE = 2           # BẢN SỬA #8: hạ từ 4 xuống 2 — model to gấp ~4 lần bản
                                     # #7 (568M/560M so với 135M), khởi điểm phải dè dặt hơn.
N_NEG_PER_ROW = 2
ENCODE_BATCH_SIZE = 16              # BẢN SỬA #8: hạ từ 64 xuống 16 — cùng lý do trên.

TOP_K_RETRIEVE = 100
DEV_EVAL_SAMPLE_SIZE = 300
TOP_K_RERANK = 5                    # trần trên cho top_n tĩnh VÀ trần của adaptive-k

_START_TIME = time.time()


def elapsed() -> float:
    return time.time() - _START_TIME


def remaining() -> float:
    return TIME_BUDGET_SEC - elapsed()


def checkpoint(label: str) -> None:
    print(f"[{elapsed()/60:6.1f} phút] {label}  (còn lại ~{remaining()/60:.1f} phút trong ngân sách)")


# ==============================================================================
# BƯỚC 1 — Chunk corpus theo Điều  (KHÔNG đổi so với bản #7)
# ==============================================================================
DIEU_RE = re.compile(r"^[ \t]*Điều\s+(\d+)[a-zđA-ZĐ]?[\.\s]", re.MULTILINE)
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
    matches = list(DIEU_RE.finditer(passage))
    if not matches:
        return [{"id": f"{doc_id}_0", "dieu_so": "0", "loai_vb": "", "so_hieu": "", "text": passage.strip()}]
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(passage)
        dieu = m.group(1)
        chunks.append({"id": f"{doc_id}_{dieu}_{i}", "dieu_so": dieu, "loai_vb": "", "so_hieu": "",
                        "text": passage[start:end].strip()})
    return chunks


def load_corpus(contexts_dir: Path) -> list:
    if not contexts_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy {contexts_dir} — kiểm tra lại layout thư mục.")
    files = sorted(contexts_dir.glob("context_*.json"))
    if not files:
        nested = contexts_dir / "selected-contexts"
        if nested.exists():
            files = sorted(nested.glob("context_*.json"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy context_*.json trong {contexts_dir}")

    all_chunks, n_no_dieu = [], 0
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
        if len(chunks) == 1 and chunks[0]["dieu_so"] == "0":
            n_no_dieu += 1
        loai_vb, so_hieu = extract_vb_info(passage)
        for c in chunks:
            c["loai_vb"], c["so_hieu"] = loai_vb, so_hieu
        all_chunks.extend(chunks)

    pct = round(100 * (1 - n_no_dieu / len(files)), 2) if files else 0.0
    print(f"  {len(files)} văn bản -> {len(all_chunks)} chunk. {pct}% có cấu trúc Điều.")
    if pct < 95.0:
        print("  [CẢNH BÁO] < 95% — kiểm tra vài context_*.json thật, có thể cần chỉnh DIEU_RE.")
    return all_chunks


# ==============================================================================
# BƯỚC 2 — BM25 tự viết bằng numpy  (KHÔNG đổi)
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
# BƯỚC 3 — Sinh nhãn từ citation trong train.json  (KHÔNG đổi)
# ==============================================================================
DIEU_CITATION_RE = re.compile(r"Điều\s+(\d+)\s*[a-zđA-ZĐ]?\b")


def extract_citations(answer: str) -> list:
    out = []
    for m in DIEU_CITATION_RE.finditer(answer):
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


# ==============================================================================
# BƯỚC 4 — Fine-tune 2 dense encoder TUẦN TỰ trên 1 GPU (BẢN SỬA #8)
# ==============================================================================
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
            print(f"    _build_training_rows: {i+1}/{n}  ({elapsed()/60:.1f} phút)")
    return rows


def enable_cuda_perf() -> None:
    import torch
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [GPU] {torch.cuda.get_device_name(0)} — {total_gb:.1f}GB VRAM, dùng toàn bộ "
              f"(không đặt trần nhân tạo, chỉ lùi batch khi OOM thật xảy ra).")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _add_prefix(rows: list, query_prefix: str, passage_prefix: str) -> list:
    if not query_prefix and not passage_prefix:
        return rows
    out = []
    for r in rows:
        r2 = dict(r)
        r2["anchor"] = query_prefix + r["anchor"]
        for k in r:
            if k.startswith("positive") or k.startswith("negative"):
                r2[k] = passage_prefix + r[k]
        out.append(r2)
    return out


def finetune_one_encoder(base_model: str, rows: list, device: str, query_prefix: str,
                          passage_prefix: str, checkpoint_out: Path) -> tuple:
    """Fine-tune MỘT SentenceTransformer, trả về (model, info_dict). KHÔNG train nếu
    REUSE_CHECKPOINT_IF_EXISTS=True và đã có checkpoint hợp lệ từ trước — tải lại thay vì
    train lại từ đầu (BẢN SỬA #8, tiết kiệm hàng giờ cho các lần chạy sau)."""
    import torch
    from sentence_transformers import SentenceTransformer

    meta_path = checkpoint_out.parent / (checkpoint_out.name + "_meta.json")
    if REUSE_CHECKPOINT_IF_EXISTS and checkpoint_out.exists() and meta_path.exists():
        try:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            model = SentenceTransformer(str(checkpoint_out), device=device)
            model.max_seq_length = DENSE_MAX_SEQ_LEN
            print(f"  [{base_model}] Tái dùng checkpoint có sẵn tại {checkpoint_out} "
                  f"(REUSE_CHECKPOINT_IF_EXISTS=True) — KHÔNG train lại.")
            meta["reused_checkpoint"] = True
            return model, meta
        except Exception as e:
            print(f"  [{base_model}] Checkpoint có nhưng tải lỗi ({e}) -> train lại từ đầu.")

    model = SentenceTransformer(base_model, device=device)
    model.max_seq_length = DENSE_MAX_SEQ_LEN
    info = {"used_finetune": False, "max_steps": 0, "mini_batch_size_final": None,
            "calib_time_s": None, "reused_checkpoint": False}

    try:
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
        import accelerate  # noqa: F401
        if tuple(map(int, accelerate.__version__.split(".")[:2])) < (1, 1):
            raise ImportError(f"accelerate {accelerate.__version__} quá cũ, cần >= 1.1.0")
    except ImportError as e:
        print(f"  [{base_model}] [THIẾU PACKAGE] {e} -> dùng zero-shot, không fine-tune.")
        info["reason"] = f"thiếu package: {e}"
        return model, info

    prefixed_rows = _add_prefix(rows, query_prefix, passage_prefix)
    dataset = Dataset.from_list(prefixed_rows)

    batch_size, mini_batch_size = TRAIN_BATCH_SIZE, TRAIN_MINI_BATCH_SIZE
    for attempt in range(4):
        try:
            loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
            calib_steps = min(10, max(1, len(dataset) // batch_size))
            calib_args = SentenceTransformerTrainingArguments(
                output_dir=str(TRAINER_TMP_DIR), max_steps=calib_steps,
                per_device_train_batch_size=batch_size, logging_steps=calib_steps + 1,
                save_strategy="no", report_to=[], disable_tqdm=True, fp16=(device == "cuda"),
            )
            calib_start = time.time()
            print(f"  [{base_model}] Calib training (batch hiệu dụng={batch_size}, "
                  f"mini_batch={mini_batch_size})...")
            SentenceTransformerTrainer(model=model, args=calib_args, train_dataset=dataset, loss=loss).train()
            calib_time = (time.time() - calib_start) / calib_steps

            budget_left = min(remaining() - 5 * 60, FINETUNE_TIME_BUDGET_SEC - (time.time() - calib_start))
            max_steps = max(0, int(budget_left / max(calib_time, 1e-6)))
            max_steps = min(max_steps, (len(dataset) // batch_size) * 8)
            print(f"  [{base_model}] Calib ~{calib_time:.2f}s/step, ngân sách còn "
                  f"~{budget_left/60:.1f} phút -> {max_steps} step.")

            if max_steps > 0:
                args = SentenceTransformerTrainingArguments(
                    output_dir=str(TRAINER_TMP_DIR), max_steps=max_steps,
                    per_device_train_batch_size=batch_size, learning_rate=2e-5,
                    warmup_steps=0.05, lr_scheduler_type="cosine",
                    logging_steps=max(1, max_steps // 20), save_strategy="no", report_to=[],
                    fp16=(device == "cuda"),
                )
                SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss).train()
            break
        except Exception as e:
            # BẢN SỬA (log lỗi thật trên bản Kaggle: torch.AcceleratorError OOM KHÔNG phải
            # RuntimeError trên PyTorch bản mới -- except RuntimeError không bắt được, sập cả
            # tiến trình thay vì lùi batch. Bắt rộng theo NỘI DUNG thông điệp thay vì theo class
            # exception, vì class exception cho OOM đã đổi giữa các bản PyTorch): torch.cuda.
            # empty_cache() cũng được bọc try/except riêng -- khi VRAM cạn quá sâu, chính lệnh
            # dọn cache có thể tự OOM (đã gặp thật), không để nó làm sập nốt vòng lặp lùi batch.
            if "out of memory" in str(e).lower() and mini_batch_size > 1:
                print(f"  [{base_model}] [CUDA OOM] mini_batch_size={mini_batch_size} -> thử "
                      f"{mini_batch_size // 2} (batch hiệu dụng {batch_size} GIỮ NGUYÊN).")
                if device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                mini_batch_size = max(1, mini_batch_size // 2)
                continue
            raise

    model = model.to(device)
    model.save_pretrained(str(checkpoint_out))
    info.update({"used_finetune": True, "max_steps": max_steps,
                 "mini_batch_size_final": mini_batch_size, "calib_time_s": round(calib_time, 4)})
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(info, f)
    print(f"  [{base_model}] Checkpoint đã lưu -> {checkpoint_out}")
    return model, info


def finetune_dense_pair(train_positive, train_data, chunk_by_id, all_chunks, bm25):
    """Train (hoặc tái dùng checkpoint) CẢ 2 encoder, TUẦN TỰ trên 1 GPU. Trả về
    (dense_channels, finetune_info) — dense_channels chưa có "embeddings" (điền ở Bước 5)."""
    import torch

    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    if cuda_ok:
        print(f"  Device: cuda ({torch.cuda.get_device_name(0)})")
        enable_cuda_perf()
    else:
        print("  [CẢNH BÁO] torch.cuda.is_available()=False — xem hướng dẫn cài torch+CUDA "
              "ở đầu file. Đang tiếp tục chạy CPU — chậm hơn nhiều.")

    finetune_info = {"n_pairs_available": len(train_positive), "n_pairs_used": 0, "models": {}}
    use_finetune = len(train_positive) >= MIN_TRAIN_PAIRS and remaining() > 20 * 60

    from sentence_transformers import SentenceTransformer
    if not use_finetune:
        reason = (f"{len(train_positive)} positive pairs < {MIN_TRAIN_PAIRS}"
                  if len(train_positive) < MIN_TRAIN_PAIRS else "hết ngân sách thời gian")
        print(f"  {reason} -> dùng zero-shot cho cả 2 encoder.")
        m_a = SentenceTransformer(BASE_DENSE_MODEL_A, device=device); m_a.max_seq_length = DENSE_MAX_SEQ_LEN
        m_b = SentenceTransformer(BASE_DENSE_MODEL_B, device=device); m_b.max_seq_length = DENSE_MAX_SEQ_LEN
        finetune_info["reason"] = reason
        dense_channels = [
            {"name": "bge-m3", "model": m_a, "embeddings": None, "query_prefix": "", "passage_prefix": ""},
            {"name": "e5-large", "model": m_b, "embeddings": None, "query_prefix": "query: ", "passage_prefix": "passage: "},
        ]
        return dense_channels, finetune_info

    train_positive_used = train_positive
    if len(train_positive) > MAX_TRAIN_EXAMPLES:
        sampled_qids = random.sample(list(train_positive.keys()), MAX_TRAIN_EXAMPLES)
        train_positive_used = {qid: train_positive[qid] for qid in sampled_qids}
        print(f"  Có {len(train_positive)} positive pairs, lấy mẫu {MAX_TRAIN_EXAMPLES} "
              f"(tái lập được nhờ SEED={SEED}).")
    finetune_info["n_pairs_used"] = len(train_positive_used)

    print("  Đang tạo training rows (dùng chung cho cả 2 encoder)...")
    rows = _build_training_rows(train_positive_used, train_data, chunk_by_id, all_chunks, bm25)

    specs = [
        {"name": "bge-m3", "base_model": BASE_DENSE_MODEL_A, "query_prefix": "", "passage_prefix": "",
         "out": CHECKPOINT_DIR / "bge-m3-ft"},
        {"name": "e5-large", "base_model": BASE_DENSE_MODEL_B, "query_prefix": "query: ", "passage_prefix": "passage: ",
         "out": CHECKPOINT_DIR / "e5-large-ft"},
    ]
    dense_channels = []
    for spec in specs:
        print(f"\n  --- Fine-tune {spec['name']} ({spec['base_model']}) ---")
        m, info = finetune_one_encoder(spec["base_model"], rows, device, spec["query_prefix"],
                                        spec["passage_prefix"], spec["out"])
        finetune_info["models"][spec["name"]] = info
        dense_channels.append({"name": spec["name"], "model": m, "embeddings": None,
                                "query_prefix": spec["query_prefix"], "passage_prefix": spec["passage_prefix"]})
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    return dense_channels, finetune_info


# ==============================================================================
# BƯỚC 5 — Encode toàn bộ corpus cho CẢ 2 encoder, TUẦN TỰ (mỗi lần dùng hết 1 GPU)
# ==============================================================================
def encode_corpus_for_channel(ch: dict, all_chunks: list) -> None:
    """Điền `ch["embeddings"]` tại chỗ. Dùng GPU (nếu có) cho tốc độ — khác với lúc phục vụ
    truy vấn (Bước 6/7), nơi 1 trong 2 encoder sẽ bị chuyển sang CPU để nhường VRAM cho
    reranker (xem offload_one_channel_to_cpu())."""
    import torch
    model = ch["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    texts_raw = [c["text"] for c in all_chunks]
    texts = [ch["passage_prefix"] + t for t in texts_raw] if ch["passage_prefix"] else texts_raw

    batch_size = ENCODE_BATCH_SIZE
    while True:
        try:
            emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                show_progress_bar=True, normalize_embeddings=True, device=device)
            break
        except Exception as e:
            # BẢN SỬA: cùng lý do ở finetune_one_encoder() phía trên -- bắt rộng theo nội dung
            # thông điệp, bọc riêng empty_cache().
            if "out of memory" in str(e).lower() and batch_size > 1:
                print(f"    [CUDA OOM] encode batch_size={batch_size} -> thử {batch_size // 2}")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                batch_size = max(1, batch_size // 2)
                continue
            raise
    ch["model"] = model
    ch["embeddings"] = emb


def offload_one_channel_to_cpu(dense_channels: list, index: int = -1) -> None:
    """BẢN SỬA #8: chuyển 1 encoder sang CPU trước khi tải reranker (Bước 5b) — giữ CẢ 3
    model (2 encoder + reranker, mỗi cái ~1,1GB fp16) trên VRAM 4GB cùng lúc gần như chắc
    chắn OOM. Encode 1 câu hỏi ngắn lúc truy vấn (không phải encode nguyên corpus 160k
    chunk) đủ rẻ để CPU vẫn nhanh — corpus embeddings (numpy, đã tính xong ở Bước 5) không
    bị ảnh hưởng, chúng luôn nằm ở RAM chứ không phải VRAM."""
    import torch
    if not torch.cuda.is_available():
        return
    ch = dense_channels[index]
    ch["model"] = ch["model"].to("cpu").float()
    torch.cuda.empty_cache()
    print(f"  [VRAM] Chuyển encoder '{ch['name']}' sang CPU để nhường chỗ cho reranker "
          f"(chỉ ảnh hưởng tốc độ encode CÂU HỎI lúc truy vấn, không ảnh hưởng corpus "
          f"embeddings đã tính xong).")


def load_reranker():
    """Dùng ZERO-SHOT `AITeamVN/Vietnamese_Reranker` — CHƯA fine-tune ở bản này, xem TODO
    ở main(). Trả về (None, None) nếu tải lỗi — answer_question() tự bỏ qua rerank."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    for attempt in range(2):
        try:
            print(f"  Đang tải reranker AITeamVN/Vietnamese_Reranker"
                  f"{' — thử lại lần 2' if attempt else ''}...")
            tokenizer = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Reranker")
            model = AutoModelForSequenceClassification.from_pretrained("AITeamVN/Vietnamese_Reranker")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            if device == "cuda":
                model = model.half()
            model.eval()
            print(f"  Reranker sẵn sàng trên {next(model.parameters()).device}"
                  f"{' (fp16)' if device == 'cuda' else ''}.")
            return model, tokenizer
        except Exception as e:
            if attempt == 0:
                print(f"  [Lần 1 lỗi: {e}] thử lại sau 5s...")
                time.sleep(5)
                continue
            print(f"  [CẢNH BÁO] Không tải được reranker sau 2 lần thử ({e}) -> bỏ qua rerank.")
            return None, None


RERANK_SUBBATCH = 24
# Xem BẢN SỬA #6 (file lịch sử cũ): xử lý theo LÔ NHỎ thay vì nhồi hết vào 1 forward — điểm
# số mỗi cặp KHÔNG đổi, chỉ đổi tốc độ (tránh nghẽn băng thông bộ nhớ trên GPU nhỏ).


def rerank(question: str, candidates: list, reranker_model, reranker_tokenizer,
           max_candidates: int = TOP_K_RETRIEVE, max_length: int = 1024, sub_batch: int = RERANK_SUBBATCH):
    import torch
    if reranker_model is None or not candidates:
        return candidates, None
    subset = candidates[:max_candidates]
    device = next(reranker_model.parameters()).device
    pairs = [[question, c["text"]] for c in subset]
    scores = np.empty(len(pairs), dtype=np.float32)
    bs, i = max(1, sub_batch), 0
    while i < len(pairs):
        batch = pairs[i:i + bs]
        try:
            with torch.no_grad():
                inputs = reranker_tokenizer(batch, padding=True, truncation=True,
                                             return_tensors="pt", max_length=max_length).to(device)
                out = reranker_model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()
            scores[i:i + len(batch)] = out
            i += bs
        except Exception as e:
            # BẢN SỬA: cùng lý do ở finetune_one_encoder() -- bắt rộng theo nội dung thông
            # điệp, bọc riêng empty_cache().
            if "out of memory" in str(e).lower() and bs > 1:
                print(f"  [CUDA OOM rerank] sub_batch={bs} -> thử {bs // 2}")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                bs = max(1, bs // 2)
                continue
            if "out of memory" in str(e).lower():
                print("  [CUDA OOM lúc rerank] bỏ qua rerank cho câu hỏi này, dùng thứ hạng RRF.")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                return candidates, None
            raise
    order = np.argsort(-scores)
    reranked = [subset[i2] for i2 in order]
    sorted_scores = scores[order]
    return reranked + candidates[max_candidates:], sorted_scores


def adaptive_k_cutoff(scores, min_k: int = 1, max_k: int = TOP_K_RERANK, search_window: int = 15) -> int:
    """Adaptive-k (Taguchi et al. 2025, arXiv:2506.08479)."""
    if scores is None or len(scores) == 0:
        return min_k
    n = min(len(scores), search_window)
    if n <= 1:
        return min_k
    gaps = [scores[i] - scores[i + 1] for i in range(n - 1)]
    k_star = int(np.argmax(gaps)) + 1
    return max(min_k, min(k_star, max_k))


def rrf_retrieve(question: str, bm25: BM25, dense_channels: list, all_chunks: list, top_k: int = TOP_K_RETRIEVE):
    """RRF fusion N kênh: BM25 + mỗi encoder trong `dense_channels`. Tiền tố query PHẢI
    khớp tiền tố đã dùng lúc encode corpus (Bước 5) — bẫy đã ghi ở CONFIG/BẢN SỬA #8."""
    bm25_ranked = bm25.top_k(tokenize_simple(question), top_k)
    rank_maps = [{idx: r for r, idx in enumerate(bm25_ranked)}]
    all_idx = set(bm25_ranked)
    for ch in dense_channels:
        q_text = ch["query_prefix"] + question if ch["query_prefix"] else question
        q_emb = ch["model"].encode([q_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = ch["embeddings"] @ q_emb
        ranked_ch = list(np.argsort(-scores)[:top_k])
        rank_maps.append({idx: r for r, idx in enumerate(ranked_ch)})
        all_idx |= set(ranked_ch)
    rrf = {i: sum(1 / (60 + rm.get(i, top_k + 1)) for rm in rank_maps) for i in all_idx}
    ranked = sorted(rrf, key=rrf.get, reverse=True)
    return [all_chunks[i] for i in ranked]


# ==============================================================================
# BƯỚC 6 — Sinh câu trả lời
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


def answer_question(question: str, bm25, dense_channels: list, all_chunks: list, top_n: int,
                     reranker_model=None, reranker_tokenizer=None, use_adaptive_k: bool = False) -> str:
    ranked = rrf_retrieve(question, bm25, dense_channels, all_chunks)
    if not ranked:
        return "Không tìm thấy thông tin pháp lý cho câu hỏi này."
    scores = None
    if reranker_model is not None:
        ranked, scores = rerank(question, ranked, reranker_model, reranker_tokenizer)
    n = adaptive_k_cutoff(scores) if (use_adaptive_k and scores is not None) else top_n
    return render_answer(ranked, n)


# ==============================================================================
# BƯỚC 6b — Dev-eval (METEOR/ROUGE-L) + Recall@k gộp chung
# ==============================================================================
def try_dev_eval(bm25, dense_channels, all_chunks, train_data, train_positive,
                  reranker_model=None, reranker_tokenizer=None) -> tuple:
    try:
        import nltk
        if str(NLTK_CACHE_DIR) not in nltk.data.path:
            nltk.data.path.insert(0, str(NLTK_CACHE_DIR))
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True, download_dir=str(NLTK_CACHE_DIR))
            nltk.download("omw-1.4", quiet=True, download_dir=str(NLTK_CACHE_DIR))
        from nltk.translate.meteor_score import meteor_score
        from rouge_score import rouge_scorer
    except Exception as e:
        print(f"  Bỏ qua dev-eval (thiếu nltk/rouge_score: {e}). Dùng TOP_N_ANSWER=3, không rerank.")
        return 3, False, False, {"meteor": None, "rouge_l": None, "recall_at_k": {}, "n_dev": 0}

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    random.seed(SEED)  # re-seed CỐ Ý — giữ 300 câu dev-eval LUÔN CỐ ĐỊNH bất kể cấu hình khác.
    n_sample = min(DEV_EVAL_SAMPLE_SIZE, len(train_data))
    ids = random.sample(list(train_data.keys()), n_sample)
    recall_ids = [q for q in ids if q in train_positive]
    print(f"  Mẫu dev-eval: {len(ids)} câu ({len(recall_ids)} câu có citation resolve được).")

    configs = [("BM25+dense (không rerank)", None, None)]
    if reranker_model is not None:
        configs.append(("BM25+dense+rerank", reranker_model, reranker_tokenizer))

    ks = [1, 3, 5, 10, 30, 100]
    best_n, best_m, best_r, best_use_rerank, best_use_adaptive = 3, -1.0, None, False, False
    recall_at_k_by_label = {}
    for label, rr_model, rr_tok in configs:
        print(f"  --- {label} ---")
        ranked_cache, scores_cache = {}, {}
        t0 = time.time()
        for i, qid in enumerate(ids):
            item = train_data[qid]
            ranked = rrf_retrieve(item["question"], bm25, dense_channels, all_chunks)
            scores = None
            if rr_model is not None and ranked:
                ranked, scores = rerank(item["question"], ranked, rr_model, rr_tok)
            ranked_cache[qid] = ranked
            scores_cache[qid] = scores
            if (i + 1) % 50 == 0 or (i + 1) == len(ids):
                print(f"    retrieval+rerank {i+1}/{len(ids)} ... {time.time()-t0:.0f}s "
                      f"({elapsed()/60:.1f} phút)", flush=True)

        if recall_ids:
            hits = {k: 0 for k in ks}
            for qid in recall_ids:
                ranked_ids = [c["id"] for c in ranked_cache[qid]]
                pos_id = train_positive[qid]
                for k in ks:
                    if pos_id in ranked_ids[:k]:
                        hits[k] += 1
            nr = len(recall_ids)
            print(f"  Recall@k ({label}, n={nr}):")
            for k in ks:
                print(f"    Recall@{k:<3d} = {hits[k]}/{nr} = {100*hits[k]/nr:.1f}%")
            recall_at_k_by_label[label] = {str(k): round(hits[k] / nr, 4) for k in ks}

        for top_n in range(1, TOP_K_RERANK + 1):
            ms, rs = [], []
            for qid in ids:
                ranked = ranked_cache[qid]
                pred = render_answer(ranked, top_n) if ranked else "Không tìm thấy thông tin pháp lý cho câu hỏi này."
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms) / len(ms), sum(rs) / len(rs)
            print(f"    top_n={top_n}  METEOR={m:.4f}  ROUGE-L={r:.4f}  (n={len(ids)})")
            if m > best_m:
                best_m, best_r, best_n = m, r, top_n
                best_use_rerank, best_use_adaptive = (rr_model is not None), False

        if rr_model is not None:
            ms, rs = [], []
            for qid in ids:
                ranked, scores = ranked_cache[qid], scores_cache[qid]
                k = adaptive_k_cutoff(scores) if ranked else 1
                pred = render_answer(ranked, k) if ranked else "Không tìm thấy thông tin pháp lý cho câu hỏi này."
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms) / len(ms), sum(rs) / len(rs)
            print(f"    adaptive-k       METEOR={m:.4f}  ROUGE-L={r:.4f}  (n={len(ids)})")
            if m > best_m:
                best_m, best_r, best_use_rerank, best_use_adaptive = m, r, True, True

    print(f"  => chọn TOP_N_ANSWER={best_n}, dùng reranker={best_use_rerank}, "
          f"dùng adaptive-k={best_use_adaptive} (METEOR={best_m:.4f})")
    eval_info = {"meteor": round(best_m, 4), "rouge_l": (round(best_r, 4) if best_r is not None else None),
                 "recall_at_k": recall_at_k_by_label, "n_dev": len(ids)}
    return best_n, best_use_rerank, best_use_adaptive, eval_info


def build_submission(answers: dict, expected_ids: set, out_zip: Path) -> None:
    errors = []
    got = set(answers.keys())
    if got != expected_ids:
        errors.append(f"Key lệch: thiếu {len(expected_ids-got)}, thừa {len(got-expected_ids)}")
    for qid, ans in answers.items():
        if not isinstance(ans, str) or not ans.strip():
            errors.append(f"[{qid}] answer rỗng hoặc không phải string")
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
    print(f"  OK — {out_zip} ({len(normalized)} câu trả lời, đã kiểm tra lại từ đĩa)")


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    set_all_seeds(SEED)
    checkpoint("Bắt đầu")
    print(f"  Cache: {CACHE_DIR} · Checkpoint: {CHECKPOINT_DIR}")
    print(f"  SEED={SEED} · USE_WARMUP={USE_WARMUP} · REUSE_CHECKPOINT_IF_EXISTS={REUSE_CHECKPOINT_IF_EXISTS}")

    print("\n=== Bước 1: Chunk corpus ===")
    all_chunks = load_corpus(CONTEXTS_DIR)
    checkpoint("Xong chunking")

    print("\n=== Bước 2: BM25 index ===")
    tokenized = [tokenize_simple(f"{c.get('loai_vb','')} {c['text']}") for c in all_chunks]
    bm25 = BM25(tokenized)
    checkpoint("Xong BM25 index")

    print("\n=== Bước 3: Sinh nhãn từ train.json" + (" + warmup.json" if USE_WARMUP else "") + " ===")
    with TRAIN_PATH.open(encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"  train.json: {len(train_data)} câu")

    train_data_for_pairs = dict(train_data)
    n_warmup_used = 0
    if USE_WARMUP and WARMUP_PATH.exists():
        try:
            with WARMUP_PATH.open(encoding="utf-8") as f:
                warmup_data = json.load(f)
            n_bad_type = 0
            for qid, item in warmup_data.items():
                if not isinstance(item, dict):
                    n_bad_type += 1
                    continue
                q, a = item.get("question"), item.get("answer")
                if isinstance(q, str) and isinstance(a, str):
                    train_data_for_pairs[f"warmup_{qid}"] = {"question": q, "answer": a}
                    n_warmup_used += 1
                else:
                    n_bad_type += 1
            print(f"  warmup.json: {len(warmup_data)} câu, {n_warmup_used} câu đúng schema -> gộp thêm"
                  + (f", {n_bad_type} câu sai kiểu -> bỏ qua." if n_bad_type else "."))
        except Exception as e:
            print(f"  [CẢNH BÁO] Có {WARMUP_PATH} nhưng đọc lỗi ({e}) -> bỏ qua.")
    elif USE_WARMUP:
        print(f"  USE_WARMUP=True nhưng không thấy {WARMUP_PATH} -> chỉ dùng train.json.")
    else:
        print("  USE_WARMUP=False -> chỉ dùng train.json.")

    train_positive, chunk_by_id = build_train_pairs(train_data_for_pairs, all_chunks)
    print(f"  Positive pairs: {len(train_positive)}/{len(train_data_for_pairs)}")
    checkpoint("Xong sinh nhãn")

    print("\n=== Bước 4: Fine-tune 2 dense encoder (tuần tự, 1 GPU) ===")
    dense_channels, finetune_info = finetune_dense_pair(train_positive, train_data_for_pairs,
                                                          chunk_by_id, all_chunks, bm25)
    checkpoint("Xong Bước 4 (2 dense encoder)")

    print("\n=== Bước 5: Encode toàn bộ corpus cho cả 2 encoder (tuần tự) ===")
    for ch in dense_channels:
        t0 = time.time()
        print(f"  [{ch['name']}] encode {len(all_chunks)} chunk"
              + (f' (tiền tố "{ch["passage_prefix"]}")' if ch["passage_prefix"] else "") + " ...")
        encode_corpus_for_channel(ch, all_chunks)
        print(f"    -> {ch['embeddings'].shape}, {time.time()-t0:.0f}s")
    checkpoint("Xong encode corpus")

    # BẢN SỬA #8: nhường VRAM cho reranker — xem docstring offload_one_channel_to_cpu().
    offload_one_channel_to_cpu(dense_channels, index=-1)

    print("\n=== Bước 5b: Tải reranker (zero-shot — CHƯA fine-tune, xem TODO cuối file) ===")
    reranker_model, reranker_tokenizer = load_reranker()
    checkpoint("Xong tải reranker")

    print("\n=== Bước 6: Dev-eval chọn TOP_N_ANSWER + đo Recall@k ===")
    top_n_answer, use_reranker, use_adaptive, eval_info = try_dev_eval(
        bm25, dense_channels, all_chunks, train_data, train_positive,
        reranker_model, reranker_tokenizer)
    checkpoint("Xong dev-eval + Recall@k")

    print("\n=== Bước 7: Sinh câu trả lời cho public-official.json ===")
    with PUBLIC_PATH.open(encoding="utf-8") as f:
        questions = json.load(f)
    rr_model = reranker_model if use_reranker else None
    rr_tok = reranker_tokenizer if use_reranker else None
    answers = {}
    for i, (qid, item) in enumerate(questions.items()):
        answers[qid] = answer_question(item["question"], bm25, dense_channels, all_chunks,
                                        top_n_answer, reranker_model=rr_model,
                                        reranker_tokenizer=rr_tok, use_adaptive_k=use_adaptive)
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(questions)}  ({elapsed()/60:.1f} phút)")
    n_empty = sum(1 for a in answers.values() if not a.strip())
    print(f"  Đã sinh {len(answers)} câu trả lời, {n_empty} câu rỗng")
    checkpoint("Xong sinh câu trả lời")

    print("\n=== Bước 8: Đóng gói submission.zip ===")
    build_submission(answers, set(questions.keys()), OUT_DIR / "submission.zip")
    checkpoint(f"XONG — tổng thời gian {elapsed()/60:.1f} phút (trần an toàn {TIME_BUDGET_SEC/3600:.0f} giờ)")

    # TODO (lever tiếp theo, xem PHAN_TICH_KY_THUAT.md): reranker vẫn zero-shot. Sau khi xác
    # nhận Recall@k của bản 2-encoder này thật sự cao hơn bản 1-encoder cũ, fine-tune riêng
    # reranker (cùng công thức CachedMNRL/cross-encoder trên nhãn citation Điều-level đã có
    # sẵn ở train_positive) là việc đáng đầu tư tiếp theo.
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "use_warmup": USE_WARMUP,
        "n_warmup_used": n_warmup_used, "hardware": "local (.py, 1 GPU)",
        "n_train_pairs_available": finetune_info["n_pairs_available"],
        "n_train_pairs_used": finetune_info["n_pairs_used"],
        "finetune_models": finetune_info.get("models", {}),
        "reranker_finetuned": False,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "top_n_answer": top_n_answer, "use_reranker": use_reranker, "use_adaptive_k": use_adaptive,
        "dev_meteor": eval_info["meteor"], "dev_rouge_l": eval_info["rouge_l"],
        "dev_n": eval_info["n_dev"], "dev_recall_at_k": eval_info["recall_at_k"],
        "n_empty_answers": n_empty, "elapsed_min": round(elapsed() / 60, 1),
    }
    try:
        with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Đã ghi thêm 1 dòng vào {EXPERIMENT_LOG_PATH} (sổ thí nghiệm — không ghi đè).")
    except OSError as e:
        print(f"  [CẢNH BÁO] Không ghi được sổ thí nghiệm ({e}) — không ảnh hưởng submission.zip.")
    print("  [SỔ THÍ NGHIỆM — copy dòng dưới đây nếu cần đối chiếu sau này]")
    print("  " + json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()