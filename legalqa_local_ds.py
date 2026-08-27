#!/usr/bin/env python
"""
legalqa_dual_encoder.py — LegalQA (UIT DSC2026 Task 2), kiến trúc "mạnh nhất":
2 dense encoder khác họ (BAAI/bge-m3 + intfloat/multilingual-e5-large) fine-tune SONG
SONG trên 2 GPU riêng (subprocess), fusion RRF 3 kênh (BM25 + bge-m3-ft + e5-ft-large).

Đây là bản .py TRÍCH XUẤT Y HỆT logic của legalqa_kaggle_t4x2.ipynb (không viết lại tay,
để tránh lệch giữa 2 bản) — dùng khi muốn chạy như một script thuần (vd qua `python
legalqa_dual_encoder.py`, cron job, hoặc máy đa-GPU khác ngoài Kaggle) thay vì notebook.

CẦN MÁY THẬT SỰ CÓ ÍT NHẤT 1 GPU ~16GB (lý tưởng 2 GPU, kiểu Kaggle T4x2) — kiến trúc 2
encoder không phù hợp GPU 4GB (dùng bản `legalqa_local.py` — 1 encoder nhẹ — cho trường
hợp đó). Nếu chỉ có 1 GPU, script tự chuyển fine-tune 2 encoder sang chạy TUẦN TỰ thay vì
song song (xem "run_parallel" ở Bước 4) — vẫn chạy đúng, chỉ chậm hơn.

CÁCH DÙNG:
    pip install -q -U sentence-transformers datasets "accelerate>=1.1.0" nltk rouge_score sentencepiece
    python legalqa_dual_encoder.py

ĐƯỜNG DẪN: mặc định viết theo layout Kaggle (/kaggle/input/..., /kaggle/working,
/kaggle/temp/...) — xem khối DATA_DIR/OUT_DIR/CACHE_DIR. Nếu chạy ngoài Kaggle,
sửa các dòng đó cho khớp layout máy bạn rồi chạy như bình thường.
"""

import os
from pathlib import Path

# ==============================================================================
# ĐƯA CẤU HÌNH ĐƯỜNG DẪN & BIẾN MÔI TRƯỜNG RA GLOBAL SCOPE
# Sửa lỗi: Đặt ở đây để đảm bảo mọi process con (do multiprocessing/subprocess 
# sinh ra) đều thừa kế đúng cấu hình HF_HOME và CACHE_DIR, không tải nhầm.
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR)
CONTEXT_DIR = BASE_DIR / "selected-contexts"
TRAIN_PATH = BASE_DIR / "train.json"
WARMUP_PATH = BASE_DIR / "warmup.json"
PUBLIC_PATH = BASE_DIR / "public-official.json"
OUT_DIR = BASE_DIR

CACHE_DIR = BASE_DIR / "cache"
HF_CACHE_DIR = CACHE_DIR / "hf"
NLTK_CACHE_DIR = CACHE_DIR / "nltk_data"
TRAINER_TMP_DIR = CACHE_DIR / "trainer_tmp"

for _d in (OUT_DIR, HF_CACHE_DIR, NLTK_CACHE_DIR, TRAINER_TMP_DIR):
    os.makedirs(_d, exist_ok=True)

# Ép kiểu str cho các biến môi trường
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def main() -> None:
    # Tham số
    CHUNK_SIZE = 512      # không sử dụng — chunk theo Điều (xem Bước 1), giữ lại đúng như đề bài
    TOP_K_RETRIEVE = 100  # số ứng viên lấy ra sau RRF fusion (BM25 + dense)
    TOP_K_RERANK = 5      # trần trên cho số Điều đưa vào 1 câu trả lời (top_n tĩnh VÀ trần adaptive-k)
    USE_FINETUNE = True   # SỬA cho Kaggle T4x2: bật mặc định — 16GB/thẻ dư sức fine-tune, không
                           # còn ràng buộc 4GB như máy cá nhân. Đặt False nếu muốn chạy thử nhanh
                           # hoặc đang tiết kiệm quota GPU (30h/tuần trên tài khoản free).

    BASE_DENSE_MODEL_A = "BAAI/bge-m3"                        # ~568M, đa ngôn ngữ, không cần tiền tố
    BASE_DENSE_MODEL_B = "intfloat/multilingual-e5-large"     # ~560M, CẦN tiền tố "query: "/"passage: "
    DENSE_MAX_SEQ_LEN = 256
    
    # SỬA: Đảm bảo path truyền vào cấu hình lưu/load thành str tuyệt đối
    CHECKPOINT_DIR = str(OUT_DIR / "checkpoints")   
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    MIN_TRAIN_PAIRS = 50
    MAX_TRAIN_EXAMPLES = 3000
    N_NEG_PER_ROW = 2

    TRAIN_BATCH_SIZE = 64        # batch HIỆU DỤNG (số in-batch negative)
    TRAIN_MINI_BATCH_SIZE = 32   # batch THẬT mỗi forward
    ENCODE_BATCH_SIZE = 256      # batch encode corpus mỗi tiến trình GPU
    RERANK_SUBBATCH = 64         # reranker VẪN zero-shot ở bản này

    TIME_BUDGET_SEC = 8 * 3600         # Kaggle GPU session thường giới hạn ~9-12h liên tục
    FINETUNE_TIME_BUDGET_SEC = 3 * 3600
    DEV_EVAL_SAMPLE_SIZE = 300

    SEED = 42
    USE_WARMUP = True   # đặt False để ablation: chỉ dùng train.json, không gộp warmup.json
    
    # SỬA: Đảm bảo path lưu experiment thành string 
    EXPERIMENT_LOG_PATH = str(OUT_DIR / "experiment_log.jsonl")

    print(f"OUT_DIR   = {OUT_DIR}")
    print(f"CACHE_DIR = {CACHE_DIR}  (tạm, mất khi session kết thúc)")

    # Cell 3: Kiểm tra GPU (mong đợi 2x Tesla T4) + tiện ích thời gian
    import time
    import torch

    N_GPU = torch.cuda.device_count()
    print(f"Số GPU thấy được: {N_GPU}")
    for i in range(N_GPU):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} — {p.name}, {p.total_memory/1024**3:.1f} GB")

    if N_GPU == 0:
        DEVICES = ["cpu"]
        print("[CẢNH BÁO] Không thấy GPU nào — kiểm tra Settings > Accelerator = GPU T4 x2. "
              "Sẽ chạy CPU, RẤT chậm cho Bước 4/5/6/7.")
    elif N_GPU == 1:
        DEVICES = ["cuda:0"]
        print("[CẢNH BÁO] Chỉ thấy 1 GPU — vẫn chạy được nhưng KHÔNG tận dụng song song 2 thẻ "
              "ở Bước 5/6/7. Kiểm tra Settings > Accelerator = GPU T4 x2 nếu muốn đủ 2 thẻ.")
    else:
        DEVICES = [f"cuda:{i}" for i in range(N_GPU)]
        print(f"OK — sẽ dùng song song {DEVICES} ở Bước 5 (encode corpus) và Bước 6/7 (rerank).")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    _START_TIME = time.time()

    def elapsed() -> float:
        return time.time() - _START_TIME

    def remaining() -> float:
        return TIME_BUDGET_SEC - elapsed()

    def checkpoint(label: str) -> None:
        print(f"[{elapsed()/60:6.1f} phút] {label}  (còn lại ~{remaining()/60:.1f} phút trong ngân sách)")

    checkpoint("Bắt đầu")

    # Cell 4: Import chung + hằng số regex cho chunk theo Điều
    import re
    import json
    import math
    import random
    import zipfile
    from collections import defaultdict, Counter

    import numpy as np

    DIEU_RE = re.compile(r"^[ \t]*Điều\s+(\d+)[a-zđA-ZĐ]?[\.\s]", re.MULTILINE)
    SO_HEADER_RE = re.compile(r"Số\s*[:：]\s*([0-9A-Za-zĐđ/\-]+)")
    SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zĐđ]{2,10}(?:-[A-Za-zĐđ]{2,10})?")
    LOAI_VB_CANON = ["Thông tư liên tịch", "Nghị định", "Luật", "Thông tư", "Quyết định",
                     "Pháp lệnh", "Nghị quyết", "Bộ luật", "Chỉ thị"]
    LOAI_PATTERN = re.compile("(" + "|".join(re.escape(x) for x in LOAI_VB_CANON) + ")", re.IGNORECASE)
    DIEU_CITATION_RE = re.compile(r"Điều\s+(\d+)\s*[a-zđA-ZĐ]?\b")
    _TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
    _DIEU_PREFIX_STRIP_RE = re.compile(r"^\s*Điều\s+\d+[a-zđA-ZĐ]?\.?\s*", re.IGNORECASE)


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


    def tokenize_simple(text: str) -> list:
        return _TOKEN_RE.findall(text.lower())


    def norm_so_hieu(s: str) -> str:
        return s.strip().upper()

    def set_all_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_all_seeds(SEED)
    print(f"  Đã seed toàn cục với SEED={SEED} (random/numpy/torch) — trước mọi lời gọi random "
          f"ở Bước 3/4, để nhiều lần chạy cùng code fine-tune trên cùng 1 tập con, tái lập được.")

    # Cell 5: Bước 1 — Chunk corpus theo Điều
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


    def load_corpus(contexts_dir) -> list:
        contexts_dir = Path(contexts_dir)
        if not contexts_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy {contexts_dir} — kiểm tra lại CONTEXT_DIR ở đầu file.")
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


    print("=== Bước 1: Chunk corpus ===")
    all_chunks = load_corpus(CONTEXT_DIR)
    checkpoint("Xong chunking")

    # Cell 6: Bước 2 — BM25
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

            self.inverted: dict = {}
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


    print("=== Bước 2: BM25 index ===")
    tokenized = [tokenize_simple(f"{c.get('loai_vb','')} {c['text']}") for c in all_chunks]
    bm25 = BM25(tokenized)
    checkpoint("Xong BM25 index")

    # Cell 7: Bước 3 — Sinh nhãn (question -> chunk)
    def extract_citations(answer) -> list:
        if not isinstance(answer, str):
            return []
        out = []
        for m in DIEU_CITATION_RE.finditer(answer):
            window = answer[m.end(): m.end() + 60]
            so_m = SO_HIEU_RE.search(window)
            if so_m and so_m.start() <= 40:
                out.append((m.group(1), so_m.group(0)))
        return out


    def build_train_pairs(train_data: dict, all_chunks: list):
        so_hieu_index = {}
        for c in all_chunks:
            if c["so_hieu"] and c["dieu_so"] != "0":
                so_hieu_index.setdefault((c["dieu_so"], norm_so_hieu(c["so_hieu"])), c["id"])

        positive = {}
        n_skipped_type = 0
        for qid, item in train_data.items():
            if not isinstance(item.get("answer"), str):
                n_skipped_type += 1
                continue
            for dieu, so_hieu in extract_citations(item["answer"]):
                key = (dieu, norm_so_hieu(so_hieu))
                if key in so_hieu_index:
                    positive[qid] = so_hieu_index[key]
                    break
        if n_skipped_type:
            print(f"  [CẢNH BÁO] {n_skipped_type} câu có answer KHÔNG phải string -> bỏ qua.")
        chunk_by_id = {c["id"]: c for c in all_chunks}
        return positive, chunk_by_id


    print("=== Bước 3: Sinh nhãn từ train.json" + (" + warmup.json" if USE_WARMUP else "") + " ===")
    with open(TRAIN_PATH, encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"  train.json: {len(train_data)} câu")

    train_data_for_pairs = dict(train_data)
    n_warmup_used = 0
    if USE_WARMUP and os.path.exists(WARMUP_PATH):
        try:
            with open(WARMUP_PATH, encoding="utf-8") as f:
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
                  + (f", {n_bad_type} câu sai kiểu -> bỏ qua." if n_bad_type else ".")
                  + " KHÔNG gộp vào mẫu dev-eval/Recall@k ở Bước 6.")
        except Exception as e:
            print(f"  [CẢNH BÁO] Có WARMUP_PATH nhưng đọc lỗi ({e}) -> bỏ qua, chỉ dùng train.json.")
    elif USE_WARMUP:
        print(f"  USE_WARMUP=True nhưng không thấy {WARMUP_PATH} -> chỉ dùng train.json.")
    else:
        print(f"  USE_WARMUP=False -> chỉ dùng train.json (bỏ qua warmup.json dù có tồn tại).")

    train_positive, chunk_by_id = build_train_pairs(train_data_for_pairs, all_chunks)
    print(f"  Positive pairs: {len(train_positive)}/{len(train_data_for_pairs)}")
    checkpoint("Xong sinh nhãn")


    # Cell 8: Bước 4 — Fine-tune 2 dense encoder SONG SONG THẬT trên 2 GPU riêng
    import subprocess
    import sys

    print("=== Bước 4: Fine-tune 2 dense encoder song song ===")
    
    # SỬA: Đảm bảo path string
    WORKER_SCRIPT = str(CACHE_DIR / "_train_encoder_worker.py")
    
    worker_code = '''
    import argparse, json, os, sys, time


    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--base-model", required=True)
        p.add_argument("--gpu-index", required=True)
        p.add_argument("--rows-path", required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--max-seq-len", type=int, default=256)
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--mini-batch-size", type=int, default=16)
        p.add_argument("--time-budget-sec", type=float, required=True)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--query-prefix", default="")
        p.add_argument("--passage-prefix", default="")
        args = p.parse_args()

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        from datasets import Dataset
        from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                            SentenceTransformerTrainingArguments)
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

        with open(args.rows_path, encoding="utf-8") as f:
            rows = json.load(f)
        if args.query_prefix or args.passage_prefix:
            fixed = []
            for r in rows:
                r2 = dict(r)
                r2["anchor"] = args.query_prefix + r["anchor"]
                for k in r:
                    if k.startswith("positive") or k.startswith("negative"):
                        r2[k] = args.passage_prefix + r[k]
                fixed.append(r2)
            rows = fixed
        dataset = Dataset.from_list(rows)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(args.base_model, device=device)
        model.max_seq_length = args.max_seq_len

        batch_size, mini_batch_size = args.batch_size, args.mini_batch_size
        max_steps, calib_time = 0, None
        t0 = time.time()
        for attempt in range(4):
            try:
                loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
                calib_steps = min(10, max(1, len(dataset) // batch_size))
                calib_args = SentenceTransformerTrainingArguments(
                    output_dir=args.output_dir + "_tmp", max_steps=calib_steps,
                    per_device_train_batch_size=batch_size, logging_steps=calib_steps + 1,
                    save_strategy="no", report_to=[], disable_tqdm=True, fp16=(device == "cuda:0"))
                c0 = time.time()
                print(f"[{args.base_model}] calib training (batch={batch_size}, mini_batch={mini_batch_size})...", flush=True)
                SentenceTransformerTrainer(model=model, args=calib_args, train_dataset=dataset, loss=loss).train()
                calib_time = (time.time() - c0) / calib_steps

                budget_left = args.time_budget_sec - (time.time() - t0) - 60
                max_steps = max(0, int(budget_left / max(calib_time, 1e-6)))
                max_steps = min(max_steps, (len(dataset) // batch_size) * 8)
                print(f"[{args.base_model}] calib {calib_time:.2f}s/step, ngan sach con "
                      f"{budget_left/60:.1f} phut -> {max_steps} step", flush=True)

                if max_steps > 0:
                    targs = SentenceTransformerTrainingArguments(
                        output_dir=args.output_dir + "_tmp", max_steps=max_steps,
                        per_device_train_batch_size=batch_size, learning_rate=2e-5,
                        warmup_steps=0.05, lr_scheduler_type="cosine",
                        logging_steps=max(1, max_steps // 20), save_strategy="no", report_to=[],
                        fp16=(device == "cuda:0"))
                    SentenceTransformerTrainer(model=model, args=targs, train_dataset=dataset, loss=loss).train()
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and mini_batch_size > 1:
                    torch.cuda.empty_cache()
                    mini_batch_size = max(1, mini_batch_size // 2)
                    print(f"[{args.base_model}] OOM -> mini_batch_size={mini_batch_size}", flush=True)
                    continue
                raise

        model.save_pretrained(args.output_dir)
        meta = {"max_steps": max_steps, "mini_batch_final": mini_batch_size,
                "calib_time_s": calib_time, "elapsed_s": time.time() - t0}
        with open(args.output_dir + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        print(f"[{args.base_model}] DONE -> {args.output_dir}", flush=True)


    if __name__ == "__main__":
        main()
    '''
    with open(WORKER_SCRIPT, "w", encoding="utf-8") as f:
        f.write(worker_code)

    def _build_training_rows(train_positive, train_data, chunk_by_id, all_chunks, bm25, n_neg=N_NEG_PER_ROW):
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


    finetune_info = {"used_finetune": False, "reason": None, "n_pairs_available": len(train_positive),
                      "n_pairs_used": 0, "models": {}}

    use_finetune = USE_FINETUNE and len(train_positive) >= MIN_TRAIN_PAIRS and remaining() > 10 * 60
    DENSE_CHANNELS = []

    if not use_finetune:
        reason = "USE_FINETUNE=False" if not USE_FINETUNE else (
            f"{len(train_positive)} positive pairs < {MIN_TRAIN_PAIRS}" if len(train_positive) < MIN_TRAIN_PAIRS
            else "hết ngân sách thời gian")
        print(f"  {reason} -> dùng zero-shot cho cả 2 encoder, không fine-tune.")
        finetune_info["reason"] = reason
        from sentence_transformers import SentenceTransformer
        m_a = SentenceTransformer(BASE_DENSE_MODEL_A, device=DEVICES[0]); m_a.max_seq_length = DENSE_MAX_SEQ_LEN
        m_b = SentenceTransformer(BASE_DENSE_MODEL_B, device=DEVICES[-1]); m_b.max_seq_length = DENSE_MAX_SEQ_LEN
        DENSE_CHANNELS = [
            {"name": "bge-m3", "model": m_a, "embeddings": None, "query_prefix": "", "passage_prefix": ""},
            {"name": "e5-large", "model": m_b, "embeddings": None, "query_prefix": "query: ", "passage_prefix": "passage: "},
        ]
    else:
        train_positive_used = train_positive
        if len(train_positive) > MAX_TRAIN_EXAMPLES:
            sampled_qids = random.sample(list(train_positive.keys()), MAX_TRAIN_EXAMPLES)
            train_positive_used = {qid: train_positive[qid] for qid in sampled_qids}
            print(f"  Có {len(train_positive)} positive pairs, lấy mẫu {MAX_TRAIN_EXAMPLES} "
                  f"(tái lập được nhờ SEED={SEED}).")
        finetune_info["n_pairs_used"] = len(train_positive_used)

        print(f"  Đang tạo training rows (dùng chung cho cả 2 encoder)...")
        rows = _build_training_rows(train_positive_used, train_data_for_pairs, chunk_by_id, all_chunks, bm25)
        
        # SỬA: Ép chuỗi string để dùng cho Worker Popen args an toàn
        rows_path = str(CACHE_DIR / "train_rows.json")
        with open(rows_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        print(f"  {len(rows)} rows -> {rows_path}")

        # SỬA: Ép chuỗi string cho output dict an toàn truyền đi 
        specs = [
            {"name": "bge-m3", "base_model": BASE_DENSE_MODEL_A, "gpu": DEVICES[0].split(":")[-1],
             "out": str(Path(CHECKPOINT_DIR) / "bge-m3-ft"), "query_prefix": "", "passage_prefix": ""},
            {"name": "e5-large", "base_model": BASE_DENSE_MODEL_B, "gpu": DEVICES[-1].split(":")[-1],
             "out": str(Path(CHECKPOINT_DIR) / "e5-large-ft"), "query_prefix": "query: ", "passage_prefix": "passage: "},
        ]
        
        run_parallel = len(DEVICES) > 1 and specs[0]["gpu"] != specs[1]["gpu"]
        time_budget_each = max(600.0, min(remaining() - 5 * 60, FINETUNE_TIME_BUDGET_SEC)
                                / (1.0 if run_parallel else 2.0))
        print(f"  Chạy {'SONG SONG (2 GPU riêng)' if run_parallel else 'TUẦN TỰ (chỉ 1 GPU khả dụng)'} "
              f"— ngân sách mỗi encoder ~{time_budget_each/60:.0f} phút.")

        def _launch(spec):
            log_path = str(CACHE_DIR / f"train_{spec['name']}.log")
            cmd = [sys.executable, WORKER_SCRIPT,
                   "--base-model", spec["base_model"], "--gpu-index", spec["gpu"],
                   "--rows-path", rows_path, "--output-dir", spec["out"],
                   "--max-seq-len", str(DENSE_MAX_SEQ_LEN), "--batch-size", str(TRAIN_BATCH_SIZE),
                   "--mini-batch-size", str(TRAIN_MINI_BATCH_SIZE), "--time-budget-sec", str(time_budget_each),
                   "--seed", str(SEED), "--query-prefix", spec["query_prefix"], "--passage-prefix", spec["passage_prefix"]]
            lf = open(log_path, "w")
            print(f"  Khởi động fine-tune {spec['name']} trên GPU {spec['gpu']} -> log: {log_path}")
            return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT), lf

        failed = []
        if run_parallel:
            procs = [(spec, *_launch(spec)) for spec in specs]
            print(f"  Đang chờ {len(procs)} tiến trình fine-tune song song...", flush=True)
            for spec, proc, lf in procs:
                rc = proc.wait()
                print(f"  {spec['name']}: xong, mã thoát {rc}")
                if rc != 0:
                    failed.append(spec["name"])
                lf.close()
        else:
            for spec in specs:
                proc, lf = _launch(spec)
                rc = proc.wait()
                print(f"  {spec['name']}: xong, mã thoát {rc}")
                if rc != 0:
                    failed.append(spec["name"])
                lf.close()
        if failed:
            raise SystemExit(f"Fine-tune lỗi: {failed} — xem log trong thư mục cache.")

        from sentence_transformers import SentenceTransformer
        for spec in specs:
            meta_path = spec["out"] + "_meta.json"
            with open(meta_path, encoding="utf-8") as f:
                m = json.load(f)
            finetune_info["models"][spec["name"]] = m
            print(f"  {spec['name']}: {m['max_steps']} step, mini_batch cuối={m['mini_batch_final']}, "
                  f"{m['elapsed_s']/60:.1f} phút")

        m_a = SentenceTransformer(specs[0]["out"], device=DEVICES[0])
        m_b = SentenceTransformer(specs[1]["out"], device=DEVICES[-1])
        DENSE_CHANNELS = [
            {"name": "bge-m3", "model": m_a, "embeddings": None, "query_prefix": "", "passage_prefix": ""},
            {"name": "e5-large", "model": m_b, "embeddings": None, "query_prefix": "query: ", "passage_prefix": "passage: "},
        ]
        finetune_info["used_finetune"] = True
        print(f"  Checkpoint đã lưu trong {CHECKPOINT_DIR} — tự tải về nếu muốn dùng lại phiên sau.")

    checkpoint("Xong Bước 4 (2 dense encoder)")


    # Cell 9: Bước 5 — Encode toàn bộ corpus CHO CẢ 2 ENCODER
    print(f"=== Bước 5: Encode toàn bộ corpus cho {len(DENSE_CHANNELS)} encoder ===")
    texts_raw = [c["text"] for c in all_chunks]

    for ch in DENSE_CHANNELS:
        t0 = time.time()
        texts = [ch["passage_prefix"] + t for t in texts_raw] if ch["passage_prefix"] else texts_raw
        print(f"  [{ch['name']}] encode {len(texts)} chunk"
              + (f' (tiền tố "{ch["passage_prefix"]}")' if ch["passage_prefix"] else "") + " ...")
        model = ch["model"]
        if len(DEVICES) > 1 and DEVICES[0].startswith("cuda"):
            pool = model.start_multi_process_pool(target_devices=DEVICES)
            try:
                emb = model.encode_multi_process(texts, pool, batch_size=ENCODE_BATCH_SIZE,
                                                  normalize_embeddings=True)
            finally:
                model.stop_multi_process_pool(pool)
        else:
            device = DEVICES[0]
            model = model.to(device)
            if device.startswith("cuda"):
                model = model.half()
            batch_size = ENCODE_BATCH_SIZE
            while True:
                try:
                    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                        show_progress_bar=True, normalize_embeddings=True, device=device)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and batch_size > 1:
                        print(f"    [CUDA OOM] batch_size={batch_size} -> thử {batch_size // 2}")
                        torch.cuda.empty_cache()
                        batch_size = max(1, batch_size // 2)
                        continue
                    raise
            ch["model"] = model
        ch["embeddings"] = emb
        print(f"    -> {emb.shape}, {time.time()-t0:.0f}s")

    checkpoint("Xong encode corpus (2 encoder)")


    # Cell 10: Bước 5b — Tải reranker
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    def load_reranker_on(device: str):
        for attempt in range(2):
            try:
                print(f"  Đang tải reranker AITeamVN/Vietnamese_Reranker lên {device}"
                      f"{' — thử lại lần 2' if attempt else ''}...")
                tok = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Reranker")
                mdl = AutoModelForSequenceClassification.from_pretrained("AITeamVN/Vietnamese_Reranker")
                mdl = mdl.to(device)
                if device.startswith("cuda"):
                    mdl = mdl.half()
                mdl.eval()
                return mdl, tok
            except Exception as e:
                if attempt == 0:
                    print(f"  [Lần 1 lỗi: {e}] thử lại sau 5s...")
                    time.sleep(5)
                    continue
                print(f"  [CẢNH BÁO] Không tải được reranker trên {device} ({e}) -> bỏ qua "
                      f"reranker trên thẻ này.")
                return None, None


    print("=== Bước 5b: Tải reranker (mỗi GPU 1 bản) ===")
    reranker_models, reranker_tokenizers = {}, {}
    for dev in DEVICES:
        m, t = load_reranker_on(dev)
        if m is not None:
            reranker_models[dev] = m
            reranker_tokenizers[dev] = t

    HAS_RERANKER = len(reranker_models) > 0
    RERANK_DEVICES = list(reranker_models.keys())
    print(f"  Reranker sẵn sàng trên: {RERANK_DEVICES or '(không tải được — sẽ chạy không rerank)'}")
    checkpoint("Xong tải reranker")

    # Cell 11: Hàm retrieval + rerank theo lô
    from concurrent.futures import ThreadPoolExecutor

    _print_lock = __import__("threading").Lock()

    def rrf_retrieve(question: str, bm25, dense_channels, all_chunks, top_k: int = TOP_K_RETRIEVE):
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


    def rerank(question: str, candidates: list, reranker_model, reranker_tokenizer,
               max_candidates: int = TOP_K_RETRIEVE, max_length: int = 1024, sub_batch: int = RERANK_SUBBATCH):
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
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    continue
                if "out of memory" in str(e).lower():
                    return candidates, None
                raise
        order = np.argsort(-scores)
        reranked = [subset[i2] for i2 in order]
        sorted_scores = scores[order]
        return reranked + candidates[max_candidates:], sorted_scores


    def adaptive_k_cutoff(scores, min_k: int = 1, max_k: int = TOP_K_RERANK, search_window: int = 15) -> int:
        if scores is None or len(scores) == 0:
            return min_k
        n = min(len(scores), search_window)
        if n <= 1:
            return min_k
        gaps = [scores[i] - scores[i + 1] for i in range(n - 1)]
        k_star = int(np.argmax(gaps)) + 1
        return max(min_k, min(k_star, max_k))


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


    def answer_question(question: str, bm25, dense_channels, all_chunks, top_n: int,
                         reranker_model=None, reranker_tokenizer=None, use_adaptive_k: bool = False) -> str:
        ranked = rrf_retrieve(question, bm25, dense_channels, all_chunks)
        if not ranked:
            return "Không tìm thấy thông tin pháp lý cho câu hỏi này."
        scores = None
        if reranker_model is not None:
            ranked, scores = rerank(question, ranked, reranker_model, reranker_tokenizer)
        n = adaptive_k_cutoff(scores) if (use_adaptive_k and scores is not None) else top_n
        return render_answer(ranked, n)


    def split_evenly(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


    def parallel_process(ids, worker_fn, devices, label: str = "", progress_every: int = 50):
        ids = list(ids)
        devices = list(devices) if devices else ["cpu"]
        chunks = split_evenly(ids, len(devices))
        results = {}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(devices)) as ex:
            futures = [ex.submit(worker_fn, chunk, dev, i) for i, (chunk, dev) in enumerate(zip(chunks, devices))]
            for f in futures:
                results.update(f.result())
        print(f"    [{label}] {len(ids)} câu / {len(devices)} thiết bị song song -> {time.time()-t0:.0f}s")
        return results


    def _progress_print(label, worker_idx, i, n):
        if (i + 1) % 50 == 0 or (i + 1) == n:
            with _print_lock:
                print(f"    [{label} · luồng {worker_idx}] {i+1}/{n}")

    # Cell 12: Bước 6 — Dev-eval
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

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    random.seed(SEED)

    print("=== Bước 6: Dev-eval chọn TOP_N_ANSWER + đo Recall@k ===")
    n_sample = min(DEV_EVAL_SAMPLE_SIZE, len(train_data))
    dev_ids = random.sample(list(train_data.keys()), n_sample)
    recall_ids = [q for q in dev_ids if q in train_positive]
    print(f"  Mẫu dev-eval: {len(dev_ids)} câu ({len(recall_ids)} câu có citation resolve được "
          f"-> dùng luôn để đo Recall@k, không chạy lại retrieval riêng).")

    ks = [1, 3, 5, 10, 30, 100]
    configs = [("BM25+dense (không rerank)", False)]
    if HAS_RERANKER:
        configs.append(("BM25+dense+rerank", True))

    best_n, best_m, best_r, best_use_rerank, best_use_adaptive = 3, -1.0, None, False, False
    recall_at_k_by_label = {}
    for label, use_rr in configs:
        print(f"  --- {label} ---")
        if use_rr and len(RERANK_DEVICES) > 1:
            def _worker(chunk, dev, widx, _label=label):
                out = {}
                for i, qid in enumerate(chunk):
                    item = train_data[qid]
                    ranked = rrf_retrieve(item["question"], bm25, DENSE_CHANNELS, all_chunks)
                    scores = None
                    if ranked:
                        ranked, scores = rerank(item["question"], ranked,
                                                 reranker_models[dev], reranker_tokenizers[dev])
                    out[qid] = (ranked, scores)
                    _progress_print(_label, widx, i, len(chunk))
                return out
            merged = parallel_process(dev_ids, _worker, RERANK_DEVICES, label=label)
            ranked_cache = {q: v[0] for q, v in merged.items()}
            scores_cache = {q: v[1] for q, v in merged.items()}
        else:
            rr_dev = RERANK_DEVICES[0] if (use_rr and RERANK_DEVICES) else None
            ranked_cache, scores_cache = {}, {}
            t0 = time.time()
            for i, qid in enumerate(dev_ids):
                item = train_data[qid]
                ranked = rrf_retrieve(item["question"], bm25, DENSE_CHANNELS, all_chunks)
                scores = None
                if rr_dev is not None and ranked:
                    ranked, scores = rerank(item["question"], ranked,
                                             reranker_models[rr_dev], reranker_tokenizers[rr_dev])
                ranked_cache[qid] = ranked
                scores_cache[qid] = scores
                if (i + 1) % 50 == 0 or (i + 1) == len(dev_ids):
                    print(f"    retrieval+rerank {i+1}/{len(dev_ids)} ... {time.time()-t0:.0f}s")

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
            for qid in dev_ids:
                ranked = ranked_cache[qid]
                pred = render_answer(ranked, top_n) if ranked else "Không tìm thấy thông tin pháp lý cho câu hỏi này."
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms) / len(ms), sum(rs) / len(rs)
            print(f"    top_n={top_n}  METEOR={m:.4f}  ROUGE-L={r:.4f}  (n={len(dev_ids)})")
            if m > best_m:
                best_m, best_r, best_n = m, r, top_n
                best_use_rerank, best_use_adaptive = use_rr, False

        if use_rr:
            ms, rs = [], []
            for qid in dev_ids:
                ranked, scores = ranked_cache[qid], scores_cache[qid]
                k = adaptive_k_cutoff(scores) if ranked else 1
                pred = render_answer(ranked, k) if ranked else "Không tìm thấy thông tin pháp lý cho câu hỏi này."
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms) / len(ms), sum(rs) / len(rs)
            print(f"    adaptive-k       METEOR={m:.4f}  ROUGE-L={r:.4f}  (n={len(dev_ids)})")
            if m > best_m:
                best_m, best_r, best_use_rerank, best_use_adaptive = m, r, True, True

    print(f"  => chọn TOP_N_ANSWER={best_n}, dùng reranker={best_use_rerank}, "
          f"dùng adaptive-k={best_use_adaptive} (METEOR={best_m:.4f})")
    top_n_answer, use_reranker, use_adaptive = best_n, best_use_rerank, best_use_adaptive
    eval_info = {"meteor": round(best_m, 4), "rouge_l": (round(best_r, 4) if best_r is not None else None),
                 "recall_at_k": recall_at_k_by_label, "n_dev": len(dev_ids)}
    checkpoint("Xong dev-eval + Recall@k")


    # Cell 13: Bước 7 — Sinh câu trả lời cho public-official.json
    print("=== Bước 7: Sinh câu trả lời cho public-official.json ===")
    with open(PUBLIC_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    qids = list(questions.keys())

    if use_reranker and len(RERANK_DEVICES) > 1:
        def _worker(chunk, dev, widx):
            out = {}
            for i, qid in enumerate(chunk):
                out[qid] = answer_question(questions[qid]["question"], bm25, DENSE_CHANNELS,
                                            all_chunks, top_n_answer,
                                            reranker_model=reranker_models[dev],
                                            reranker_tokenizer=reranker_tokenizers[dev],
                                            use_adaptive_k=use_adaptive)
                _progress_print("Bước 7", widx, i, len(chunk))
            return out
        answers = parallel_process(qids, _worker, RERANK_DEVICES, label="Bước 7")
    else:
        rr_dev = RERANK_DEVICES[0] if (use_reranker and RERANK_DEVICES) else None
        rr_model = reranker_models[rr_dev] if rr_dev else None
        rr_tok = reranker_tokenizers[rr_dev] if rr_dev else None
        answers = {}
        for i, qid in enumerate(qids):
            answers[qid] = answer_question(questions[qid]["question"], bm25, DENSE_CHANNELS,
                                            all_chunks, top_n_answer, reranker_model=rr_model,
                                            reranker_tokenizer=rr_tok, use_adaptive_k=use_adaptive)
            if (i + 1) % 200 == 0:
                print(f"  ... {i+1}/{len(qids)}  ({elapsed()/60:.1f} phút)")

    n_empty = sum(1 for a in answers.values() if not a.strip())
    print(f"  Đã sinh {len(answers)} câu trả lời, {n_empty} câu rỗng")
    checkpoint("Xong sinh câu trả lời")

    # Cell 14: Bước 8 — Validate + đóng gói submission.zip
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


    print("=== Bước 8: Đóng gói submission.zip ===")
    build_submission(answers, set(questions.keys()), Path(OUT_DIR) / "submission.zip")
    checkpoint(f"XONG — tổng thời gian {elapsed()/60:.1f} phút (trần an toàn {TIME_BUDGET_SEC/3600:.0f} giờ)")

    n_empty = sum(1 for a in answers.values() if not a.strip())
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "use_warmup": USE_WARMUP,
        "n_warmup_used": n_warmup_used, "hardware": f"kaggle_t4x{N_GPU}",
        "n_train_pairs_available": finetune_info["n_pairs_available"],
        "n_train_pairs_used": finetune_info["n_pairs_used"],
        "used_finetune": finetune_info["used_finetune"], "finetune_reason": finetune_info["reason"],
        "finetune_models": finetune_info["models"],
        "reranker_finetuned": False,
        "checkpoint_dir": CHECKPOINT_DIR if finetune_info["used_finetune"] else None,
        "top_n_answer": top_n_answer, "use_reranker": use_reranker, "use_adaptive_k": use_adaptive,
        "dev_meteor": eval_info["meteor"], "dev_rouge_l": eval_info["rouge_l"],
        "dev_n": eval_info["n_dev"], "dev_recall_at_k": eval_info["recall_at_k"],
        "n_empty_answers": n_empty, "elapsed_min": round(elapsed() / 60, 1),
    }
    try:
        with open(EXPERIMENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Đã ghi thêm 1 dòng vào {EXPERIMENT_LOG_PATH} (sổ thí nghiệm — không ghi đè, "
              f"giữ lại khi Save Version).")
    except OSError as e:
        print(f"  [CẢNH BÁO] Không ghi được sổ thí nghiệm ({e}) — không ảnh hưởng submission.zip.")
    print("  [SỔ THÍ NGHIỆM — copy dòng dưới đây nếu cần đối chiếu sau này]")
    print("  " + json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()