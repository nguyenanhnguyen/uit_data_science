"""
legalqa_optimized.py — LegalQA (UIT DSC2026 Task 2)
Tối ưu tốc độ cho RTX 2050 4GB VRAM, vẫn giữ fine-tune và rerank.
Các thay đổi so với bản #6:
- Tăng ENCODE_BATCH_SIZE lên 128 (tự lùi nếu OOM)
- Tăng TRAIN_MINI_BATCH_SIZE lên 8
- Giảm MAX_TRAIN_EXAMPLES xuống 2000 (vẫn đủ)
- Giảm DEV_EVAL_SAMPLE_SIZE xuống 200 (nhanh hơn)
- Tăng RERANK_SUBBATCH lên 32 (cân bằng tốc độ/VRAM)
- Thêm seed cố định toàn cục để so sánh đáng tin

CÁCH DÙNG: đặt file này cạnh train.json, public-official.json, selected-contexts/ rồi chạy:
    python legalqa_optimized.py
Output: submission.zip trong cùng thư mục.

THƯ VIỆN CẦN CÀI:
    pip install numpy sentence-transformers datasets accelerate nltk rouge_score tiktoken sentencepiece
"""
from __future__ import annotations
import os
from pathlib import Path

# ---- Cache & env ----
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
HF_CACHE_DIR = CACHE_DIR / "hf"
NLTK_CACHE_DIR = CACHE_DIR / "nltk_data"
TRAINER_TMP_DIR = CACHE_DIR / "trainer_tmp"
for _d in (HF_CACHE_DIR, NLTK_CACHE_DIR, TRAINER_TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ---- Seed cố định để kết quả lặp lại ----
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)

import re
import json
import math
import time
import zipfile
from collections import defaultdict, Counter

# ==============================================================================
# CONFIG — Các tham số điều chỉnh để tối ưu tốc độ
# ==============================================================================
CONTEXTS_DIR = HERE / "selected-contexts"
TRAIN_PATH = HERE / "train.json"
PUBLIC_PATH = HERE / "public-official.json"
OUT_DIR = HERE

TIME_BUDGET_SEC = 3 * 3600          # 3 giờ an toàn, không cắt ngang
FINETUNE_TIME_BUDGET_SEC = 90 * 60
MIN_TRAIN_PAIRS = 50
MAX_TRAIN_EXAMPLES = 2000           # giảm từ 3000 để train nhanh hơn

BASE_DENSE_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
DENSE_MAX_SEQ_LEN = 256
TRAIN_BATCH_SIZE = 32               # hiệu dụng (in-batch negative)
TRAIN_MINI_BATCH_SIZE = 8           # tăng từ 4 lên 8 (tự lùi nếu OOM)
N_NEG_PER_ROW = 2
ENCODE_BATCH_SIZE = 128             # tăng từ 64 lên 128 (tự lùi)
TOP_K_RETRIEVE = 100
DEV_EVAL_SAMPLE_SIZE = 200          # giảm từ 300 để dev-eval nhanh hơn
RERANK_SUBBATCH = 32                # tăng từ 24 lên 32

_START_TIME = time.time()

def elapsed() -> float:
    return time.time() - _START_TIME

def remaining() -> float:
    return TIME_BUDGET_SEC - elapsed()

def checkpoint(label: str) -> None:
    print(f"[{elapsed()/60:5.1f} phút] {label}  (còn lại ~{remaining()/60:.1f} phút)")

# ==============================================================================
# Các hàm xử lý giống bản #6 (giữ nguyên)
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
        end = matches[i+1].start() if i+1 < len(matches) else len(passage)
        dieu = m.group(1)
        chunks.append({"id": f"{doc_id}_{dieu}_{i}", "dieu_so": dieu,
                        "loai_vb": "", "so_hieu": "",
                        "text": passage[start:end].strip()})
    return chunks

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
        print("  [CẢNH BÁO] < 95% — kiểm tra vài context_*.json.")
    return all_chunks

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
def tokenize_simple(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())

class BM25:
    def __init__(self, tokenized_docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(tokenized_docs)
        self.doc_len = np.array([len(d) for d in tokenized_docs], dtype=np.float64)
        self.avgdl = self.doc_len.mean() if self.N else 0.0
        raw_postings = defaultdict(list)
        for i, doc in enumerate(tokenized_docs):
            for term, f in Counter(doc).items():
                raw_postings[term].append((i, f))
        self.inverted = {}
        for term, postings in raw_postings.items():
            idxs = np.fromiter((p[0] for p in postings), dtype=np.int32, count=len(postings))
            freqs = np.fromiter((p[1] for p in postings), dtype=np.float64, count=len(postings))
            self.inverted[term] = (idxs, freqs)
        df = {t: len(idxs) for t, (idxs, _) in self.inverted.items()}
        idf_raw = {t: math.log((self.N - n + 0.5)/(n + 0.5) + 1) for t, n in df.items()}
        avg_idf = sum(idf_raw.values()) / len(idf_raw) if idf_raw else 0.0
        eps = 0.25 * avg_idf
        self.idf = {t: (v if v > 0 else eps) for t, v in idf_raw.items()}

    def get_scores(self, query_tokens):
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

    def top_k(self, query_tokens, k: int):
        scores = self.get_scores(query_tokens)
        return list(np.argsort(-scores)[:k])

DIEU_CITATION_RE = re.compile(r"Điều\s+(\d+)\s*[a-zđA-ZĐ]?\b")
def extract_citations(answer):
    if not isinstance(answer, str):
        return []
    out = []
    for m in DIEU_CITATION_RE.finditer(answer):
        window = answer[m.end(): m.end()+60]
        so_m = SO_HIEU_RE.search(window)
        if so_m and so_m.start() <= 40:
            out.append((m.group(1), so_m.group(0)))
    return out

def norm_so_hieu(s):
    return s.strip().upper()

def build_train_pairs(train_data, all_chunks):
    so_hieu_index = {}
    for c in all_chunks:
        if c["so_hieu"] and c["dieu_so"] != "0":
            so_hieu_index.setdefault((c["dieu_so"], norm_so_hieu(c["so_hieu"])), c["id"])
    positive = {}
    for qid, item in train_data.items():
        for dieu, so_hieu in extract_citations(item.get("answer", "")):
            key = (dieu, norm_so_hieu(so_hieu))
            if key in so_hieu_index:
                positive[qid] = so_hieu_index[key]
                break
    chunk_by_id = {c["id"]: c for c in all_chunks}
    return positive, chunk_by_id

# ==============================================================================
# Các hàm fine-tune, encode, rerank (giữ nguyên logic, chỉ thay tham số)
# ==============================================================================
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
        if (i+1) % 500 == 0 or (i+1) == n:
            print(f"    _build_training_rows: {i+1}/{n}  ({elapsed()/60:.1f} phút)")
    return rows

def enable_cuda_perf():
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  [GPU] {torch.cuda.get_device_name(0)} — {total_gb:.1f}GB VRAM (dùng toàn bộ, OOM-backoff tự lùi)")
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
        enable_cuda_perf()
    else:
        print("  [CẢNH BÁO] Không thấy GPU — dùng CPU (RẤT CHẬM).")
    use_finetune = len(train_positive) >= MIN_TRAIN_PAIRS and remaining() > 5*60
    if not use_finetune:
        print(f"  Bỏ fine-tune (lý do: {len(train_positive)}<{MIN_TRAIN_PAIRS} hoặc hết ngân sách)")
        model = SentenceTransformer(BASE_DENSE_MODEL, device=device)
        model.max_seq_length = DENSE_MAX_SEQ_LEN
        return model

    model = SentenceTransformer(BASE_DENSE_MODEL, device=device)
    model.max_seq_length = DENSE_MAX_SEQ_LEN

    try:
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
        import accelerate
        if tuple(map(int, accelerate.__version__.split(".")[:2])) < (1,1):
            raise ImportError("accelerate < 1.1.0")
    except ImportError as e:
        print(f"  [THIẾU PACKAGE] {e} — dùng zero-shot.")
        return model

    if len(train_positive) > MAX_TRAIN_EXAMPLES:
        sampled_qids = random.sample(list(train_positive.keys()), MAX_TRAIN_EXAMPLES)
        train_positive_used = {qid: train_positive[qid] for qid in sampled_qids}
        print(f"  Lấy mẫu {MAX_TRAIN_EXAMPLES} / {len(train_positive)} positive pairs.")
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
            calib_steps = min(10, max(1, len(dataset)//batch_size))
            calib_args = SentenceTransformerTrainingArguments(
                output_dir=str(TRAINER_TMP_DIR), max_steps=calib_steps,
                per_device_train_batch_size=batch_size, logging_steps=calib_steps+1,
                save_strategy="no", report_to=[], disable_tqdm=True,
                fp16=cuda_ok,
            )
            calib_start = time.time()
            print(f"  Calib training (batch={batch_size}, mini_batch={mini_batch_size})...")
            SentenceTransformerTrainer(model=model, args=calib_args, train_dataset=dataset, loss=loss).train()
            calib_time = (time.time() - calib_start) / calib_steps

            budget_left = min(remaining() - 3*60, FINETUNE_TIME_BUDGET_SEC - (time.time()-calib_start))
            max_steps = max(0, int(budget_left / max(calib_time, 1e-6)))
            max_steps = min(max_steps, (len(dataset)//batch_size)*8)
            print(f"  Calib: ~{calib_time:.2f}s/step, còn ~{budget_left/60:.1f} phút -> {max_steps} steps")

            if max_steps > 0:
                args = SentenceTransformerTrainingArguments(
                    output_dir=str(TRAINER_TMP_DIR), max_steps=max_steps,
                    per_device_train_batch_size=batch_size, learning_rate=2e-5,
                    warmup_steps=0.05, lr_scheduler_type="cosine",
                    logging_steps=max(1, max_steps//20), save_strategy="no", report_to=[],
                    fp16=cuda_ok,
                )
                SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss).train()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and mini_batch_size > 1:
                print(f"  OOM, giảm mini_batch: {mini_batch_size} -> {mini_batch_size//2}")
                torch.cuda.empty_cache()
                mini_batch_size = max(1, mini_batch_size//2)
                continue
            raise
    model = model.to(device)
    actual_device = next(model.parameters()).device
    print(f"  Device sau train: {actual_device}")
    return model

def encode_corpus(model, all_chunks):
    import torch
    if torch.cuda.is_available():
        enable_cuda_perf()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    actual_device = next(model.parameters()).device
    print(f"  Encode trên {actual_device}{' (fp16)' if device=='cuda' else ''}")
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
                print(f"  OOM, giảm batch encode: {batch_size} -> {batch_size//2}")
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size//2)
                continue
            raise

def load_reranker():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    for attempt in range(2):
        try:
            print(f"  Tải reranker AITeamVN/Vietnamese_Reranker (zero-shot){' (thử lại)' if attempt else ''}...")
            tokenizer = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Reranker")
            model = AutoModelForSequenceClassification.from_pretrained("AITeamVN/Vietnamese_Reranker")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            if device == "cuda":
                model = model.half()
            model.eval()
            print(f"  Reranker sẵn sàng trên {next(model.parameters()).device}")
            return model, tokenizer
        except Exception as e:
            if attempt == 0:
                print(f"  Lỗi lần 1: {e}, thử lại sau 5s...")
                time.sleep(5)
                continue
            print(f"  Không tải được reranker ({e}) — bỏ qua rerank.")
            return None, None

def rerank(question, candidates, reranker_model, reranker_tokenizer,
           max_candidates=100, max_length=1024, sub_batch=RERANK_SUBBATCH):
    import torch
    if reranker_model is None or not candidates:
        return candidates, None
    subset = candidates[:max_candidates]
    device = next(reranker_model.parameters()).device
    pairs = [[question, c["text"]] for c in subset]
    scores = np.empty(len(pairs), dtype=np.float32)
    bs, i = max(1, sub_batch), 0
    while i < len(pairs):
        batch = pairs[i:i+bs]
        try:
            with torch.no_grad():
                inputs = reranker_tokenizer(batch, padding=True, truncation=True,
                                             return_tensors="pt", max_length=max_length).to(device)
                out = reranker_model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()
            scores[i:i+len(batch)] = out
            i += bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1:
                print(f"  OOM rerank, giảm sub_batch: {bs} -> {bs//2}")
                torch.cuda.empty_cache()
                bs = max(1, bs//2)
                continue
            if "out of memory" in str(e).lower():
                print("  OOM rerank, bỏ qua cho câu hỏi này")
                return candidates, None
            raise
    order = np.argsort(-scores)
    reranked = [subset[i] for i in order]
    return reranked + candidates[max_candidates:], scores[order]

def adaptive_k_cutoff(scores, min_k=1, max_k=5, search_window=15):
    if scores is None or len(scores) == 0:
        return min_k
    n = min(len(scores), search_window)
    if n <= 1:
        return min_k
    gaps = [scores[i] - scores[i+1] for i in range(n-1)]
    k = int(np.argmax(gaps)) + 1
    return max(min_k, min(k, max_k))

def rrf_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks, top_k=TOP_K_RETRIEVE):
    bm25_ranked = bm25.top_k(tokenize_simple(question), top_k)
    q_emb = dense_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    dense_scores = dense_embeddings @ q_emb
    dense_ranked = list(np.argsort(-dense_scores)[:top_k])
    bm25_map = {idx: r for r, idx in enumerate(bm25_ranked)}
    dense_map = {idx: r for r, idx in enumerate(dense_ranked)}
    all_idx = set(bm25_ranked) | set(dense_ranked)
    rrf = {i: 1/(60 + bm25_map.get(i, top_k+1)) + 1/(60 + dense_map.get(i, top_k+1)) for i in all_idx}
    ranked = sorted(rrf, key=rrf.get, reverse=True)
    return [all_chunks[i] for i in ranked]

_DIEU_PREFIX_STRIP_RE = re.compile(r"^\s*Điều\s+\d+[a-zđA-ZĐ]?\.?\s*", re.IGNORECASE)
def render_answer(selected_chunks, top_n):
    parts, seen = [], set()
    for c in selected_chunks:
        if c["id"] in seen or len(parts) >= top_n:
            continue
        seen.add(c["id"])
        loai = c["loai_vb"] or "văn bản"
        so = c["so_hieu"] or ""
        dieu = c["dieu_so"]
        lead = f"Căn cứ Điều {dieu} {loai} {so} quy định như sau:" if dieu != "0" else f"Căn cứ {loai} {so} quy định như sau:"
        body = _DIEU_PREFIX_STRIP_RE.sub("", c["text"], count=1) if dieu != "0" else c["text"]
        parts.append(f"{lead}\n{body}")
    return "\n\n".join(parts)

def answer_question(question, bm25, dense_model, dense_embeddings, all_chunks, top_n,
                     reranker_model=None, reranker_tokenizer=None, use_adaptive_k=False):
    ranked = rrf_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks)
    if not ranked:
        return "Không tìm thấy thông tin pháp lý cho câu hỏi này."
    scores = None
    if reranker_model is not None:
        ranked, scores = rerank(question, ranked, reranker_model, reranker_tokenizer)
    n = adaptive_k_cutoff(scores) if (use_adaptive_k and scores is not None) else top_n
    return render_answer(ranked, n)

def try_dev_eval(bm25, dense_model, dense_embeddings, all_chunks, train_data, train_positive,
                  reranker_model=None, reranker_tokenizer=None):
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
        print(f"  Bỏ qua dev-eval ({e})")
        return 3, False, False

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    ids = random.sample(list(train_data.keys()), min(DEV_EVAL_SAMPLE_SIZE, len(train_data)))
    recall_ids = [q for q in ids if q in train_positive]
    print(f"  Dev-eval: {len(ids)} câu ({len(recall_ids)} có citation -> đo Recall)")

    configs = [("RRF", None, None)]
    if reranker_model is not None:
        configs.append(("RRF+rerank", reranker_model, reranker_tokenizer))

    ks = [1,3,5,10,30,100]
    best_n, best_m, best_use_rr, best_use_adaptive = 3, -1.0, False, False

    for label, rr_model, rr_tok in configs:
        ranked_cache, scores_cache = {}, {}
        t0 = time.time()
        for i, qid in enumerate(ids):
            item = train_data[qid]
            ranked = rrf_retrieve(item["question"], bm25, dense_model, dense_embeddings, all_chunks)
            scores = None
            if rr_model is not None and ranked:
                ranked, scores = rerank(item["question"], ranked, rr_model, rr_tok)
            ranked_cache[qid] = ranked
            scores_cache[qid] = scores
            if (i+1) % 50 == 0 or (i+1) == len(ids):
                print(f"    {label} {i+1}/{len(ids)} ... {time.time()-t0:.0f}s")

        if recall_ids:
            hits = {k:0 for k in ks}
            for qid in recall_ids:
                ranked_ids = [c["id"] for c in ranked_cache[qid]]
                pos = train_positive[qid]
                for k in ks:
                    if pos in ranked_ids[:k]:
                        hits[k] += 1
            print(f"  Recall@k ({label}, n={len(recall_ids)}):")
            for k in ks:
                print(f"    Recall@{k:<3d} = {100*hits[k]/len(recall_ids):.1f}%")

        for top_n in range(1, 6):
            ms, rs = [], []
            for qid in ids:
                ranked = ranked_cache[qid]
                pred = render_answer(ranked, top_n) if ranked else "Không tìm thấy..."
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms)/len(ms), sum(rs)/len(rs)
            print(f"    top_n={top_n}  METEOR={m:.4f}  ROUGE-L={r:.4f}")
            if m > best_m:
                best_m, best_n, best_use_rr, best_use_adaptive = m, top_n, (rr_model is not None), False

        if rr_model is not None:
            ms, rs = [], []
            for qid in ids:
                ranked, scores = ranked_cache[qid], scores_cache[qid]
                k = adaptive_k_cutoff(scores) if ranked else 1
                pred = render_answer(ranked, k) if ranked else "Không tìm thấy..."
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms)/len(ms), sum(rs)/len(rs)
            print(f"    adaptive-k       METEOR={m:.4f}  ROUGE-L={r:.4f}")
            if m > best_m:
                best_m, best_use_rr, best_use_adaptive = m, True, True

    print(f"  Chọn: TOP_N={best_n}, rerank={best_use_rr}, adaptive={best_use_adaptive} (METEOR={best_m:.4f})")
    return best_n, best_use_rr, best_use_adaptive

def build_submission(answers, expected_ids, out_zip):
    got = set(answers.keys())
    if got != expected_ids:
        raise ValueError(f"Key mismatch: missing {len(expected_ids-got)}, extra {len(got-expected_ids)}")
    for qid, ans in answers.items():
        if not isinstance(ans, str) or not ans.strip():
            print(f"  [WARN] {qid} empty")
    payload = {qid: {"answer": str(ans)} for qid, ans in answers.items()}
    json_path = out_zip.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="submission.json")
    with zipfile.ZipFile(out_zip) as zf:
        assert zf.namelist() == ["submission.json"]
    print(f"  ✅ {out_zip}")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    checkpoint("Bắt đầu")
    print(f"  Cache: {CACHE_DIR}")

    print("\n=== Bước 1: Chunk corpus ===")
    all_chunks = load_corpus(CONTEXTS_DIR)
    checkpoint("Xong chunking")

    print("\n=== Bước 2: BM25 ===")
    tokenized = [tokenize_simple(f"{c.get('loai_vb','')} {c['text']}") for c in all_chunks]
    bm25 = BM25(tokenized)
    checkpoint("Xong BM25")

    print("\n=== Bước 3: Sinh nhãn ===")
    with open(TRAIN_PATH, encoding="utf-8") as f:
        train_data = json.load(f)
    train_positive, chunk_by_id = build_train_pairs(train_data, all_chunks)
    print(f"  Positive: {len(train_positive)}/{len(train_data)}")
    checkpoint("Xong sinh nhãn")

    print("\n=== Bước 4: Dense retriever ===")
    dense_model = finetune_or_load_dense(train_positive, train_data, chunk_by_id, all_chunks, bm25)
    checkpoint("Xong dense")

    print("\n=== Bước 5: Encode corpus ===")
    dense_embeddings = encode_corpus(dense_model, all_chunks)
    checkpoint("Xong encode")

    print("\n=== Bước 6: Reranker ===")
    reranker_model, reranker_tokenizer = load_reranker()
    checkpoint("Xong reranker")

    print("\n=== Bước 7: Dev-eval ===")
    top_n, use_rerank, use_adaptive = try_dev_eval(
        bm25, dense_model, dense_embeddings, all_chunks,
        train_data, train_positive, reranker_model, reranker_tokenizer
    )
    checkpoint("Xong dev-eval")

    print("\n=== Bước 8: Predict public ===")
    with open(PUBLIC_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    rr_model = reranker_model if use_rerank else None
    rr_tok = reranker_tokenizer if use_rerank else None
    answers = {}
    for i, (qid, item) in enumerate(questions.items()):
        answers[qid] = answer_question(item["question"], bm25, dense_model, dense_embeddings,
                                        all_chunks, top_n, rr_model, rr_tok, use_adaptive)
        if (i+1) % 200 == 0:
            print(f"  ... {i+1}/{len(questions)}  ({elapsed()/60:.1f} phút)")
    empty = sum(1 for a in answers.values() if not a.strip())
    print(f"  {len(answers)} answers, {empty} empty")
    checkpoint("Xong predict")

    print("\n=== Bước 9: Đóng gói ===")
    build_submission(answers, set(questions.keys()), OUT_DIR / "submission.zip")
    checkpoint("XONG")

if __name__ == "__main__":
    main()