"""
legalqa_final.py — Kết hợp hai tầng rerank (document + article) và template tối ưu.
Chạy trên RTX 2050 4GB VRAM, không giới hạn thời gian.
Output: submission.zip
"""

from __future__ import annotations
import os
import sys
import re
import json
import time
import random
import pickle
import zipfile
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =========================== CONFIG ===========================
HERE = Path(__file__).resolve().parent
CONTEXTS_DIR = HERE / "selected-contexts"
TRAIN_PATH = HERE / "train.json"
PUBLIC_PATH = HERE / "public-official.json"

# ---- Model ----
DENSE_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"   # 135M
RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"               # 568M, zero‑shot

# ---- Hyperparameters ----
DENSE_MAX_SEQ = 256
RERANKER_MAX_LEN = 1024
ENCODE_BATCH = 64
TOP_K_RETRIEVE = 100          # số chunk lấy từ RRF
TOP_K_DOC = 5                 # số document giữ lại sau lượt 1
MAX_DIEU_PER_DOC = 40         # giới hạn số Điều xét trong mỗi doc
MAX_CANDS = 150               # tổng số Điều đưa vào rerank lượt 2
KEEP_ARTICLES = 3             # số Điều cuối cùng (có thể dùng adaptive‑k)
BATCH_RERANK = 8              # batch cho reranker (vừa 4GB VRAM)

# ---- Template ----
LEAD = "cancu"                # "cancu" hoặc "theo"
CONCL = "echo2"               # "none", "echo", "echo2", "q"
STRIP_DIEU = True
DROP_DECO = True
TOP_N = 1                     # mặc định, sẽ được điều chỉnh bởi adaptive‑k

# ---- Cache ----
CACHE = HERE / "cache_final"
CACHE.mkdir(parents=True, exist_ok=True)
OUT = HERE / "output_final"
OUT.mkdir(parents=True, exist_ok=True)

# =========================== UTILITIES ===========================
_START = time.time()
def elapsed():
    return time.time() - _START

def checkpoint(label):
    print(f"[{elapsed()/60:5.1f} phút] {label}")

def save_cache(obj, name):
    with open(CACHE / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

def load_cache(name):
    p = CACHE / f"{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

# =========================== CHUNK / CORPUS ===========================
DIEU_RE = re.compile(r"(?m)^\s*Điều\s+(\d+[a-zA-ZđĐ]?)\s*[.．:]")
SO_HEADER_RE = re.compile(r"Số\s*[:：]\s*([0-9A-Za-zĐđ/\-]+)")
SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zĐđ]{2,10}(?:-[A-Za-zĐđ]{2,10})?")
LOAI_VB_CANON = ["Thông tư liên tịch", "Nghị định", "Luật", "Thông tư", "Quyết định",
                 "Pháp lệnh", "Nghị quyết", "Bộ luật", "Chỉ thị"]
LOAI_PATTERN = re.compile("(" + "|".join(re.escape(x) for x in LOAI_VB_CANON) + ")", re.IGNORECASE)

DIEU_PREFIX_RE = re.compile(r"^\s*Điều\s+\d+[a-zA-ZđĐ]?\s*[.．:]?\s*")
DECO_RE = re.compile(r"^[\s\-_=–—.·*]+$")

def extract_vb_info(passage):
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

def split_dieu(passage):
    ms = list(DIEU_RE.finditer(passage))
    out = []
    for i, m in enumerate(ms):
        end = ms[i+1].start() if i+1 < len(ms) else len(passage)
        out.append((m.group(1), passage[m.start():end].strip()))
    return out

def load_corpus():
    if load_cache("doc_index"):
        return load_cache("doc_index")
    idx = {}
    for fp in tqdm(list(CONTEXTS_DIR.glob("context_*.json")), desc="Loading corpus"):
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        doc_id = str(d["id"])
        passage = d.get("passage", "")
        loai, so = extract_vb_info(passage)
        idx[doc_id] = {"passage": passage, "loai": loai, "so": so, "link": d.get("link", "")}
    save_cache(idx, "doc_index")
    return idx

def chunk_corpus(doc_index):
    chunks = []
    chunk_to_doc = {}
    for doc_id, info in doc_index.items():
        arts = split_dieu(info["passage"])
        if not arts:
            cid = f"{doc_id}_0"
            chunks.append({"id": cid, "doc_id": doc_id, "dieu": "", "text": info["passage"].strip(),
                           "loai": info["loai"], "so": info["so"]})
            chunk_to_doc[cid] = doc_id
        else:
            for dieu, text in arts:
                cid = f"{doc_id}_{dieu}"
                chunks.append({"id": cid, "doc_id": doc_id, "dieu": dieu, "text": text,
                               "loai": info["loai"], "so": info["so"]})
                chunk_to_doc[cid] = doc_id
    return chunks, chunk_to_doc

# =========================== BM25 ===========================
try:
    from pyvi import ViTokenizer
    def tokenize_vn(text):
        return ViTokenizer.tokenize(text).lower().split()
except ImportError:
    def tokenize_vn(text):
        return text.lower().split()

def build_bm25(chunks):
    cache = load_cache("bm25_tokens")
    if cache is not None:
        return BM25Okapi(cache)
    tokenized = []
    for c in tqdm(chunks, desc="BM25 tokenizing"):
        tokenized.append(tokenize_vn(c["text"]))
    save_cache(tokenized, "bm25_tokens")
    return BM25Okapi(tokenized)

# =========================== DENSE ===========================
def load_dense(device):
    model = SentenceTransformer(DENSE_MODEL, device=device)
    model.max_seq_length = DENSE_MAX_SEQ
    return model

def encode_corpus(model, chunks):
    cache = load_cache("dense_embeddings")
    if cache is not None:
        return cache
    texts = [c["text"] for c in chunks]
    print("Encoding corpus (first time, may take a while)...")
    emb = model.encode(texts, batch_size=ENCODE_BATCH, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    save_cache(emb, "dense_embeddings")
    return emb

# =========================== RERANKER ===========================
def load_reranker(device):
    try:
        # Ưu tiên FlagReranker nếu có
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
        return reranker
    except ImportError:
        print("FlagReranker not found, fallback to AutoModel + tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        model = model.to(device)
        if device == "cuda":
            model = model.half()
        model.eval()
        return model, tokenizer

def rerank_pairs(pairs, reranker, batch_size=BATCH_RERANK, max_len=RERANKER_MAX_LEN):
    """pairs: list of (query, text)"""
    if hasattr(reranker, "compute_score"):
        scores = reranker.compute_score(pairs, batch_size=batch_size,
                                        max_length=max_len, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        return np.array(scores)
    else:
        # AutoModel + tokenizer
        model, tokenizer = reranker
        device = next(model.parameters()).device
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_len, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits.view(-1).float().cpu().numpy()
            all_scores.extend(logits)
        return np.array(all_scores)

# =========================== POOL STAGE ===========================
def stage_pool(qids, questions, chunks, bm25, dense_model, dense_embeddings):
    cache = load_cache("pool")
    if cache is not None and all(q in cache for q in qids):
        return cache
    pool = {}
    for qid in tqdm(qids, desc="Pooling"):
        q = questions[qid]
        # BM25
        token_q = tokenize_vn(q)
        bm25_scores = bm25.get_scores(token_q)
        bm25_idx = np.argsort(-bm25_scores)[:50]
        # Dense
        q_emb = dense_model.encode([q], normalize_embeddings=True)[0]
        dense_scores = dense_embeddings @ q_emb
        dense_idx = np.argsort(-dense_scores)[:30]
        # RRF
        rank_map = {}
        for r, idx in enumerate(bm25_idx):
            rank_map[idx] = r
        for r, idx in enumerate(dense_idx):
            rank_map[idx] = min(rank_map.get(idx, 100), r)
        rrf_scores = {}
        for idx in set(bm25_idx) | set(dense_idx):
            r = rank_map[idx]
            rrf_scores[idx] = 1/(60+r)
        sorted_idx = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:TOP_K_RETRIEVE]
        pool[qid] = [chunks[i]["id"] for i in sorted_idx]
    save_cache(pool, "pool")
    return pool

# =========================== TWO-STAGE RERANK ===========================
def stage_article(qids, questions, chunks, chunk_to_doc, pool, reranker, doc_index):
    cache = load_cache("articles")
    if cache is not None and all(q in cache for q in qids):
        return cache

    chunk_text = {c["id"]: c["text"] for c in chunks}
    doc_dieu_cache = {}
    def get_dieu(doc_id):
        if doc_id not in doc_dieu_cache:
            doc_dieu_cache[doc_id] = split_dieu(doc_index[doc_id]["passage"])
        return doc_dieu_cache[doc_id]

    articles = {}
    for qid in tqdm(qids, desc="Two-stage rerank"):
        q = questions[qid]
        pool_ids = pool[qid]

        # Lượt 1: chunk → document (max-agg)
        pairs = [(q, chunk_text[cid]) for cid in pool_ids if cid in chunk_text]
        if not pairs:
            articles[qid] = {"docs": [], "arts": []}
            continue
        scores = rerank_pairs(pairs, reranker, batch_size=BATCH_RERANK)
        doc_scores = defaultdict(float)
        best_chunk = {}
        for cid, sc in zip(pool_ids, scores):
            doc = chunk_to_doc[cid]
            if sc > doc_scores[doc]:
                doc_scores[doc] = sc
                best_chunk[doc] = cid
        top_docs = sorted(doc_scores, key=doc_scores.get, reverse=True)[:TOP_K_DOC]
        docs_info = [[d, float(doc_scores[d])] for d in top_docs]

        # Lượt 2: cắt Điều trong top_docs
        cands = []
        for doc_id in top_docs:
            dieu_list = get_dieu(doc_id)
            if not dieu_list:
                # fallback: chunk tốt nhất của doc
                cands.append((doc_id, "", chunk_text[best_chunk[doc_id]]))
                continue
            if len(dieu_list) > MAX_DIEU_PER_DOC:
                q_tokens = set(q.lower().split())
                dieu_list = sorted(dieu_list,
                                   key=lambda a: -len(q_tokens & set(a[1].lower().split()))
                                  )[:MAX_DIEU_PER_DOC]
            cands.extend((doc_id, d, t) for d, t in dieu_list)
        cands = cands[:MAX_CANDS]
        if not cands:
            articles[qid] = {"docs": docs_info, "arts": []}
            continue

        pairs2 = [(q, t) for _, _, t in cands]
        scores2 = rerank_pairs(pairs2, reranker, batch_size=BATCH_RERANK)
        ranked = sorted(zip(cands, scores2), key=lambda x: -x[1])
        keep = ranked[:KEEP_ARTICLES]
        arts = [[doc_id, dieu, float(sc), text] for (doc_id, dieu, text), sc in keep]
        articles[qid] = {"docs": docs_info, "arts": arts}

    save_cache(articles, "articles")
    return articles

# =========================== COMPOSE ===========================
def adaptive_k_cutoff(scores, min_k=1, max_k=5, window=15):
    if scores is None or len(scores) == 0:
        return min_k
    n = min(len(scores), window)
    if n <= 1:
        return min_k
    gaps = [scores[i] - scores[i+1] for i in range(n-1)]
    k = int(np.argmax(gaps)) + 1
    return max(min_k, min(k, max_k))

def compose_answer(q, articles, doc_index, args):
    arts = articles[q]["arts"]
    if not arts:
        return "Không tìm thấy thông tin pháp lý cho câu hỏi này."

    # Nếu dùng adaptive-k, lấy điểm số của các Điều để chọn số lượng
    scores = [sc for _, _, sc, _ in arts]
    top_n = adaptive_k_cutoff(scores) if args["use_adaptive"] else args["top_n"]
    top_n = min(top_n, len(arts))

    parts = []
    seen = set()
    for doc_id, dieu, sc, text in arts:
        if len(parts) >= top_n:
            break
        if doc_id in seen:
            continue
        seen.add(doc_id)
        loai = doc_index[doc_id].get("loai", "văn bản")
        so = doc_index[doc_id].get("so", "")
        body = text
        if args["strip_dieu"] and dieu:
            body = DIEU_PREFIX_RE.sub("", body, count=1)
        if args["drop_deco"]:
            body = "\n".join(l for l in body.split("\n") if not DECO_RE.match(l))
        head = f"Điều {dieu} " if dieu else ""
        tail = " ".join(x for x in (loai, so) if x)
        if args["lead"] == "none":
            lead = ""
        else:
            verb = "Căn cứ" if args["lead"] == "cancu" else "Theo"
            lead = f"{verb} {head}{tail} quy định như sau:"
        parts.append(f"{lead}\n{body}".strip() if lead else body.strip())

    ans = "\n\n".join(parts)
    if args["concl"] != "none":
        qclean = q.strip().rstrip("?").strip()
        if qclean:
            ql = qclean[0].lower() + qclean[1:]
            if args["concl"] == "echo":
                ans += f"\nNhư vậy, theo quy định nêu trên thì {ql}."
            elif args["concl"] == "echo2":
                ans += f"\nTheo đó, {ql}.\nNhư vậy, theo quy định nêu trên thì {ql}."
            elif args["concl"] == "q":
                ans += "\n" + qclean + "."
    return ans

def stage_compose(qids, questions, articles, doc_index, args):
    return {q: compose_answer(q, articles, doc_index, args) for q in qids}

# =========================== SUBMISSION ===========================
def build_submission(answers, expected_ids, out_zip):
    got = set(answers.keys())
    if got != expected_ids:
        raise ValueError(f"Key mismatch: missing {len(expected_ids-got)}, extra {len(got-expected_ids)}")
    for qid, ans in answers.items():
        if not isinstance(ans, str) or not ans.strip():
            print(f"  [WARN] {qid} answer empty")
    payload = {qid: {"answer": str(ans)} for qid, ans in answers.items()}
    js = out_zip.with_suffix(".json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(js, arcname="submission.json")
    print(f"✅ {out_zip} ({len(payload)} answers)")

# =========================== DEV EVAL ===========================
def try_dev_eval(dev_qids, train, questions, articles, doc_index, args):
    try:
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        from nltk.translate.meteor_score import meteor_score
    except Exception as e:
        print(f"Dev eval skipped: {e}")
        return
    answers = stage_compose(dev_qids, questions, articles, doc_index, args)
    ms = []
    for qid in dev_qids:
        ref = train[qid]["answer"]
        pred = answers[qid]
        ms.append(meteor_score([ref.split()], pred.split()))
    m = sum(ms)/len(ms)
    print(f"Dev METEOR (n={len(dev_qids)}): {m:.4f}")
    return m

# =========================== MAIN ===========================
def main():
    print("=== LEGALQA FINAL (Two-stage rerank + template optimized) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    with open(TRAIN_PATH, encoding="utf-8") as f:
        train = json.load(f)
    with open(PUBLIC_PATH, encoding="utf-8") as f:
        public = json.load(f)

    # Dev split (300 câu, có thể tăng nếu muốn)
    random.seed(42)
    dev_qids = random.sample(list(train.keys()), min(300, len(train)))
    qids = dev_qids + list(public.keys())
    questions = {q: train[q]["question"] for q in dev_qids}
    questions.update({q: public[q]["question"] for q in public.keys()})
    print(f"Total: {len(qids)} (dev {len(dev_qids)} + public {len(public)})")

    # Corpus
    doc_index = load_corpus()
    chunks, chunk_to_doc = chunk_corpus(doc_index)
    print(f"Chunks: {len(chunks)}")

    # BM25
    bm25 = build_bm25(chunks)

    # Dense
    dense_model = load_dense(device)
    dense_embeddings = encode_corpus(dense_model, chunks)

    # Reranker
    print("Loading reranker...")
    reranker = load_reranker(device)

    # Stage pool
    print("Stage pool...")
    pool = stage_pool(qids, questions, chunks, bm25, dense_model, dense_embeddings)
    checkpoint("pool done")

    # Stage article (two-stage rerank)
    print("Stage article (two-stage rerank)...")
    articles = stage_article(qids, questions, chunks, chunk_to_doc, pool, reranker, doc_index)
    checkpoint("article done")

    # Compose args
    args = {
        "top_n": TOP_N,
        "lead": LEAD,
        "concl": CONCL,
        "strip_dieu": STRIP_DIEU,
        "drop_deco": DROP_DECO,
        "use_adaptive": True,  # bật adaptive‑k
    }

    # Dev eval (optional)
    if dev_qids:
        dev_score = try_dev_eval(dev_qids, train, questions, articles, doc_index, args)

    # Final answers for public
    print("Composing answers for public...")
    answers = stage_compose(list(public.keys()), questions, articles, doc_index, args)

    # Submission
    out_zip = OUT / "submission.zip"
    build_submission(answers, set(public.keys()), out_zip)
    checkpoint("All done")

if __name__ == "__main__":
    main()