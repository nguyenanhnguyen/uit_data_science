"""
legalqa_v2.py — Hai tầng rerank + compose tối ưu, không cần artifact nặng.
Chạy trên RTX 2050 4GB, sử dụng cache để thử nghiệm nhanh.

Cách dùng:
    python legalqa_v2.py
Output: submission.zip (có thể đổi tên nếu muốn)
"""
from __future__ import annotations
import os, sys, re, json, time, pickle, random, zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================== CONFIG ===========================
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "LegalQA_Public_Test"   # nếu bạn có thư mục này
# Nếu không, gán trực tiếp:
# DATA = HERE
CONTEXTS = DATA / "selected-contexts"
TRAIN_PATH = DATA / "train.json"
PUBLIC_PATH = DATA / "public-official.json"

# Nếu không có thư mục data, hãy để file cùng cấp với script:
if not CONTEXTS.exists():
    CONTEXTS = HERE / "selected-contexts"
    TRAIN_PATH = HERE / "train.json"
    PUBLIC_PATH = HERE / "public-official.json"

CACHE = HERE / "cache_v2"
CACHE.mkdir(parents=True, exist_ok=True)
OUT = HERE / "output_v2"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Model ----
DENSE_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"  # hoặc "BAAI/bge-m3"
RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"              # hoặc "xlm-roberta-base"

# ---- Hyperparameters ----
TOP_K_BM25 = 50
TOP_K_DENSE = 30
TOP_K_POOL = 100                # số chunk đưa vào rerank lượt 1
DOC_K = 5                       # số document giữ lại
MAX_DIEU_PER_DOC = 40
MAX_CANDS = 150
KEEP_ARTICLES = 5               # số Điều cuối cùng (top_n compose thực tế)
MAX_LEN_RERANK = 512
BATCH_RERANK = 8                # an toàn cho 4GB

# ---- Compose ----
LEAD = "cancu"                  # "cancu" hoặc "theo" hoặc "none"
CONCL = "echo2"                 # "none", "echo", "echo2", "q"
STRIP_DIEU = True
DROP_DECO = True
TOP_N = 1                       # số Điều đưa vào đáp án (mặc định 1)

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

# =========================== CHUNKING ===========================
DIEU_RE = re.compile(r"(?m)^\s*Điều\s+(\d+[a-zA-ZđĐ]?)\s*[.．:]")
SO_HEADER_RE = re.compile(r"Số\s*[:：]\s*([0-9A-Za-zĐđ/\-]+)")
SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zĐđ]{2,10}(?:-[A-Za-zĐđ]{2,10})?")
LOAI_VB_CANON = ["Thông tư liên tịch", "Nghị định", "Luật", "Thông tư", "Quyết định",
                 "Pháp lệnh", "Nghị quyết", "Bộ luật", "Chỉ thị"]
LOAI_PATTERN = re.compile("(" + "|".join(re.escape(x) for x in LOAI_VB_CANON) + ")", re.IGNORECASE)

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
    """Đọc toàn bộ context_*.json và tạo dict doc_id -> passage + metadata."""
    if load_cache("doc_index"):
        return load_cache("doc_index")
    idx = {}
    for fp in tqdm(list(CONTEXTS.glob("context_*.json")), desc="Loading corpus"):
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        doc_id = str(d["id"])
        passage = d.get("passage", "")
        loai, so = extract_vb_info(passage)
        idx[doc_id] = {"passage": passage, "loai": loai, "so": so, "link": d.get("link", "")}
    save_cache(idx, "doc_index")
    return idx

def chunk_corpus(doc_index):
    """Chia mỗi văn bản thành các Điều, trả về list chunk dict và mapping chunk->doc."""
    chunks = []
    chunk_to_doc = {}
    for doc_id, info in doc_index.items():
        arts = split_dieu(info["passage"])
        if not arts:
            # fallback: toàn bộ passage như một chunk (để xử lý văn bản không có Điều)
            chunks.append({
                "id": f"{doc_id}_0",
                "doc_id": doc_id,
                "dieu": "",
                "text": info["passage"].strip(),
                "loai": info["loai"],
                "so": info["so"]
            })
            chunk_to_doc[f"{doc_id}_0"] = doc_id
        else:
            for dieu, text in arts:
                cid = f"{doc_id}_{dieu}"
                chunks.append({
                    "id": cid,
                    "doc_id": doc_id,
                    "dieu": dieu,
                    "text": text,
                    "loai": info["loai"],
                    "so": info["so"]
                })
                chunk_to_doc[cid] = doc_id
    return chunks, chunk_to_doc

# =========================== BM25 ===========================
def tokenize_vn(text):
    return ViTokenizer.tokenize(text).lower().split()

def build_bm25(chunks):
    tokenized = [tokenize_vn(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)

# =========================== DENSE RETRIEVER ===========================
def load_dense(device):
    model = SentenceTransformer(DENSE_MODEL, device=device)
    model.max_seq_length = 256
    return model

def encode_corpus(model, chunks):
    cache = load_cache("dense_embeddings")
    if cache is not None:
        return cache
    texts = [c["text"] for c in chunks]
    print("Encoding corpus (may take a while)...")
    emb = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    save_cache(emb, "dense_embeddings")
    return emb

# =========================== RERANKER ===========================
def load_reranker(device):
    try:
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
        # Để tương thích với cross-encoder, ta wrap
        return reranker
    except:
        print("FlagReranker không có, dùng CrossEncoder xlm-roberta-base")
        return CrossEncoder("xlm-roberta-base", device=device)

def rerank_pairs(pairs, reranker, batch_size=BATCH_RERANK, max_len=MAX_LEN_RERANK):
    """pairs: list of (query, text)"""
    if hasattr(reranker, "compute_score"):
        scores = reranker.compute_score(pairs, batch_size=batch_size, max_length=max_len, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        return scores
    else:
        # CrossEncoder
        scores = reranker.predict(pairs, batch_size=batch_size)
        return scores

# =========================== STAGE POOL ===========================
def stage_pool(qids, questions, chunks, bm25, dense_model, dense_embeddings):
    """Trả về {qid: [chunk_ids]} top-100"""
    cache = load_cache("pool")
    if cache is not None and all(q in cache for q in qids):
        return cache
    pool = {}
    for qid in tqdm(qids, desc="Pooling"):
        q = questions[qid]
        # BM25
        token_q = tokenize_vn(q)
        bm25_scores = bm25.get_scores(token_q)
        bm25_idx = np.argsort(-bm25_scores)[:TOP_K_BM25]
        # Dense
        q_emb = dense_model.encode([q], normalize_embeddings=True)[0]
        dense_scores = dense_embeddings @ q_emb
        dense_idx = np.argsort(-dense_scores)[:TOP_K_DENSE]
        # RRF
        rank_map = {}
        for r, idx in enumerate(bm25_idx):
            rank_map[idx] = r
        for r, idx in enumerate(dense_idx):
            rank_map[idx] = min(rank_map.get(idx, TOP_K_POOL+1), r)
        # tính điểm RRF
        rrf_scores = {}
        for idx in set(bm25_idx) | set(dense_idx):
            r = rank_map[idx]
            rrf_scores[idx] = 1/(60+r)
        sorted_idx = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:TOP_K_POOL]
        pool[qid] = [chunks[i]["id"] for i in sorted_idx]
    save_cache(pool, "pool")
    return pool

# =========================== STAGE ARTICLE (HAI TẦNG) ===========================
def stage_article(qids, questions, chunks, chunk_to_doc, pool, reranker, doc_index):
    """Trả về {qid: {"docs": [[doc_id, score]], "arts": [[doc_id, dieu, score, text]]}}"""
    cache = load_cache("articles")
    if cache is not None and all(q in cache for q in qids):
        return cache

    # Map chunk_id -> text
    chunk_text = {c["id"]: c["text"] for c in chunks}
    # Map doc_id -> dieu list
    doc_dieu_cache = {}
    def get_dieu(doc_id):
        if doc_id not in doc_dieu_cache:
            doc_dieu_cache[doc_id] = split_dieu(doc_index[doc_id]["passage"])
        return doc_dieu_cache[doc_id]

    articles = {}

    for qid in tqdm(qids, desc="Article rerank"):
        q = questions[qid]
        pool_ids = pool[qid]
        # Lượt 1: rerank chunk -> document score (max-agg)
        pairs = [(q, chunk_text[cid]) for cid in pool_ids if cid in chunk_text]
        if not pairs:
            articles[qid] = {"docs": [], "arts": []}
            continue
        scores = rerank_pairs(pairs, reranker)
        # group by document
        doc_scores = defaultdict(float)
        best_chunk = {}
        for cid, sc in zip(pool_ids, scores):
            doc = chunk_to_doc[cid]
            if sc > doc_scores[doc]:
                doc_scores[doc] = sc
                best_chunk[doc] = cid
        top_docs = sorted(doc_scores, key=doc_scores.get, reverse=True)[:DOC_K]
        docs_info = [[d, float(doc_scores[d])] for d in top_docs]

        # Lượt 2: cắt Điều trong top_docs và rerank
        cands = []
        for doc_id in top_docs:
            dieu_list = get_dieu(doc_id)
            if not dieu_list:
                # fallback: chunk tốt nhất của document này
                cands.append((doc_id, "", chunk_text[best_chunk[doc_id]]))
                continue
            # lọc sơ bộ bằng token overlap (nếu có nhiều Điều)
            if len(dieu_list) > MAX_DIEU_PER_DOC:
                q_tokens = set(q.lower().split())
                dieu_list = sorted(dieu_list, key=lambda a: -len(q_tokens & set(a[1].lower().split())))[:MAX_DIEU_PER_DOC]
            cands.extend((doc_id, d, t) for d, t in dieu_list)
        cands = cands[:MAX_CANDS]
        if not cands:
            articles[qid] = {"docs": docs_info, "arts": []}
            continue
        pairs2 = [(q, t) for _, _, t in cands]
        scores2 = rerank_pairs(pairs2, reranker)
        ranked = sorted(zip(cands, scores2), key=lambda x: -x[1])
        keep = ranked[:KEEP_ARTICLES]
        arts = [[doc_id, dieu, float(sc), text] for (doc_id, dieu, text), sc in keep]
        articles[qid] = {"docs": docs_info, "arts": arts}

    save_cache(articles, "articles")
    return articles

# =========================== STAGE COMPOSE ===========================
DIEU_PREFIX_RE = re.compile(r"^\s*Điều\s+\d+[a-zA-ZđĐ]?\s*[.．:]?\s*")
DECO_RE = re.compile(r"^[\s\-_=–—.·*]+$")

def compose_answer(q, articles, doc_index, args):
    arts = articles[q]["arts"]
    parts = []
    seen = set()
    for doc_id, dieu, sc, text in arts:
        if len(parts) >= args["top_n"]:
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
    if not parts:
        return "Không tìm thấy thông tin pháp lý cho câu hỏi này."
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
    answers = {}
    for q in qids:
        answers[q] = compose_answer(q, articles, doc_index, args)
    return answers

# =========================== SUBMISSION ===========================
def build_submission(answers, expected_ids, out_zip):
    got = set(answers.keys())
    if got != expected_ids:
        raise ValueError(f"Key mismatch: missing {len(expected_ids-got)}, extra {len(got-expected_ids)}")
    bad = [q for q, a in answers.items() if not a.strip()]
    if bad:
        raise ValueError(f"{len(bad)} answers empty: {bad[:5]}")
    payload = {qid: {"answer": str(ans)} for qid, ans in answers.items()}
    js = out_zip.with_suffix(".json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(js, arcname="submission.json")
    print(f"✅ {out_zip} ({len(payload)} answers)")

# =========================== MAIN ===========================
def main():
    print("=== LEGALQA V2 (two-stage rerank) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load data
    print("Loading data...")
    with open(TRAIN_PATH, encoding="utf-8") as f:
        train = json.load(f)
    with open(PUBLIC_PATH, encoding="utf-8") as f:
        public = json.load(f)
    # Chọn subset để dev (ví dụ 300 câu)
    dev_qids = random.sample(list(train.keys()), min(300, len(train)))
    qids = dev_qids + list(public.keys())  # nếu muốn cả public, có thể tách riêng
    questions = {q: train[q]["question"] for q in dev_qids}
    questions.update({q: public[q]["question"] for q in public.keys()})
    print(f"Total questions: {len(qids)} (dev {len(dev_qids)} + public {len(public)})")

    # 2. Corpus
    print("Loading corpus...")
    doc_index = load_corpus()
    chunks, chunk_to_doc = chunk_corpus(doc_index)
    print(f"Total chunks: {len(chunks)}")

    # 3. BM25
    print("Building BM25...")
    bm25 = build_bm25(chunks)

    # 4. Dense
    print("Loading dense model...")
    dense_model = load_dense(device)
    dense_embeddings = encode_corpus(dense_model, chunks)

    # 5. Reranker
    print("Loading reranker...")
    reranker = load_reranker(device)

    # 6. Stage pool
    print("Stage pool...")
    pool = stage_pool(qids, questions, chunks, bm25, dense_model, dense_embeddings)
    checkpoint("pool done")

    # 7. Stage article
    print("Stage article (two-stage rerank)...")
    articles = stage_article(qids, questions, chunks, chunk_to_doc, pool, reranker, doc_index)
    checkpoint("article done")

    # 8. Stage compose
    print("Stage compose...")
    args = {"top_n": TOP_N, "lead": LEAD, "concl": CONCL, "strip_dieu": STRIP_DIEU, "drop_deco": DROP_DECO}
    answers = stage_compose(qids, questions, articles, doc_index, args)
    checkpoint("compose done")

    # (Tùy chọn) đánh giá trên dev nếu có scorer
    try:
        from nltk.translate.meteor_score import meteor_score
        ms = []
        for q in dev_qids:
            ref = train[q]["answer"]
            pred = answers[q]
            ms.append(meteor_score([ref.split()], pred.split()))
        print(f"Dev METEOR (n={len(dev_qids)}): {sum(ms)/len(ms):.4f}")
    except:
        pass

    # 9. Submission
    print("Creating submission.zip...")
    build_submission({q: answers[q] for q in public.keys()}, set(public.keys()), OUT / "submission.zip")
    checkpoint("All done")

if __name__ == "__main__":
    main()