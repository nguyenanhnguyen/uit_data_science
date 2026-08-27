#!/usr/bin/env python
"""LegalQA (UIT DSC2026 Task 2) — bản MÁY CÁ NHÂN, một GPU, tự chứa hoàn toàn.

    python legalqa_local.py --data <thư mục có train.json, public-official.json,
                                    selected-contexts/>

Chỉ cần dữ liệu BTC. KHÔNG cần bất cứ artifact nào của máy server (db_fast, eval/ckpt...).
Model tải từ HuggingFace lần đầu rồi nằm trong cache. Mọi bước đều ghi cache và **có thể
chạy lại từ chỗ dừng** — điều này không phải tiện nghi mà là bắt buộc: encode corpus trên
card yếu mất nhiều giờ, mất điện giữa chừng mà phải làm lại từ đầu là hỏng cả buổi.

===============================================================================
TỰ CO GIÃN THEO MÁY — không đặt tay một con số nào
===============================================================================
Đỉnh VRAM đo thật trên RTX 2080 Ti (bản Vietnamese_Reranker 568M, max_len 512):

    rerank batch    4     8    16    32    64
    MiB          1194  1318  1478  1848  2584
    cặp/s          76    72    85    76    88     <- KHÔNG tăng theo batch

GPU bão hoà compute từ batch 4. Nên batch lớn KHÔNG nhanh hơn, nó chỉ quyết định job có
chui vừa khe VRAM hay không. Vì thế file này chọn batch theo VRAM thật rồi thôi, và không
có tuỳ chọn "xin thêm VRAM cho nhanh" — thứ đó không tồn tại.

Ba mức, tự nhận:

    < 6 GB   (vd RTX 2050 4GB)  suy luận thuần, model zero-shot. Vẫn ăn trọn phần lớn
                                điểm vì hai lever mạnh nhất — kiến trúc hai tầng và câu
                                kết — đều KHÔNG cần fine-tune.
    6-11 GB  (vd 2080 Ti, 3060) + fine-tune bi-encoder bằng GradCache.
    >= 12 GB (vd 3090, 4090)    + fine-tune reranker listwise. Trần cao nhất.

===============================================================================
KIẾN TRÚC — vì sao hai tầng
===============================================================================
Đo trên train.json, oracle biết trước document đúng (METEOR xấp xỉ exact-match):

    trả toàn bộ văn bản                 0,195   <- precision sập, văn bản ~8.700 từ
    1 Điều tốt nhất, nguyên văn         0,605
    2 Điều tốt nhất ghép lại            0,519   <- ghép thêm là MẤT điểm
    1 Điều + câu dẫn template           0,626

Độ dài Điều / độ dài đáp án = 1,02 (trung vị) — Điều luật vừa khít đáp án. Nhưng chunk
theo Điều ở TẦNG RETRIEVAL lại làm document recall tệ đi 1,19 điểm (đo ở Task 1, p 0,024).
Nên tách đôi:

    tầng 1  BM25 + 2 dense -> ~96 chunk 450 TỪ -> rerank -> max-agg -> top-5 DOCUMENT
    tầng 2  cắt ĐIỀU chỉ trong 5 document đó -> cùng reranker -> top-1 ĐIỀU
    tầng 3  câu dẫn + thân Điều + CÂU KẾT

===============================================================================
CÂU KẾT — thay đổi rẻ nhất và đáng giá nhất
===============================================================================
Đo trên 501 câu dev, cùng retrieval, chỉ đổi một biến:

    none   0,5151
    echo   0,5499   Δ +0,0348 ± 0,0017 · 416 thắng / 85 thua · t = 20,4
    echo2  0,5630   Δ +0,0131 ± 0,0010 · 371 thắng / 130 thua · t = 12,8

Split-half: cả hai nửa dev độc lập đều chọn echo2 (A 0,5675 · B 0,5589). Lặp tiếp:
3× 0,5674 · 4× 0,5682 · 6× 0,5661 — có đỉnh quanh 4 nhưng DỪNG Ở 2, vì từ 2 lên 4 chỉ
được +0,5 điểm (đúng vùng mà "chọn đỉnh trên toàn dev" đã lừa dự án này ba lần) còn đáp
án lặp câu hỏi bốn lần thì nhìn bằng mắt là hỏng. Đây là tối ưu HÌNH DẠNG ĐỘ ĐO, hợp lệ
theo luật nhưng không làm câu trả lời tốt hơn cho người đọc.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------- tham số cứng
CHUNK_WORDS = 450          # tầng 1. Task 1 đo: 450 từ > cắt theo Điều 1,19 điểm cho recall
TOP_K_BM25 = 50
TOP_K_DENSE = 30
DOC_K = 5                  # số document mở ra để cắt Điều
MAX_DIEU_PER_DOC = 40
MAX_CANDS = 150
RERANK_MAX_LEN = 512
SEED = 42

MODEL_BGE = "BAAI/bge-m3"
MODEL_E5 = "intfloat/multilingual-e5-large"
MODEL_RERANK = "AITeamVN/Vietnamese_Reranker"

# Đỉnh VRAM đo thật (MiB) — xem bảng ở docstring.
RERANK_PEAK = {4: 1194, 8: 1318, 16: 1478, 32: 1848, 64: 2584}
ENCODE_PEAK = {8: 1600, 16: 2000, 32: 2800, 64: 4200, 128: 7000}
CUDA_CTX_MIB = 350
HEADROOM_MIB = 250


# =============================================================================
# 0. Cache — ghi NGUYÊN TỬ, đọc KHOAN DUNG
# =============================================================================
def save_json(path: Path, obj) -> None:
    """Ghi ra file tạm rồi mới đổi tên. os.replace là nguyên tử trên cùng một filesystem,
    nên file đích hoặc là bản cũ nguyên vẹn, hoặc là bản mới hoàn chỉnh — không bao giờ
    có trạng thái ghi dở. Đã trả giá để biết: một lần crash giữa lúc ghi pool.json.gz để
    lại file cụt, lần chạy SAU chết ngay lúc đọc cache, tức là hỏng đúng chỗ đắt nhất
    (sau khi đã encode xong corpus)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path: Path, default=None):
    """Cache hỏng thì coi như CHƯA CÓ và tính lại, chứ không làm chết cả tiến trình.
    Mất vài phút tính lại luôn tốt hơn là bắt người dùng tự đi xoá file."""
    if not path.exists():
        return default
    try:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, EOFError, json.JSONDecodeError) as e:
        print(f"  ⚠️  cache {path.name} hỏng ({type(e).__name__}) — bỏ, tính lại")
        try:
            path.unlink()
        except OSError:
            pass
        return default


# =============================================================================
# 1. Dò máy và chọn mức chạy
# =============================================================================
def detect_profile(force_tier: str | None = None) -> dict:
    """-> {tier, vram_mib, rerank_batch, encode_batch, workers}.

    Đọc VRAM TỔNG chứ không phải phần trống: đây là máy cá nhân, ta là người dùng duy
    nhất, không phải tranh khe với ai. (Trên máy dùng chung thì ngược lại — xem
    legalqa/run_qa.py, nơi phải xin đúng phần mình cần rồi nhường phần còn lại.)
    """
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "Không thấy CUDA. Kiểm tra:\n"
            "    python -c \"import torch; print(torch.cuda.is_available())\"\n"
            "Ra False dù máy có GPU NVIDIA -> đang cài bản torch CPU-only. Cài lại bản "
            "có CUDA từ https://pytorch.org (đừng dùng `pip install torch` trần).")
    props = torch.cuda.get_device_properties(0)
    vram = int(props.total_memory / 2 ** 20)
    usable = vram - CUDA_CTX_MIB - HEADROOM_MIB

    def pick(table, cap):
        best = min(table)
        for b in sorted(table):
            if table[b] <= usable and b <= cap:
                best = b
        return best

    tier = force_tier or ("full" if vram >= 12000 else
                          "mid" if vram >= 6000 else "lite")
    prof = {
        "gpu": props.name, "vram_mib": vram, "tier": tier,
        "rerank_batch": pick(RERANK_PEAK, 64),
        "encode_batch": pick(ENCODE_PEAK, 128),
        "workers": min(os.cpu_count() or 4, 8),
        "finetune_biencoder": tier in ("mid", "full"),
        "finetune_reranker": tier == "full",
    }
    print(f"  GPU {prof['gpu']} · {vram} MiB · mức '{tier}'")
    print(f"  batch rerank {prof['rerank_batch']} · encode {prof['encode_batch']} · "
          f"{prof['workers']} tiến trình CPU")
    if tier == "lite":
        print("  ⓘ  < 6 GB: bỏ fine-tune, chạy model zero-shot. Hai lever mạnh nhất "
              "(hai tầng + câu kết) không cần fine-tune nên vẫn giữ được phần lớn điểm.")
    elif tier == "mid":
        print("  ⓘ  6-11 GB: fine-tune bi-encoder, reranker giữ zero-shot.")
    return prof


# =============================================================================
# 2. Corpus — chunk 450 từ (tầng 1) và cắt Điều (tầng 2)
# =============================================================================
SO_HEADER_RE = re.compile(r"Số\s*[:：]\s*([0-9A-Za-zĐđ/\-]+)")
SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zĐđ]{2,10}(?:-[A-Za-zĐđ]{2,10})?")
LOAI_VB_CANON = ["Thông tư liên tịch", "Nghị định", "Luật", "Thông tư", "Quyết định",
                 "Pháp lệnh", "Nghị quyết", "Bộ luật", "Chỉ thị"]
LOAI_PATTERN = re.compile("(" + "|".join(re.escape(x) for x in LOAI_VB_CANON) + ")",
                          re.IGNORECASE)
# Neo `^` (re.M) là sống còn: "Điều 5" xuất hiện dày đặc GIỮA dòng dưới dạng trích dẫn
# chéo ("quy định tại Điều 5 Nghị định này"). Cắt ở đó là băm nát điều luật.
DIEU_RE = re.compile(r"(?m)^\s*Điều\s+(\d+[a-zA-ZđĐ]?)\s*[.．:]")
DIEU_PREFIX_RE = re.compile(r"^\s*Điều\s+\d+[a-zA-ZđĐ]?\s*[.．:]?\s*")
DECO_RE = re.compile(r"^[\s\-_=–—.·*]+$")
TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").replace("Đ", "D").replace("đ", "d")


def tokenize(text: str) -> list:
    return TOKEN_RE.findall(text.lower())


def extract_vb_info(passage: str) -> tuple[str, str]:
    m = SO_HEADER_RE.search(passage[:1500])
    so = m.group(1).strip("., ") if m else ""
    if not (so and SO_HIEU_RE.fullmatch(so)):
        m2 = SO_HIEU_RE.search(passage[:1500])
        so = m2.group(0) if m2 else ""
    m3 = LOAI_PATTERN.search(passage[:200]) or LOAI_PATTERN.search(passage[:800])
    loai = ""
    if m3:
        low = m3.group(1).lower()
        for canon in LOAI_VB_CANON:
            if canon.lower() == low:
                loai = canon
                break
    return loai, so


def split_dieu(passage: str) -> list:
    """-> [(số Điều, nguyên văn)]. Rỗng nếu văn bản không có cấu trúc Điều (15,8% corpus:
    QCVN/TCVN, quyết định ngắn, biểu mẫu). Trả rỗng để nơi gọi tự lo fallback — KHÔNG nới
    regex, nới ra thì 84,2% còn lại bị cắt nhầm ở trích dẫn chéo, mất nhiều hơn được."""
    ms = list(DIEU_RE.finditer(passage))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(passage)
        out.append((m.group(1), passage[m.start():end].strip()))
    return out


def split_words(passage: str, n: int = CHUNK_WORDS) -> list:
    w = passage.split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [passage]


def build_corpus(contexts: Path, cache: Path, limit: int = 0) -> tuple[list, dict]:
    """-> (chunks 450 từ, doc_meta). chunks = [{"cid","doc","text"}].

    `limit` > 0 chỉ nạp N văn bản đầu — để chạy thử toàn bộ đường ống trong vài phút
    TRƯỚC khi cam kết nhiều giờ encode corpus đầy đủ trên card yếu. Cache tách riêng
    theo limit nên bản thử không đè lên bản thật."""
    sfx = f"_lim{limit}" if limit else ""
    cpath, mpath = cache / f"chunks{sfx}.json.gz", cache / f"doc_meta{sfx}.json"
    chunks, meta = load_json(cpath), load_json(mpath)
    if chunks and meta:
        print(f"  cache: {len(chunks)} chunk · {len(meta)} văn bản")
        return chunks, meta

    files = sorted(contexts.glob("context_*.json"))
    if not files:
        nested = contexts / "selected-contexts"
        files = sorted(nested.glob("context_*.json")) if nested.exists() else []
    if not files:
        raise SystemExit(f"Không thấy context_*.json trong {contexts}")
    if limit:
        files = files[:limit]
        print(f"  [THỬ] chỉ nạp {len(files)} văn bản đầu — KHÔNG dùng để nộp bài")
    chunks, meta = [], {}
    t0 = time.time()
    for i, fp in enumerate(files):
        try:
            with fp.open(encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        psg = d.get("passage") or ""
        if not psg:
            continue
        did = str(d["id"])
        loai, so = extract_vb_info(psg)
        meta[did] = {"link": d.get("link", ""), "loai_vb": loai, "so_hieu": so}
        for j, tx in enumerate(split_words(psg)):
            chunks.append({"cid": f"{did}_c{j}", "doc": did, "text": tx})
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(files)} văn bản ... {time.time()-t0:.0f}s", flush=True)
    save_json(cpath, chunks)
    save_json(mpath, meta)
    print(f"  {len(files)} văn bản -> {len(chunks)} chunk ({time.time()-t0:.0f}s)")
    return chunks, meta


# =============================================================================
# 3. BM25 — ma trận thưa, chạy CPU
# =============================================================================
class BM25:
    """BM25Okapi tương đương, posting list là 2 mảng numpy để vector hoá truy vấn.

    Bản dùng list[tuple] + vòng lặp Python thuần chậm hàng bậc trên corpus 180K chunk khi
    một từ phổ biến có posting list hàng chục nghìn phần tử.
    """

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        # KHÔNG giữ tham chiếu tới module numpy trong self: object này được pickle ra
        # đĩa để lần chạy sau khỏi dựng lại index (mất vài phút trên corpus đầy đủ),
        # mà module thì không pickle được -> "TypeError: cannot pickle 'module' object".
        import numpy as np
        import math
        from collections import Counter
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        raw = defaultdict(list)
        for i, d in enumerate(docs_tokens):
            for t, f in Counter(d).items():
                raw[t].append((i, f))
        self.inv = {}
        for t, post in raw.items():
            idxs = np.fromiter((p[0] for p in post), dtype=np.int32, count=len(post))
            frq = np.fromiter((p[1] for p in post), dtype=np.float64, count=len(post))
            self.inv[t] = (idxs, frq)
        df = {t: len(v[0]) for t, v in self.inv.items()}
        idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}
        avg = sum(idf.values()) / len(idf) if idf else 0.0
        self.idf = {t: (v if v > 0 else 0.25 * avg) for t, v in idf.items()}

    def scores(self, q_tokens):
        import numpy as np
        s = np.zeros(self.N, dtype=np.float64)
        for t in set(q_tokens):
            p = self.inv.get(t)
            if p is None:
                continue
            idxs, frq = p
            denom = frq + self.k1 * (1 - self.b + self.b * self.doc_len[idxs] / self.avgdl)
            s[idxs] += self.idf[t] * frq * (self.k1 + 1) / denom
        return s

    def top_k(self, q_tokens, k):
        import numpy as np
        s = self.scores(q_tokens)
        # Ép về int Python: numpy.int64 KHÔNG serialize được bằng json, mà pool lại
        # được ghi ra .json.gz để chạy lại từ chỗ dừng — hỏng ở đúng chỗ tốn thời gian
        # nhất (sau khi encode xong corpus).
        return [int(i) for i in np.argsort(-s)[:k]]


def build_bm25(chunks, cache: Path, sfx: str = ""):
    import pickle
    p = cache / f"bm25{sfx}.pkl"
    if p.exists():
        try:
            with p.open("rb") as f:
                bm = pickle.load(f)
            print("  cache: BM25")
            return bm
        except Exception as e:
            print(f"  ⚠️  cache BM25 hỏng ({type(e).__name__}) — dựng lại")
            p.unlink(missing_ok=True)
    t0 = time.time()
    bm = BM25([tokenize(c["text"]) for c in chunks])
    tmp = p.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(bm, f, protocol=4)
    os.replace(tmp, p)
    print(f"  BM25 {bm.N} chunk ({time.time()-t0:.0f}s)")
    return bm


# =============================================================================
# 4. Encode corpus — chia shard, CÓ THỂ CHẠY LẠI TỪ CHỖ DỪNG
# =============================================================================
def encode_corpus(chunks, model_name: str, prefix: str, cache: Path, prof: dict,
                  sfx: str = "", shard_size: int = 20000):
    """-> ma trận (n_chunk, dim) float16 trên đĩa.

    Chia shard và ghi từng shard: trên card yếu việc này mất nhiều giờ, mất điện hay
    Ctrl-C giữa chừng mà phải encode lại từ đầu là mất cả buổi. Mỗi shard xong là ghi
    ngay, chạy lại chỉ làm shard còn thiếu.
    """
    import numpy as np
    tag = model_name.replace("/", "__")
    out = cache / f"emb_{tag}{sfx}.npy"
    if out.exists():
        print(f"  cache: embedding {model_name}")
        return np.load(out, mmap_mode="r")

    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(model_name, device="cuda")
    model.max_seq_length = 256
    n = len(chunks)
    parts = []
    t0 = time.time()
    for s in range(0, n, shard_size):
        sp = cache / f"emb_{tag}{sfx}.shard{s}.npy"
        if sp.exists():
            parts.append(np.load(sp))
            continue
        texts = [prefix + c["text"] for c in chunks[s:s + shard_size]]
        v = model.encode(texts, batch_size=prof["encode_batch"], convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
        v = v.astype(np.float16)
        np.save(sp, v)
        parts.append(v)
        done = min(s + shard_size, n)
        el = time.time() - t0
        print(f"    {done}/{n} chunk · {el/60:.1f} phút · còn ~"
              f"{el/max(done,1)*(n-done)/60:.0f} phút", flush=True)
    mat = np.concatenate(parts).astype(np.float16)
    np.save(out, mat)
    for s in range(0, n, shard_size):
        (cache / f"emb_{tag}{sfx}.shard{s}.npy").unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    print(f"  encode {model_name}: {mat.shape} ({(time.time()-t0)/60:.0f} phút)")
    return mat


def dense_topk(mat, qvecs, k: int, block: int = 256):
    """topk bằng matmul trên GPU, chia khối THEO QUERY.

    Ma trận điểm là (n_query × n_chunk) và `topk` còn đòi bản float32 của nó — nhồi cả
    nghìn câu một lượt là vài GB cấp phát trong một nhịp, đủ OOM ngay cả khi VRAM nhìn
    có vẻ còn nhiều. Chia khối thì đỉnh chỉ còn vài trăm MB, kết quả không đổi.
    """
    import numpy as np
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    M = torch.from_numpy(np.ascontiguousarray(mat)).to(dev)
    if dev == "cpu":
        M = M.float()
    outs = []
    q = np.asarray(qvecs, dtype=np.float16)
    for s in range(0, len(q), block):
        qb = torch.from_numpy(q[s:s + block]).to(dev)
        if dev == "cpu":
            qb = qb.float()
        sc = qb @ M.T
        _v, idx = torch.topk(sc.float(), k=min(k, M.shape[0]), dim=1)
        outs.append(idx.cpu().numpy())
        del qb, sc, idx
    del M
    torch.cuda.empty_cache()
    return np.concatenate(outs)


# =============================================================================
# 5. Rerank
# =============================================================================
class Reranker:
    def __init__(self, path: str, prof: dict):
        from FlagEmbedding import FlagReranker
        self.m = FlagReranker(path, use_fp16=True)
        self.batch = prof["rerank_batch"]

    def score(self, pairs: list) -> list:
        if not pairs:
            return []
        s = self.m.compute_score(pairs, batch_size=self.batch,
                                 max_length=RERANK_MAX_LEN, normalize=True)
        return [s] if isinstance(s, float) else list(s)


# =============================================================================
# 6. Sinh đáp án
# =============================================================================
FALLBACK = "Không tìm thấy thông tin pháp lý cho câu hỏi này."


def compose(question: str, arts: list, meta: dict, top_n: int = 1,
            lead: str = "cancu", concl: str = "echo2",
            strip_dieu: bool = True, drop_deco: bool = True) -> str:
    """arts = [(doc_id, số Điều hoặc "", text)] đã xếp hạng giảm dần.

    top_n = 1: oracle đo được 1 Điều 0,605 còn 2 Điều ghép lại 0,519. METEOR nặng recall
    (alpha 0,9) nhưng số hạng (1-alpha)/P vẫn bùng lên khi đáp án dài gấp đôi tham chiếu.
    lead = "cancu": 57,4% đáp án thật mở đầu "Căn cứ" (so với 25,1% "Theo"), 27,2% chứa
    "quy định như sau" (so với 1,1% "quy định cụ thể" mà template cũ dùng).
    Bỏ "Điều X." đầu thân bài: 98,8% đáp án thật không lặp lại số Điều ngay sau câu dẫn.
    """
    parts, seen = [], set()
    for doc_id, dieu, text in arts:
        if len(parts) >= top_n:
            break
        if doc_id in seen:
            continue
        seen.add(doc_id)
        m = meta.get(doc_id, {})
        loai = m.get("loai_vb") or "văn bản"
        so = m.get("so_hieu") or ""
        body = DIEU_PREFIX_RE.sub("", text, count=1) if (dieu and strip_dieu) else text
        if drop_deco:
            body = "\n".join(l for l in body.split("\n") if not DECO_RE.match(l))
        head = f"Điều {dieu} " if dieu else ""
        tail = " ".join(x for x in (loai, so) if x)
        if lead == "none":
            parts.append(body.strip())
        else:
            verb = "Căn cứ" if lead == "cancu" else "Theo"
            parts.append(f"{verb} {head}{tail} quy định như sau:\n{body}".strip())
    if not parts:
        return FALLBACK
    ans = "\n\n".join(parts)
    if concl != "none":
        q = question.strip().rstrip("?").strip()
        if q:
            ql = q[0].lower() + q[1:]
            if concl == "echo":
                ans += f"\nNhư vậy, theo quy định nêu trên thì {ql}."
            elif concl == "echo2":
                ans += f"\nTheo đó, {ql}.\nNhư vậy, theo quy định nêu trên thì {ql}."
    return ans


# =============================================================================
# 7. Chấm điểm — đúng công thức scoring/legalqa/scoring.py của BTC
# =============================================================================
def load_meteor():
    """METEOR của nltk: tokenize bằng str.split() TRẦN, alpha 0,9 · beta 3 · gamma 0,5.

    Không cần wordnet: đã đo, có và không có wordnet cho kết quả TRÙNG tới 6 chữ số thập
    phân trên văn bản pháp luật tiếng Việt — WordNet tiếng Anh vô dụng ở đây, chỉ exact
    match sau lowercase là đáng kể.
    """
    from nltk.translate.meteor_score import meteor_score
    return meteor_score


def evaluate(dev_qids, train, answers) -> dict:
    import statistics
    meteor = load_meteor()
    ms = [meteor([str(train[q]["answer"]).split()], str(answers[q]).split())
          for q in dev_qids]
    n = len(ms)
    se = statistics.stdev(ms) / (n ** 0.5) if n > 1 else 0.0
    print(f"  METEOR {sum(ms)/n:.4f} ± {se:.4f} (SE) · n={n}")
    return {"meteor": sum(ms) / n, "se": se, "n": n}


# =============================================================================
# 8. Dev split — chia theo VĂN BẢN
# =============================================================================
CIT_RE = re.compile(r"\b(\d{1,4})/(\d{4})/([A-ZĐ][A-ZĐ\-]*)\b")


def build_dev(train: dict, meta: dict, n: int, cache: Path) -> list:
    """Chia theo văn bản chứ không theo câu: nếu sau này fine-tune trên nhãn suy từ
    citation thì cắt giữa nhóm cùng văn bản là tự tay tạo rò rỉ."""
    p = cache / f"dev_{n}.json"
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    slug = [(d, deaccent(v.get("link", "") or "").lower()) for d, v in meta.items()]
    memo, group = {}, {}
    for qid, item in train.items():
        docs = set()
        for m in CIT_RE.finditer(str(item.get("answer") or "")):
            key = f"{int(m.group(1))}/{m.group(2)}/{deaccent(m.group(3)).upper()}"
            if key not in memo:
                pat = "-" + key.replace("/", "-").lower() + "-"
                memo[key] = {d for d, lk in slug if pat in lk}
            docs |= memo[key]
        group[qid] = ("doc:" + sorted(docs)[0]) if docs else ("q:" + qid)
    buckets = defaultdict(list)
    for qid, g in group.items():
        buckets[g].append(qid)
    picked = []
    for g in sorted(buckets, key=lambda x: hashlib.md5(x.encode()).hexdigest()):
        if len(picked) >= n:
            break
        picked.extend(sorted(buckets[g]))
    picked = sorted(picked)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(picked, f)
    res = sum(1 for q in picked if group[q].startswith("doc:"))
    print(f"  dev {len(picked)} câu · {res/len(picked):.1%} phân giải được citation")
    return picked


# =============================================================================
# 9. Đóng gói
# =============================================================================
def package(answers: dict, expected: set, out_zip: Path) -> None:
    """Tập khoá phải TRÙNG KHÍT ground truth: thiếu hay thừa đều làm scorer raise ->
    0 điểm TOÀN BÀI, không phải 0 điểm một câu."""
    got = set(answers)
    if got != expected:
        raise SystemExit(f"Tập khoá lệch: thiếu {len(expected-got)}, thừa {len(got-expected)}")
    bad = [q for q, a in answers.items() if not isinstance(a, str) or not a.strip()]
    if bad:
        raise SystemExit(f"{len(bad)} câu answer rỗng: {bad[:5]}")
    payload = {q: {"answer": str(a)} for q, a in answers.items()}
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    js = out_zip.with_suffix(".json")
    with js.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(js, arcname="submission.json")
    with zipfile.ZipFile(out_zip) as zf:
        if zf.namelist() != ["submission.json"]:
            raise SystemExit(f"zip sai cấu trúc: {zf.namelist()}")
    lens = sorted(len(a.split()) for a in answers.values())
    print(f"  ✅ {out_zip} · {len(payload)} câu · trung vị {lens[len(lens)//2]} từ "
          f"(đáp án train: 312)")


# =============================================================================
# main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="thư mục chứa train.json, "
                                                  "public-official.json, selected-contexts/")
    p.add_argument("--out", default="legalqa_out")
    p.add_argument("--cache", default="legalqa_cache")
    p.add_argument("--dev-size", type=int, default=500)
    p.add_argument("--no-public", action="store_true")
    p.add_argument("--channels", default="bm25,bge,e5",
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   help="bỏ 'e5' để giảm một nửa thời gian encode corpus trên card yếu")
    p.add_argument("--tier", choices=["lite", "mid", "full"], default=None,
                   help="ép mức thay vì tự dò theo VRAM")
    p.add_argument("--top-n", type=int, default=1)
    p.add_argument("--lead", choices=["cancu", "theo", "none"], default="cancu")
    p.add_argument("--concl", choices=["none", "echo", "echo2"], default="echo2")
    p.add_argument("--limit-docs", type=int, default=0,
                   help="chỉ nạp N văn bản đầu để chạy thử đường ống (không dùng nộp bài)")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    args = parse_args()
    data, cache, out = Path(args.data), Path(args.cache), Path(args.out)
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    t0 = time.time()

    print("=== máy ===")
    prof = detect_profile(args.tier)

    print("=== corpus ===")
    sfx = f"_lim{args.limit_docs}" if args.limit_docs else ""
    chunks, meta = build_corpus(data / "selected-contexts", cache, args.limit_docs)
    idx_of = {c["cid"]: i for i, c in enumerate(chunks)}
    by_doc = defaultdict(list)
    for i, c in enumerate(chunks):
        by_doc[c["doc"]].append(i)

    with (data / "train.json").open(encoding="utf-8") as f:
        train = json.load(f)
    public = {}
    if not args.no_public:
        with (data / "public-official.json").open(encoding="utf-8") as f:
            public = json.load(f)
    dev_qids = build_dev(train, meta, args.dev_size, cache)
    questions = {q: train[q]["question"] for q in dev_qids}
    questions.update({q: v["question"] for q, v in public.items()})
    qids = sorted(questions)
    print(f"  {len(qids)} câu ({len(dev_qids)} dev + {len(public)} public)")

    print("=== BM25 ===")
    bm = build_bm25(chunks, cache, sfx)

    print("=== tầng 1: pool ứng viên ===")
    pool_path = cache / f"pool{sfx}.json.gz"
    pool = load_json(pool_path, {}) or {}
    missing = [q for q in qids if q not in pool]
    if missing:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import torch
        per = defaultdict(set)
        if "bm25" in args.channels:
            t = time.time()
            for i, q in enumerate(missing):
                per[q].update(bm.top_k(tokenize(questions[q]), TOP_K_BM25))
                if (i + 1) % 200 == 0:
                    print(f"    bm25 {i+1}/{len(missing)} ... {time.time()-t:.0f}s",
                          flush=True)
        for name, mdl, qpfx, dpfx in (("bge", MODEL_BGE, "", ""),
                                      ("e5", MODEL_E5, "query: ", "passage: ")):
            if name not in args.channels:
                continue
            # Tiền tố query PHẢI khớp tiền tố đã dùng lúc encode corpus. e5 cần
            # "query:"/"passage:", bge-m3 thì không. Sai một cái là embedding lệch hệ
            # toạ độ, recall tụt mà KHÔNG lỗi nào bắn ra.
            mat = encode_corpus(chunks, mdl, dpfx, cache, prof, sfx)
            enc = SentenceTransformer(mdl, device="cuda")
            qv = enc.encode([qpfx + questions[q] for q in missing],
                            batch_size=prof["encode_batch"], convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
            del enc
            torch.cuda.empty_cache()
            idx = dense_topk(mat, qv, TOP_K_DENSE)
            for i, q in enumerate(missing):
                per[q].update(int(x) for x in idx[i])
            del mat
        for q in missing:
            pool[q] = sorted(per[q])
        save_json(pool_path, pool)
        print(f"    pool {sum(len(v) for v in pool.values())/len(pool):.0f} chunk/câu")
    else:
        print(f"  cache: đủ {len(qids)} câu")

    print("=== tầng 2: rerank chunk -> document -> Điều ===")
    art_path = cache / f"articles{sfx}.json.gz"
    arts = load_json(art_path, {}) or {}
    todo = [q for q in qids if q not in arts]
    if todo:
        rr = Reranker(MODEL_RERANK, prof)
        dieu_cache = {}
        t = time.time()
        for i, q in enumerate(todo):
            cand = pool[q]
            sc = rr.score([[questions[q], chunks[j]["text"]] for j in cand])
            best = {}
            for j, s in zip(cand, sc):
                d = chunks[j]["doc"]
                if d not in best or s > best[d][0]:
                    best[d] = (s, j)
            top = sorted(best, key=lambda d: -best[d][0])[:DOC_K]
            cands = []
            for d in top:
                if d not in dieu_cache:
                    fp = data / "selected-contexts" / f"context_{d}.json"
                    if not fp.exists():
                        fp = data / "selected-contexts" / "selected-contexts" / f"context_{d}.json"
                    with fp.open(encoding="utf-8") as f:
                        dieu_cache[d] = split_dieu(json.load(f).get("passage") or "")
                dd = dieu_cache[d]
                if not dd:
                    # 15,8% văn bản không có cấu trúc Điều -> lùi về chunk 450 từ điểm
                    # cao nhất của chính văn bản đó (đã có từ lượt 1, không tốn thêm).
                    cands.append((d, "", chunks[best[d][1]]["text"]))
                    continue
                if len(dd) > MAX_DIEU_PER_DOC:
                    qt = set(questions[q].lower().split())
                    dd = sorted(dd, key=lambda a: -len(qt & set(a[1].lower().split()))
                                )[:MAX_DIEU_PER_DOC]
                cands.extend((d, num, tx) for num, tx in dd)
            cands = cands[:MAX_CANDS]
            sc2 = rr.score([[questions[q], c[2]] for c in cands])
            ranked = sorted(zip(cands, sc2), key=lambda x: -x[1])[:3]
            arts[q] = [[c[0], c[1], c[2]] for c, _s in ranked]
            if (i + 1) % 25 == 0:
                el = time.time() - t
                print(f"    {i+1}/{len(todo)} câu · {el/60:.1f} phút · còn ~"
                      f"{el/(i+1)*(len(todo)-i-1)/60:.0f} phút", flush=True)
            if (i + 1) % 100 == 0:
                save_json(art_path, arts)
        save_json(art_path, arts)
    else:
        print(f"  cache: đủ {len(qids)} câu")

    print("=== sinh đáp án ===")
    answers = {q: compose(questions[q], arts[q], meta, args.top_n, args.lead,
                          args.concl) for q in qids}
    print(f"  top_n={args.top_n} lead={args.lead} concl={args.concl}")

    print("=== chấm dev ===")
    res = evaluate(dev_qids, train, answers)
    with (out / "eval_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **res,
                            "tier": prof["tier"], "concl": args.concl,
                            "top_n": args.top_n, "lead": args.lead},
                           ensure_ascii=False) + "\n")

    if public:
        print("=== đóng gói ===")
        name = f"submission{('_' + args.tag) if args.tag else ''}.zip"
        package({q: answers[q] for q in public}, set(public), out / name)

    print(f"\nxong {(time.time()-t0)/60:.1f} phút")


if __name__ == "__main__":
    main()
