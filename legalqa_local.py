#!/usr/bin/env python
"""LegalQA (UIT DSC2026 Task 2) — một file, chạy từ câu hỏi tới submission.zip.

    /home/vannk/.venvs/uit_eval311/bin/python legalqa/run_qa.py

Chạy lần đầu làm hết mọi thứ. Chạy lại thì bỏ qua các stage đã có cache, nên vòng lặp
thí nghiệm về CÂU CHỮ (stage compose/eval) chỉ tốn vài chục giây CPU, không đụng GPU.

    legalqa/run_qa.py --shards 4                            # chia 4 GPU, nhanh gần 4 lần
    legalqa/run_qa.py --stage compose eval --lead theo      # thử câu dẫn khác, CPU thuần
    legalqa/run_qa.py --force pool                          # ép chạy lại tầng retrieval
    legalqa/run_qa.py --gpu 5                               # bỏ qua bộ chọn GPU

===============================================================================
VRAM: xin ÍT để chạy được ngay, xin NHIỀU THẺ để chạy nhanh
===============================================================================
Đo trên 2080 Ti, rerank 400 cặp @512 (xem RERANK_PEAK_MIB):

    batch     4     8    16    32    64
    MiB    1194  1318  1478  1848  2584
    cặp/s    76    72    85    76    88

VRAM tăng 2,2 lần, tốc độ KHÔNG đổi — GPU đã bão hoà compute. Hệ quả thực hành:

  * batch lớn KHÔNG làm nhanh hơn, chỉ chiếm chỗ của người khác. Nên `--need-mib auto`
    và batch tự chọn theo khe THẬT SỰ xin được: sàn là ~2.832 MiB (đỉnh tầng pool
    2.232 + context 350 + biên 250), không phải 6.000 như bản đầu đặt tay.
  * muốn nhanh hơn thì `--shards N`: N tiến trình con, mỗi con MỘT thẻ riêng, mỗi con
    làm một phần câu hỏi rồi ghi file cache riêng, cha gộp lại. Gần tuyến tính theo N.
    Shard nào chưa xin được thẻ thì tự chờ — công việc vẫn chạy trên số thẻ đang rảnh
    và các thẻ khác nhập cuộc dần khi trống ra.

===============================================================================
KIẾN TRÚC — vì sao hai tầng rerank chứ không phải một
===============================================================================
Task 1 (LegalIR) trả về document, và backbone của nó (retrieval/run_pipeline_fast.py,
public 0,9439) được tối ưu cho đúng việc đó: chunk 450 từ, max-agg chunk->document.
result.md §7 đo được chunk theo Điều làm TỆ đi 1,19 điểm cho bài toán document.

Task 2 lại cần đúng MỘT Điều luật để chép nguyên văn. Đo trên train.json (oracle biết
trước document đúng, METEOR xấp xỉ exact-match):

    trả toàn bộ văn bản                     0,195   <- precision sập, văn bản ~8.700 từ
    1 Điều tốt nhất, nguyên văn             0,605
    2 Điều tốt nhất ghép lại                0,519   <- ghép thêm là MẤT điểm
    1 Điều + câu dẫn template               0,626
    1 Điều + câu dẫn thật của đáp án        0,676   <- trần của việc viết câu dẫn đúng

Độ dài Điều / độ dài đáp án = 1,02 (trung vị). Điều luật vừa khít đáp án.

Nên pipeline này KHÔNG đổi tầng retrieval của Task 1 mà nối thêm một tầng:

    tầng 1  4 kênh -> ~96 chunk 450 từ -> reranker -> max-agg -> top-5 DOCUMENT
    tầng 2  cắt Điều CHỈNH TRONG 5 document đó -> cùng reranker -> top-1 ĐIỀU
    tầng 3  câu dẫn + thân Điều + câu kết

Tầng 2 chỉ chấm ~75 ứng viên/câu nên rẻ, và vì nó chạy SAU khi document đã chốt, ta
không phải trả giá 1,19 điểm mà chunk-theo-Điều gây ra ở tầng retrieval.

===============================================================================
ARTIFACT DÙNG LẠI — file này chỉ chứa MÃ, không chứa 8,5 GB dữ liệu
===============================================================================
    retrieval/db_fast/            1,1 GB   dense/sparse/texts, tập chunk chuẩn
    retrieval/db_segtitle/        152 MB   BM25 trên corpus ĐÃ TÁCH TỪ
    eval/cache/bgem3ft_dense.*    361 MB   ma trận corpus do bgem3_ft/epoch2 encode
    eval/cache/e5ft_dense.*       361 MB   ma trận corpus do e5_ft/epoch2 encode
    eval/ckpt/bgem3_ft/epoch2     2,2 GB   bge-m3 fine-tune (chỉ có head dense)
    eval/ckpt/e5_ft/epoch2        2,2 GB   multilingual-e5-large fine-tune
    eval/ckpt/vn_reranker_ft_fam/epoch1    Vietnamese_Reranker fine-tune negative cùng họ

Ba checkpoint trên đã nằm sẵn trên đĩa — file này KHÔNG train và KHÔNG tải gì thêm.

Kênh sparse là ngoại lệ duy nhất: nó cần `BAAI/bge-m3` GỐC (~2,2 GB tải từ HuggingFace)
vì bản fine-tune là AutoModel thuần, không có head sparse. Nên nó **tắt mặc định**, dựa
trên số đo lại từ cache reranker của LegalIR (dev v2 1.794 câu, CPU thuần, không GPU —
`recall@5` micro-average, KHÔNG so được với 0,9428 tái cân trong result.md; chỉ đọc cột
chênh lệch):

    4 kênh đủ        0,8972   pool 96 chunk/câu
    bỏ bgesparse     0,8962   pool 83     -0,10 điểm · 285 câu bất đồng · 4 thắng/2 thua
    bỏ e5            0,8833   pool 84     -1,39 điểm · 655 câu bất đồng · 36 thắng/9 thua

Bỏ sparse mất 0,10 điểm, tức nằm gọn trong nhiễu (sai số chuẩn public = 0,75 điểm) và
chỉ 6/285 câu bất đồng đổi kết quả. Bỏ e5 thì khác hẳn — đó là thay đổi thật. Muốn bật
lại kênh sparse: `--channels bm25,bge,e5,sparse`.

===============================================================================
BA CÁI BẪY ĐÃ CÓ NGƯỜI TRẢ GIÁ — đừng gỡ mấy dòng phòng thủ bên dưới
===============================================================================
1. Query phải được encode Y HỆT lúc encode corpus: bge dense = CLS + normalize + KHÔNG
   tiền tố; e5 = SentenceTransformer mặc định + tiền tố "query: ". Sai một trong ba thứ
   đó thì embedding lệch hệ toạ độ, recall tụt mà KHÔNG có lỗi nào bắn ra.
2. Bốn kênh phải nói về CÙNG một tập chunk. Lệch thì candidate tra trượt rồi bị bỏ im
   lặng, sau khi đã tốn tiền chạy qua reranker.
3. Tập khoá submission phải trùng khít ground truth. Thiếu hoặc thừa -> scorer raise ->
   0 điểm TOÀN BÀI, không phải 0 điểm một câu (xem scoring/SCORING_LegalQA.md §2).
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = REPO / "data" / "LegalQA_Public_Test"
CONTEXTS = DATA / "selected-contexts"
HERE = REPO / "legalqa"
CACHE = HERE / "cache"
OUT = HERE / "output"

DB_FAST = REPO / "retrieval" / "db_fast"
DB_BM25 = REPO / "retrieval" / "db_segtitle"
BGE_FT = REPO / "eval" / "ckpt" / "bgem3_ft" / "epoch2"
E5_FT = REPO / "eval" / "ckpt" / "e5_ft" / "epoch2"
RERANKER = REPO / "eval" / "ckpt" / "vn_reranker_ft_fam" / "epoch1"
BGE_MAT = REPO / "eval" / "cache" / "bgem3ft_dense"
E5_MAT = REPO / "eval" / "cache" / "e5ft_dense"

STAGES = ["pool", "article", "compose", "eval", "submit"]


# =============================================================================
# 0. Môi trường — ghim HF cache, đây là chỗ hay đổ model vào ổ hệ thống
# =============================================================================
def pin_hf_home(explicit: str | None) -> Path:
    """Quyết định DỨT KHOÁT model tải về nằm ở đâu, rồi in ra.

    Vì sao phải làm: .env của repo đang có `HF_HOME=D:\\hf_cache` — đường dẫn Windows
    còn sót từ máy cá nhân. Trên Linux nó là đường dẫn TƯƠNG ĐỐI, nên HuggingFace sẽ
    tạo thư mục `D:\\hf_cache` ngay tại thư mục đang đứng, hoặc bỏ qua và đổ về
    ~/.cache/huggingface trên ổ hệ thống. Cả hai đều là thứ không ai muốn trên máy dùng
    chung. Ưu tiên: --hf-home > $HF_HOME (nếu hợp lệ) > <repo>/.hf_cache.
    """
    default = REPO / ".hf_cache"
    want = explicit or os.environ.get("HF_HOME") or str(default)
    p = Path(want)
    windows_style = bool(re.match(r"^[A-Za-z]:[\\/]", want))
    if windows_style or not p.is_absolute():
        print(f"  ⚠️  HF_HOME={want!r} không phải đường dẫn tuyệt đối kiểu POSIX "
              f"(đường dẫn Windows sót lại?) — dùng {default} thay thế.")
        p = default
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        print(f"  ⚠️  Không ghi được vào {p} ({e}) — dùng {default} thay thế.")
        p = default
        p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(p)
    os.environ["HF_HUB_CACHE"] = str(p / "hub")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return p


# =============================================================================
# 1. Chọn GPU trên máy dùng chung — cùng khoá với eval/gpu_lib.sh
# =============================================================================
GPU_LOCK = os.environ.get("LOCK", "/tmp/uit_dsc_gpu_pick2.lock")


def _gpu_free_mib(allow: list[int]) -> list[tuple[int, int]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        idx, used, total = [int(x) for x in line.split(",")]
        if idx in allow:
            rows.append((total - used, idx))
    return sorted(rows, reverse=True)


# Đỉnh VRAM THẬT của tầng rerank, đo trên 2080 Ti với vn_reranker_ft_fam/epoch1 @512
# (trọng số fp16 chiếm 1.090 MiB, phần còn lại là activation):
#
#     batch   4    8   16   32   64
#     MiB  1194 1318 1478 1848 2584
#     cặp/s  76   72   85   76   88
#
# Tốc độ KHÔNG tăng theo batch — GPU đã bão hoà compute. Nên batch lớn chỉ tổ chiếm chỗ
# của người khác mà chẳng nhanh hơn, còn batch nhỏ giúp job CHUI VỪA khe trống nhỏ và
# khởi động ngay thay vì xếp hàng. Muốn nhanh hơn thì chia việc ra NHIỀU GPU (--shards),
# không phải xin thêm VRAM trên một thẻ.
RERANK_PEAK_MIB = {4: 1194, 8: 1318, 16: 1478, 32: 1848, 64: 2584}

# Tầng pool KHÔNG phụ thuộc batch rerank: đỉnh của nó là trọng số encoder (1,1 GB) cộng
# ma trận dense 184.368×1024 fp16 (377 MiB) cộng vùng làm việc topk. Đo thật: 2.232 MiB.
# Sàn của cả job phải là max(hai tầng), nếu không thì batch nhỏ vẫn OOM ở tầng pool —
# bản đầu tôi tính sàn 1.794 MiB và đó là sai, tầng pool sẽ chết trước khi tới rerank.
POOL_PEAK_MIB = 2232
CUDA_CTX_MIB = 350      # context + phân mảnh, đo bằng nvidia-smi trừ đi torch reserved
HEADROOM_MIB = 250


def plan_batch(free_mib: int) -> tuple[int, int]:
    """Chọn batch rerank LỚN NHẤT vừa khe trống -> (batch, MiB thật sự cần).

    Vì tốc độ phẳng theo batch, đây không phải tối ưu tốc độ mà là tối ưu **khả năng
    khởi động**: ~2.800 MiB là chạy được, không cần đợi 6.000.
    """
    for b in sorted(RERANK_PEAK_MIB, reverse=True):
        need = max(RERANK_PEAK_MIB[b], POOL_PEAK_MIB) + CUDA_CTX_MIB + HEADROOM_MIB
        if need <= free_mib:
            return b, need
    b = min(RERANK_PEAK_MIB)
    return b, max(RERANK_PEAK_MIB[b], POOL_PEAK_MIB) + CUDA_CTX_MIB + HEADROOM_MIB


def min_need_mib() -> int:
    return plan_batch(0)[1]


def acquire_gpu(need_mib: int, allow: list[int], poll: int = 5, timeout_min: int = 180):
    """Chờ tới khi có thẻ đủ trống, chốt dưới khoá, trả về (chỉ số, fd khoá).

    Giữ nguyên nguyên tắc của eval/gpu_lib.sh: **đợi thì KHÔNG giữ khoá**, chỉ giữ đúng
    lúc kiểm-lại-và-chiếm. Bản bash phải đoán "khi nào tiến trình con đã thực sự cấp
    phát" bằng cách rình bộ nhớ trống tụt đi; ở đây ta LÀ tiến trình đó, nên chỉ cần
    giữ khoá qua đúng lời gọi reserve_vram() đầu tiên rồi thả — không có fd kế thừa,
    không có bộ đếm 150s, không có hai job cùng chọn một thẻ.

    Đánh đổi có ý thức: nếu model chưa nằm trong cache HF, ta thả khoá TRƯỚC khi tải
    (tải mất vài phút, ôm khoá suốt là đúng cái bệnh gpu_lib.sh viết hẳn một đoạn để
    tránh). Bù lại bằng ballast — phần VRAM khoá cứng, không bao giờ nhả.
    """
    t0 = time.time()
    while True:
        rows = _gpu_free_mib(allow)
        if rows and rows[0][0] >= need_mib:
            fd = os.open(GPU_LOCK, os.O_CREAT | os.O_WRONLY, 0o666)
            fcntl.flock(fd, fcntl.LOCK_EX)
            rows = _gpu_free_mib(allow)
            if rows and rows[0][0] >= need_mib:
                gpu, free = rows[0][1], rows[0][0]
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                print(f"  chiếm GPU {gpu} ({free} MiB trống, cần {need_mib})")
                return gpu, fd, free
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        if time.time() - t0 > timeout_min * 60:
            raise SystemExit(f"Chờ quá {timeout_min} phút mà không có GPU nào trong "
                             f"{allow} còn {need_mib} MiB. Hạ --need-mib hoặc đợi.")
        if int(time.time() - t0) % 60 < poll:
            free = rows[0][0] if rows else 0
            print(f"    ... chờ GPU (trống nhất {free} MiB / cần {need_mib}), "
                  f"{int(time.time()-t0)}s", flush=True)
        time.sleep(poll)


def _quiet_tqdm() -> None:
    """Tắt thanh tiến trình của FlagEmbedding khi KHÔNG chạy trong terminal.

    tqdm vẽ bằng ký tự `\\r`, ở terminal thì đè lên một dòng, còn ghi vào file log thì
    thành hàng nghìn dòng rác nuốt mất mọi thứ đáng đọc (đã thấy: một lượt rerank sinh
    260 dòng). tqdm 4.70 không đọc biến môi trường TQDM_DISABLE nên phải vá thẳng vào
    module đang dùng nó. Chạy tương tác thì vẫn giữ thanh tiến trình — job GPU dài
    20 phút mà im lặng hoàn toàn thì không phân biệt được với treo.
    """
    if sys.stderr.isatty():
        return
    try:
        import FlagEmbedding.flag_reranker as fr
    except ImportError:
        return

    # KHÔNG dùng functools.partial(tqdm, disable=True): flag_reranker.py:263 gọi
    # tqdm(..., disable=len(pairs)<128) — tham số ở CHỖ GỌI thắng tham số của partial,
    # nên bản vá kiểu đó vô hiệu. Đã đo: log của một lần chạy 61 phút phình lên 742 KB
    # toàn thanh tiến trình. Phải bọc lại và ép ghi đè.
    def _wrap(orig):
        def f(*a, **kw):
            kw["disable"] = True
            return orig(*a, **kw)
        return f

    fr.tqdm = _wrap(fr.tqdm)
    fr.trange = _wrap(fr.trange)


_INSTANCE_FD = None


def claim_single_instance() -> None:
    """Chặn hai bản cùng chạy. Giữ tham chiếu fd ở cấp module cho tới khi tiến trình chết.

    Đã xảy ra thật: gõ nhầm `nohup ... &` hai lần, hai job cùng `--dev-size 500` cùng
    chiếm hai GPU của máy dùng chung, cùng ghi đè legalqa/cache/pool.json.gz, và cùng
    ghi vào một run.log mà lệnh `>` thứ hai vừa cắt cụt. Không có gì trong code cản lại.

    flock tự nhả khi tiến trình chết, kể cả bị kill -9 — nên không có chuyện khoá kẹt
    vĩnh viễn như file khoá GPU v1 (xem eval/gpu_lib.sh).
    """
    global _INSTANCE_FD
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (f".instance{os.environ.get('RUN_QA_TAG','')}.lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            other = os.read(fd, 64).decode().strip()
        except OSError:
            other = "?"
        os.close(fd)
        raise SystemExit(
            f"Đã có một bản run_qa.py đang chạy (PID {other}). Hai bản cùng chạy sẽ "
            f"chiếm hai GPU và ghi đè cache của nhau.\n"
            f"Xem tiến độ:  tail -f {OUT/'run.log'}\n"
            f"Dừng bản kia: kill {other}")
    os.truncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    _INSTANCE_FD = fd


def assert_single_gpu() -> None:
    """Chốt chặn cuối: torch phải CHỈ nhìn thấy đúng một thẻ.

    FlagReranker tự bọc `torch.nn.DataParallel` khi `torch.cuda.device_count() > 1`
    (FlagEmbedding/flag_reranker.py:245). Trên máy dùng chung đó là thảm hoạ: job xin
    một thẻ nhưng nhảy lên cả tám, OOM cả tiến trình của người khác. Mà device_count()
    có lru_cache nên chỉ cần một lời gọi sớm là CUDA_VISIBLE_DEVICES đặt sau đó không
    còn tác dụng — im lặng, không lỗi. Nên phải kiểm tường minh ngay sau khi chiếm thẻ.
    """
    import torch
    n = torch.cuda.device_count()
    if n != 1:
        raise SystemExit(
            f"torch nhìn thấy {n} GPU nhưng job này chỉ được phép dùng 1 "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}). "
            f"Nghĩa là torch đã bị nạp TRƯỚC khi chọn thẻ — dừng ở đây, vì chạy tiếp "
            f"là FlagReranker sẽ bọc DataParallel lên toàn bộ thẻ của máy.")


def release_gpu_lock(fd):
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


# =============================================================================
# 2. Corpus — chỉ mục văn bản, cắt Điều, metadata
# =============================================================================
SO_HEADER_RE = re.compile(r"Số\s*[:：]\s*([0-9A-Za-zĐđ/\-]+)")
SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zĐđ]{2,10}(?:-[A-Za-zĐđ]{2,10})?")
LOAI_VB_CANON = ["Thông tư liên tịch", "Nghị định", "Luật", "Thông tư", "Quyết định",
                 "Pháp lệnh", "Nghị quyết", "Bộ luật", "Chỉ thị"]
LOAI_PATTERN = re.compile("(" + "|".join(re.escape(x) for x in LOAI_VB_CANON) + ")",
                          re.IGNORECASE)

# Neo `^` (re.M) là điều kiện sống còn: "Điều 5" xuất hiện dày đặc GIỮA dòng dưới dạng
# trích dẫn chéo ("quy định tại Điều 5 Nghị định này"). Cắt ở đó là băm nát điều luật.
# Đòi thêm dấu `.` hoặc `:` ngay sau số để loại "Điều 5 của Nghị định..." đứng đầu dòng.
DIEU_RE = re.compile(r"(?m)^\s*Điều\s+(\d+[a-zA-ZđĐ]?)\s*[.．:]")
DIEU_PREFIX_RE = re.compile(r"^\s*Điều\s+\d+[a-zA-ZđĐ]?\s*[.．:]?\s*")
DECO_RE = re.compile(r"^[\s\-_=–—.·*]+$")


def deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").replace("Đ", "D").replace("đ", "d")


def extract_vb_info(passage: str) -> tuple[str, str]:
    """-> (loại văn bản, số hiệu). Cùng thuật toán rag_model/legalqa/doc_meta.py."""
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
    """-> [(số Điều, nguyên văn Điều)]. Rỗng nếu văn bản không có cấu trúc Điều.

    15,8% văn bản trong corpus không có (QCVN/TCVN, quyết định ngắn, biểu mẫu). Trả về
    rỗng để nơi gọi tự lo fallback, KHÔNG nới lỏng regex — nới ra thì 84,2% còn lại bị
    cắt nhầm ở các trích dẫn chéo, mất nhiều hơn được.
    """
    ms = list(DIEU_RE.finditer(passage))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(passage)
        out.append((m.group(1), passage[m.start():end].strip()))
    return out


def read_doc(doc_id: str) -> dict:
    with (CONTEXTS / f"context_{doc_id}.json").open(encoding="utf-8") as f:
        return json.load(f)


def load_doc_index() -> dict:
    """{doc_id: [link, loại văn bản, số hiệu]} cho toàn bộ 8.532 văn bản, cache lại.

    Một lượt đọc 507 MB mất ~1 phút; sau đó mọi stage đọc từ cache. Cần link để phân
    giải citation -> document (dùng cho dev split), cần loại/số hiệu để dựng câu dẫn.
    """
    path = CACHE / "doc_index.json"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    print("  dựng doc_index (đọc toàn corpus, chỉ lần đầu) ...")
    idx = {}
    t0 = time.time()
    for i, fp in enumerate(sorted(CONTEXTS.glob("context_*.json"))):
        try:
            with fp.open(encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        psg = d.get("passage") or ""
        loai, so = extract_vb_info(psg)
        idx[str(d["id"])] = [d.get("link", ""), loai, so]
        if (i + 1) % 2000 == 0:
            print(f"    {i+1} văn bản ... {time.time()-t0:.0f}s", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False)
    print(f"    {len(idx)} văn bản, {time.time()-t0:.0f}s -> {path}")
    return idx


# =============================================================================
# 3. Dev split — chia theo VĂN BẢN, không phải theo câu
# =============================================================================
CIT_RE = re.compile(r"\b(\d{1,4})/(\d{4})/([A-ZĐ][A-ZĐ\-]*)\b")


def resolve_gold_docs(answer: str, slug_map: list[tuple[str, str]], cache: dict) -> set:
    """Suy document được trích từ số hiệu trong đáp án, đối chiếu slug trong link.

    KHÔNG parse ngược slug bằng regex (đã thử trong eval/check_legalqa_usable.py: tham
    lam, '90-2017-ND-CP-xu-phat-vi-pham' ra '90/2017/ND-CP-XU-PHAT-VI'). Sinh slug TỪ
    citation rồi tìm chuỗi con — một chiều, không mơ hồ. Phủ 47,8% câu train.
    """
    docs = set()
    for m in CIT_RE.finditer(answer):
        key = f"{int(m.group(1))}/{m.group(2)}/{deaccent(m.group(3)).upper()}"
        if key not in cache:
            pat = "-" + key.replace("/", "-").lower() + "-"
            cache[key] = {did for did, link in slug_map if pat in link}
        docs |= cache[key]
    return docs


def build_dev_split(train: dict, doc_index: dict, n: int) -> list[str]:
    """Lấy n câu làm dev, chia theo VĂN BẢN để không rò rỉ nếu sau này fine-tune.

    Câu nào phân giải được văn bản gốc thì đi theo nhóm của văn bản đó; câu không phân
    giải được thì tự thành một nhóm. Duyệt nhóm theo thứ tự băm (tất định, không phụ
    thuộc seed random của Python) và nhận TRỌN nhóm cho tới khi đủ n — cắt giữa nhóm là
    tự tay tạo rò rỉ.
    """
    path = CACHE / f"dev_qids_{n}.json"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    slug_map = [(did, deaccent(v[0] or "").lower()) for did, v in doc_index.items()]
    cache, group_of = {}, {}
    for qid, item in train.items():
        docs = resolve_gold_docs(item.get("answer") or "", slug_map, cache)
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
    print(f"  dev split: {len(picked)} câu / {len(ordered)} nhóm văn bản · "
          f"{n_resolved} câu ({n_resolved/max(len(picked),1):.1%}) phân giải được citation")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False)
    return picked


# =============================================================================
# 4. Preflight — hỏng thì hỏng ồn ào, ở giây thứ 0 chứ không phải phút thứ 20
# =============================================================================
SCRIPT = Path(__file__).resolve()
VENV = Path("/home/vannk/.venvs/uit_eval311/bin/python")


def check_interpreter() -> None:
    """Thiếu thư viện thì TỰ chuyển sang venv đúng, thay vì bắt người dùng gõ lại.

    Trên máy này `python` không tồn tại và `python3` là 3.8 không có FlagEmbedding —
    nên `python3 run_qa.py` chắc chắn hỏng, mà traceback `ModuleNotFoundError` thì
    không nói cho ai biết phải làm gì. Bản trước in ra dòng lệnh đúng; vẫn phiền, vì
    người dùng đang đứng trong legalqa/ còn dòng lệnh viết đường dẫn từ gốc repo.

    Nên ở đây dùng os.execv: thay luôn tiến trình bằng venv đúng, giữ nguyên tham số.
    Biến RUN_QA_REEXEC chặn lặp vô hạn nếu chính venv cũng thiếu thư viện. Venv là
    uit_eval311 (torch 2.6, transformers 4.44.2 — bản bị ghim vì FlagEmbedding 1.2.11
    cần EncoderDecoderCache, xem retrieval/requirements.txt).
    """
    # find_spec chứ KHÔNG __import__: nạp torch ở đây là hỏng cả job.
    # torch.cuda.device_count() có lru_cache, và sentence_transformers/FlagEmbedding
    # gọi nó ngay lúc import. Nếu điều đó xảy ra TRƯỚC khi acquire_gpu() đặt
    # CUDA_VISIBLE_DEVICES thì con số 8 bị đóng băng, rồi FlagReranker thấy
    # device_count()>1 và tự bọc DataParallel lên CẢ TÁM thẻ của máy dùng chung.
    # Đã xảy ra thật: job xin đúng 1 thẻ nhưng OOM trên GPU 2 của người khác.
    import importlib.util
    missing = []
    for mod in ("torch", "numpy", "scipy", "transformers", "sentence_transformers",
                "FlagEmbedding", "underthesea", "nltk"):
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError):
            missing.append(mod)
    if not missing:
        print(f"  python {sys.version_info.major}.{sys.version_info.minor} · "
              f"{sys.executable}")
        return
    if os.environ.get("RUN_QA_REEXEC") != "1" and VENV.exists():
        print(f"  {sys.executable} thiếu {', '.join(missing)} — chuyển sang {VENV}")
        os.environ["RUN_QA_REEXEC"] = "1"
        os.execv(str(VENV), [str(VENV), str(SCRIPT), *sys.argv[1:]])
    raise SystemExit(
        f"Trình thông dịch {sys.executable} thiếu: {', '.join(missing)}.\n"
        f"Venv mong đợi ({VENV}) không dùng được. Chạy bằng một trình thông dịch có "
        f"torch + FlagEmbedding + underthesea:\n    <python-cua-ban> {SCRIPT}")


def preflight(args) -> None:
    missing = []
    need = [
        (DATA / "train.json", "câu hỏi + đáp án huấn luyện"),
        (DATA / "public-official.json", "câu hỏi public test"),
        (CONTEXTS, "kho văn bản selected-contexts/"),
        (DB_BM25 / "bm25_W.npz", "BM25 corpus đã tách từ"),
        (DB_FAST / "bge_sparse.npz", "ma trận sparse của corpus"),
        (DB_FAST / "texts.json.gz", "text của chunk"),
        (Path(str(BGE_MAT) + ".npy"), "ma trận dense bge-m3-ft"),
        (Path(str(E5_MAT) + ".npy"), "ma trận dense e5-ft"),
        (BGE_FT / "model.safetensors", "checkpoint bge-m3 fine-tune"),
        (E5_FT / "model.safetensors", "checkpoint e5 fine-tune"),
        (RERANKER / "model.safetensors", "checkpoint reranker fine-tune"),
    ]
    for p, what in need:
        if not p.exists():
            missing.append(f"{p}  ({what})")
    if missing:
        raise SystemExit("Thiếu artifact:\n  - " + "\n  - ".join(missing))

    n_ctx = sum(1 for _ in CONTEXTS.glob("context_*.json"))
    if n_ctx < 8000:
        raise SystemExit(f"Chỉ thấy {n_ctx} context_*.json trong {CONTEXTS} — "
                         f"kho văn bản chưa giải nén đủ (mong đợi 8.532).")
    print(f"  artifact đủ · {n_ctx} văn bản · HF_HOME={os.environ['HF_HOME']}")

    if "sparse" in args.channels:
        hub = Path(os.environ["HF_HUB_CACHE"]) / "models--BAAI--bge-m3"
        if not hub.exists():
            print("  ⓘ  BAAI/bge-m3 chưa có trong cache — kênh sparse sẽ tải ~2,2 GB "
                  "về HF_HOME ở lần chạy này. Không có mạng thì chạy lại với "
                  "`--channels bm25,bge,e5`.")


# =============================================================================
# 5. STAGE pool — 4 kênh, hợp thành pool chunk cho mỗi câu hỏi
# =============================================================================
def stage_pool(qids: list[str], questions: dict, args, prev: dict) -> dict:
    """-> {qid: [chunk_id]}. Ghi cache/pool.json.gz (gộp với `prev` đã có sẵn).

    Đúng thuật toán retrieval/run_pipeline_fast.py:run_step1(), rút gọn còn phần Task 2
    cần: không ghi ra 4 file kênh riêng (ta không làm ablation từng kênh ở đây), chỉ
    giữ hợp của chúng.
    """
    import numpy as np
    import scipy.sparse as sp
    import torch
    from retrieval.fast_index import BM25Csr, DenseMatrix
    from eval.gpu_reserve import reserve_vram

    texts_q = [questions[q] for q in qids]
    print(f"  {len(qids)} câu hỏi · kênh: {','.join(args.channels)}")

    # ---- vector truy vấn: nạp từng model, dùng xong xoá ngay -----------------
    sp_idx = sp_val = None
    if "sparse" in args.channels:
        from retrieval.encoder import BGEM3Encoder
        t = time.time()
        enc = BGEM3Encoder(model_name="BAAI/bge-m3", use_fp16=True)
        if args.reserve_gb > 0:
            reserve_vram(args.reserve_gb, 0, ballast_gb=args.ballast_gb)
        sp_idx, sp_val = [], []
        for i in range(0, len(texts_q), args.batch_size):
            _d, ix, vl = enc.encode(texts_q[i:i + args.batch_size], type="query")
            sp_idx.extend(ix)
            sp_val.extend(vl)
        del enc
        torch.cuda.empty_cache()
        print(f"    sparse query {time.time()-t:.0f}s")

    bge_q = None
    if "bge" in args.channels:
        # CLS + normalize + KHÔNG tiền tố, q_len 64 — phải khớp eval/encode_corpus.py
        # đã encode document thế nào. Đây là bẫy số 1 ở đầu file.
        from transformers import AutoModel, AutoTokenizer
        t = time.time()
        tok = AutoTokenizer.from_pretrained(str(BGE_FT))
        mdl = AutoModel.from_pretrained(str(BGE_FT), add_pooling_layer=False)
        mdl = mdl.half().to("cuda" if torch.cuda.is_available() else "cpu").eval()
        if args.reserve_gb > 0:
            reserve_vram(args.reserve_gb, 0, ballast_gb=args.ballast_gb)
        bge_q = []
        with torch.inference_mode():
            for i in range(0, len(texts_q), args.batch_size):
                e = tok(texts_q[i:i + args.batch_size], padding=True, truncation=True,
                        max_length=64, return_tensors="pt").to(mdl.device)
                h = mdl(**e).last_hidden_state[:, 0]
                h = torch.nn.functional.normalize(h.float(), dim=-1)
                bge_q.extend(h.cpu().numpy().astype(np.float16))
        del mdl, tok
        torch.cuda.empty_cache()
        print(f"    bge dense query {time.time()-t:.0f}s")

    e5_q = None
    if "e5" in args.channels:
        from retrieval.encoder import E5Encoder
        t = time.time()
        enc = E5Encoder(model_name=str(E5_FT), use_prefix=True, segment=False)
        if args.reserve_gb > 0:
            reserve_vram(args.reserve_gb, 0, ballast_gb=args.ballast_gb)
        e5_q = []
        for i in range(0, len(texts_q), args.batch_size):
            d, _, _ = enc.encode(texts_q[i:i + args.batch_size], type="query")
            e5_q.extend(d)
        del enc
        torch.cuda.empty_cache()
        print(f"    e5 dense query {time.time()-t:.0f}s")

    if args.reserve_gb > 0:
        # Giữ chỗ cho phần tốn nhất còn lại (ma trận dense + matmul + topk). PHẢI đặt ở
        # ĐÂY, sau mọi from_pretrained: accelerate gọi empty_cache và xoá sạch chỗ đã
        # giữ trước đó. Xem eval/gpu_reserve.py.
        reserve_vram(args.reserve_gb, 0, ballast_gb=args.ballast_gb)

    # ---- chạy từng kênh -----------------------------------------------------
    per_q = defaultdict(set)
    t = time.time()

    dm = DenseMatrix(str(BGE_MAT))
    universe, aid_of = dm.chunk_ids, dict(zip(dm.chunk_ids, dm.aids))
    if "bge" in args.channels:
        _v, idx = dm.search_batch(np.asarray(bge_q), args.top_k_dense)
        for i, q in enumerate(qids):
            per_q[q].update(dm.chunk_ids[x] for x in idx[i])
    del dm
    if args.reserve_gb <= 0:
        torch.cuda.empty_cache()

    if "e5" in args.channels:
        dm = DenseMatrix(str(E5_MAT))
        _check_universe("e5", dm.chunk_ids, universe)
        _v, idx = dm.search_batch(np.asarray(e5_q), args.top_k_dense)
        for i, q in enumerate(qids):
            per_q[q].update(dm.chunk_ids[x] for x in idx[i])
        del dm
        if args.reserve_gb <= 0:
            torch.cuda.empty_cache()

    if "sparse" in args.channels:
        mat = sp.load_npz(DB_FAST / "bge_sparse.npz").T.tocsr()
        with gzip.open(DB_FAST / "bge_sparse.npz.ids.json.gz", "rt", encoding="utf-8") as f:
            meta = json.load(f)
        _check_universe("sparse", meta["chunk_ids"], universe)
        vocab_n = mat.shape[0]
        rows, cols, vals_ = [], [], []
        for j, (ix, vl) in enumerate(zip(sp_idx, sp_val)):
            for a, bv in zip(ix, vl):
                a = int(a)
                if a < vocab_n:
                    rows.append(j); cols.append(a); vals_.append(float(bv))
        qmat = sp.csr_matrix((np.asarray(vals_, dtype=np.float32),
                              (np.asarray(rows), np.asarray(cols))),
                             shape=(len(qids), vocab_n))
        for s in range(0, len(qids), 64):
            block = (qmat[s:s + 64] @ mat).toarray()
            for r in range(block.shape[0]):
                sc = block[r]
                k = min(args.top_k_dense, sc.size)
                top = np.argpartition(-sc, k - 1)[:k]
                per_q[qids[s + r]].update(meta["chunk_ids"][x] for x in top if sc[x] > 0)

    if "bm25" in args.channels:
        from underthesea import word_tokenize
        bm = BM25Csr(str(DB_BM25))
        _check_universe("bm25", bm.chunk_ids, universe)
        for i, q in enumerate(qids):
            # Câu hỏi phải tách từ ĐÚNG NHƯ corpus đã tách. Corpus có "hợp_đồng" mà câu
            # hỏi vẫn là "hợp đồng" thì không còn token nào khớp, BM25 trả về 0.
            seg = word_tokenize(texts_q[i], format="text")
            per_q[q].update(cid for _s, cid in bm.search(seg, top_k=args.top_k_bm25))
            if (i + 1) % 500 == 0:
                print(f"    bm25 {i+1}/{len(qids)} ... {time.time()-t:.0f}s", flush=True)
        del bm

    fresh = {q: sorted(per_q[q]) for q in qids}
    empty = [q for q in qids if not fresh[q]]
    if empty:
        raise SystemExit(f"{len(empty)} câu có pool rỗng: {empty[:5]}")
    avg = sum(len(v) for v in fresh.values()) / len(fresh)
    print(f"    pool {avg:.0f} chunk/câu · {time.time()-t:.0f}s")

    print(f"    đỉnh VRAM tầng pool: "
          f"{torch.cuda.max_memory_reserved()/2**20:.0f} MiB")
    pool = {**prev, **fresh}
    _save_gz(cache_out("pool", args.shard_tag),
             {"fp": pool_fingerprint(args), "pool": pool, "aid_of": aid_of})
    return pool


def _check_universe(name: str, ids, universe) -> None:
    """Bẫy số 2: bốn kênh phải cùng một tập chunk, lệch thì hỏng ồn ào tại đây."""
    extra = set(ids) - set(universe)
    if extra:
        raise SystemExit(
            f"Kênh '{name}' có {len(extra)} chunk ngoài tập của kênh bge "
            f"(vd {sorted(extra)[:3]}). Index lệch — dựng lại trước khi chạy.")


# =============================================================================
# 6. STAGE article — rerank hai lần: chunk -> document, rồi Điều trong document
# =============================================================================
def stage_article(qids: list[str], questions: dict, pool: dict, aid_of: dict,
                  doc_index: dict, args, prev: dict) -> dict:
    """-> {qid: {"docs": [[doc_id, score]], "arts": [[doc_id, dieu, score, text]]}}"""
    import torch
    from FlagEmbedding import FlagReranker
    from eval.gpu_reserve import reserve_vram

    _quiet_tqdm()

    with gzip.open(DB_FAST / "texts.json.gz", "rt", encoding="utf-8") as f:
        texts = json.load(f)
    lack = [c for q in qids for c in pool[q] if c not in texts]
    if lack:
        raise SystemExit(f"thiếu text cho {len(lack)} chunk trong pool")

    model = FlagReranker(str(RERANKER), use_fp16=True)
    if args.reserve_gb > 0:
        # Giữ chỗ NGAY SAU khi reranker lên GPU — đặt trước đó thì from_pretrained
        # xoá sạch. Chống bị chen giữa chừng trên máy dùng chung.
        reserve_vram(args.reserve_gb, 0, ballast_gb=args.ballast_gb)

    # ---- lượt 1: chunk 450 từ -> điểm document (max-agg) --------------------
    t = time.time()
    flat, spans = [], []
    for q in qids:
        start = len(flat)
        flat.extend([[questions[q], texts[c]] for c in pool[q]])
        spans.append((start, len(flat)))
    print(f"  lượt 1: {len(flat)} cặp (chunk 450 từ) ...", flush=True)
    scores = model.compute_score(flat, batch_size=args.batch_size_rerank,
                                 max_length=args.max_length, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    docs_of, best_chunk = {}, {}
    for i, q in enumerate(qids):
        s, e = spans[i]
        best = {}
        bchunk = {}
        for cid, sc in zip(pool[q], scores[s:e]):
            aid = aid_of[cid]
            if aid not in best or sc > best[aid]:
                best[aid] = sc
                bchunk[aid] = cid
        top = sorted(best, key=best.get, reverse=True)[:args.doc_k]
        docs_of[q] = [[d, float(best[d])] for d in top]
        best_chunk[q] = {d: bchunk[d] for d in top}
    print(f"    {time.time()-t:.0f}s · {len(flat)/max(len(qids),1):.0f} cặp/câu")

    # ---- lượt 2: Điều bên trong đúng những document đó ----------------------
    t = time.time()
    dieu_cache: dict[str, list] = {}
    flat2, spans2, meta2 = [], [], []
    for q in qids:
        start = len(flat2)
        cands = []
        for doc_id, _sc in docs_of[q]:
            if doc_id not in dieu_cache:
                dieu_cache[doc_id] = split_dieu(read_doc(doc_id).get("passage") or "")
            arts = dieu_cache[doc_id]
            if not arts:
                # 15,8% văn bản không có cấu trúc Điều -> lùi về chunk 450 từ điểm cao
                # nhất của chính văn bản đó (đã có sẵn từ lượt 1, không tốn gì thêm).
                cands.append((doc_id, "", texts[best_chunk[q][doc_id]]))
                continue
            if len(arts) > args.max_dieu_per_doc:
                # Văn bản p90 dài 18.400 từ. Lọc thô bằng chồng lấp token với câu hỏi
                # trước khi đưa vào cross-encoder — chỉ để chặn chi phí, thứ hạng cuối
                # vẫn do reranker quyết.
                qt = set(questions[q].lower().split())
                arts = sorted(arts, key=lambda a: -len(qt & set(a[1].lower().split()))
                              )[:args.max_dieu_per_doc]
            cands.extend((doc_id, d, tx) for d, tx in arts)
        cands = cands[:args.max_cands]
        flat2.extend([[questions[q], c[2]] for c in cands])
        spans2.append((start, len(flat2)))
        meta2.append(cands)
    print(f"  lượt 2: {len(flat2)} cặp (Điều) ...", flush=True)
    scores2 = model.compute_score(flat2, batch_size=args.batch_size_rerank,
                                  max_length=args.max_length, normalize=True)
    if isinstance(scores2, float):
        scores2 = [scores2]

    out = {}
    for i, q in enumerate(qids):
        s, e = spans2[i]
        ranked = sorted(zip(meta2[i], scores2[s:e]), key=lambda x: -x[1])
        keep = ranked[:args.keep_articles]
        out[q] = {
            "docs": docs_of[q],
            "arts": [[c[0], c[1], float(sc), c[2]] for c, sc in keep],
        }
    print(f"    {time.time()-t:.0f}s · {len(flat2)/max(len(qids),1):.0f} cặp/câu")

    print(f"    đỉnh VRAM tầng rerank: "
          f"{torch.cuda.max_memory_reserved()/2**20:.0f} MiB "
          f"(batch {args.batch_size_rerank})")
    del model
    torch.cuda.empty_cache()
    out = {**prev, **out}
    _save_gz(cache_out("articles", args.shard_tag),
             {"fp": article_fingerprint(args), "arts": out})
    return out


# =============================================================================
# 7. STAGE compose — câu dẫn + thân Điều + câu kết
# =============================================================================
FALLBACK = "Không tìm thấy thông tin pháp lý cho câu hỏi này."


def compose_answer(question: str, arts: list, doc_index: dict, args) -> str:
    """Ghép đáp án từ các Điều đã xếp hạng.

    Ba lựa chọn dưới đây đều đến từ số đo trên train.json chứ không phải khẩu vị:

    - `top_n = 1`: oracle đo được 1 Điều = 0,605 còn 2 Điều ghép lại = 0,519. METEOR có
      alpha 0,9 nên nặng recall, NHƯNG số hạng (1-alpha)/P vẫn bùng lên khi câu trả lời
      dài gấp đôi đáp án. Nhồi thêm Điều là mất điểm, không phải được thêm.
    - `lead = cancu`: 57,4% đáp án thật mở đầu bằng "Căn cứ", 25,1% bằng "Theo"; cụm
      "quy định như sau" chiếm 27,2% còn "quy định cụ thể" chỉ 1,1%.
    - bỏ "Điều X." ở đầu thân bài: 98,8% đáp án thật không lặp lại số Điều ngay sau câu
      dẫn, mà vào thẳng tiêu đề Điều.
    """
    parts, seen = [], set()
    for doc_id, dieu, _score, text in arts:
        if len(parts) >= args.top_n:
            break
        if doc_id in seen:
            continue
        seen.add(doc_id)
        _link, loai, so = doc_index.get(doc_id, ["", "", ""])
        loai = loai or "văn bản"
        body = DIEU_PREFIX_RE.sub("", text, count=1) if (dieu and args.strip_dieu) else text
        if args.drop_deco:
            body = "\n".join(l for l in body.split("\n") if not DECO_RE.match(l))
        head = f"Điều {dieu} " if dieu else ""
        tail = " ".join(x for x in (loai, so) if x)
        if args.lead == "none":
            lead = ""
        else:
            verb = "Căn cứ" if args.lead == "cancu" else "Theo"
            lead = f"{verb} {head}{tail} quy định như sau:"
        parts.append(f"{lead}\n{body}".strip() if lead else body.strip())
    if not parts:
        return FALLBACK
    ans = "\n\n".join(parts)
    if args.concl != "none":
        q = question.strip().rstrip("?").strip()
        if q:
            ql = q[0].lower() + q[1:]
            if args.concl == "echo":
                ans += f"\nNhư vậy, theo quy định nêu trên thì {ql}."
            elif args.concl == "echo2":
                ans += (f"\nTheo đó, {ql}.\n"
                        f"Như vậy, theo quy định nêu trên thì {ql}.")
            elif args.concl == "q":
                ans += "\n" + q + "."
    return ans


def stage_compose(qids: list[str], questions: dict, articles: dict, doc_index: dict,
                  args) -> dict:
    return {q: compose_answer(questions[q], articles[q]["arts"], doc_index, args)
            for q in qids}


# =============================================================================
# 8. STAGE eval — đúng công thức scoring/legalqa/scoring.py
# =============================================================================
def load_scorers():
    """METEOR của nltk (alpha 0,9 · beta 3 · gamma 0,5, tokenize bằng str.split() trần)
    và ROUGE-L từ BẢN VENDOR của BTC trong scoring/legalqa/ — không dùng gói pip, để
    khỏi lệch version so với lúc chấm thật.

    Bản vendor cần `absl-py`. Thiếu thì bỏ ROUGE-L chứ không dừng: METEOR mới là độ đo
    xếp hạng, ROUGE-L chỉ tham khảo và vốn đã hỏng với tiếng Việt
    (scoring/SCORING_LegalQA.md §4).
    """
    import nltk
    from nltk.translate.meteor_score import meteor_score

    # Scorer thật của BTC gọi nltk.download('wordnet') lúc import, nên máy chấm CÓ
    # wordnet. Ta thì chạy offline được: đã đo trực tiếp, METEOR có và không có wordnet
    # cho kết quả TRÙNG tới 6 chữ số thập phân trên văn bản pháp luật tiếng Việt — đúng
    # như scoring/SCORING_LegalQA.md §3 dự đoán (WordNet tiếng Anh vô dụng ở đây, chỉ
    # exact match sau lowercase là đáng kể). Nên thiếu wordnet KHÔNG phải lỗi.
    local = REPO / ".nltk_data"
    if local.is_dir() and str(local) not in nltk.data.path:
        nltk.data.path.insert(0, str(local))
    rouge = None
    vendor = str(REPO / "scoring" / "legalqa")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    try:
        from rouge_score import rouge_scorer
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    except ImportError as e:
        print(f"  ⓘ  bỏ qua ROUGE-L ({e}) — cài `absl-py` nếu cần con số này.")
    return meteor_score, rouge


def stage_eval(dev_qids: list[str], train: dict, answers: dict, tag: str = "") -> dict:
    meteor_fn, rouge = load_scorers()
    ms, rs = [], []
    t = time.time()
    for i, q in enumerate(dev_qids):
        ref, pred = str(train[q]["answer"]), str(answers[q])
        ms.append(meteor_fn([ref.split()], pred.split()))
        if rouge is not None:
            rs.append(rouge.score(ref, pred)["rougeL"].fmeasure)
        if (i + 1) % 250 == 0:
            print(f"    chấm {i+1}/{len(dev_qids)} ... {time.time()-t:.0f}s", flush=True)
    n = len(dev_qids)
    import statistics
    se = statistics.stdev(ms) / (n ** 0.5) if n > 1 else 0.0
    res = {"n": n, "meteor": sum(ms) / n, "se": se,
           "rougeL": (sum(rs) / n) if rs else None, "tag": tag}
    print(f"  METEOR {res['meteor']:.4f} ± {se:.4f} (SE)"
          + (f" · ROUGE-L {res['rougeL']:.4f}" if rs else "")
          + f" · n={n}")
    return res


# =============================================================================
# 9. STAGE submit
# =============================================================================
def stage_submit(answers: dict, expected: set, out_zip: Path) -> None:
    """Bẫy số 3. Kiểm TRƯỚC khi ghi, rồi đọc lại từ đĩa để xác nhận."""
    got = set(answers)
    if got != expected:
        raise SystemExit(f"Tập khoá lệch ground truth: thiếu {len(expected-got)}, "
                         f"thừa {len(got-expected)} -> scorer raise, 0 điểm TOÀN BÀI.")
    bad = [q for q, a in answers.items() if not isinstance(a, str) or not a.strip()]
    if bad:
        raise SystemExit(f"{len(bad)} câu có answer rỗng/không phải string: {bad[:5]}")

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
        if json.loads(zf.read("submission.json").decode("utf-8")) != payload:
            raise SystemExit("nội dung trong zip không khớp — ghi hỏng")
    lens = sorted(len(a.split()) for a in answers.values())
    print(f"  ✅ {out_zip} · {len(payload)} câu · độ dài trung vị "
          f"{lens[len(lens)//2]} từ (đáp án train: 312)")


# =============================================================================
# tiện ích cache
# =============================================================================
def pool_fingerprint(args) -> dict:
    """Cấu hình nào làm ĐỔI nội dung cache pool. Đổi một trong số này thì phần đã cache
    không còn so được với phần tính mới — phải bỏ hết, không được trộn."""
    return {"channels": sorted(args.channels), "bm25": args.top_k_bm25,
            "dense": args.top_k_dense}


def article_fingerprint(args) -> dict:
    return {"doc_k": args.doc_k, "max_dieu": args.max_dieu_per_doc,
            "max_cands": args.max_cands, "keep": args.keep_articles,
            "max_length": args.max_length, **pool_fingerprint(args)}


def cache_out(kind: str, tag: str) -> Path:
    """Nơi GHI cache. Mỗi shard một file riêng — hai tiến trình không bao giờ ghi chung
    một file, nên không cần khoá ghi và không có chuyện file gzip bị đan xen."""
    return CACHE / (f"{kind}.shard{tag}.json.gz" if tag else f"{kind}.json.gz")


def cache_inputs(kind: str) -> list[Path]:
    """Mọi file cần ĐỌC cho một loại cache: bản không shard cộng tất cả bản shard."""
    import glob as _glob
    paths = []
    base = CACHE / f"{kind}.json.gz"
    if base.exists():
        paths.append(base)
    paths += [Path(p) for p in sorted(_glob.glob(str(CACHE / f"{kind}.shard*.json.gz")))]
    return paths


def load_cache_all(kind: str, fp: dict, key: str) -> dict:
    merged = {}
    for p in cache_inputs(kind):
        merged.update(load_cache(p, fp, key))
    return merged


def load_cache(path: Path, fp: dict, key: str):
    """Đọc cache nếu vân tay cấu hình khớp, không thì coi như chưa có.

    Cache cộng dồn theo câu hỏi: chạy 300 câu rồi mở rộng lên 1.500 câu thì chỉ tính
    1.200 câu mới. Nhưng CHỈ đúng khi mọi câu trong file được sinh bằng cùng một cấu
    hình — nên phải chốt bằng vân tay, chứ trộn hai cấu hình vào một file là tạo ra
    một tập kết quả không cấu hình nào tái lập được.
    """
    if not path.exists():
        return {}
    try:
        d = _load_gz(path)
    except (OSError, EOFError, json.JSONDecodeError):
        print(f"    cache {path.name} đọc không được — bỏ, tính lại")
        return {}
    if d.get("fp") != fp:
        print(f"    cache {path.name} sinh bằng cấu hình khác — bỏ, tính lại")
        return {}
    return d.get(key, {})


def _save_gz(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    print(f"    -> {path}")


def _load_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# main
# =============================================================================
def run_shards(qids: list[str], todo: list[str], n: int, args) -> None:
    """Chia `todo` cho n tiến trình con, mỗi con tự xin MỘT GPU riêng.

    Vì sao là tiến trình chứ không phải luồng hay DataParallel:
      - CUDA_VISIBLE_DEVICES chỉ đọc được MỘT LẦN cho mỗi tiến trình, nên một tiến
        trình = một thẻ là cách duy nhất giữ được kỷ luật "mỗi job một thẻ".
      - DataParallel chia theo BATCH nên vẫn phải giữ mọi thẻ suốt cả job; ở đây mỗi
        con chạy độc lập, con nào xin được thẻ thì chạy, không có điểm đồng bộ.
      - Mỗi con ghi file cache RIÊNG (cache_out) rồi cha gộp lại — không ghi chung file.

    Tốc độ rerank không tăng theo batch (xem RERANK_PEAK_MIB) nhưng tăng gần tuyến tính
    theo số thẻ, vì mỗi thẻ chạy một luồng công việc tách rời.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    chunks = [todo[i::n] for i in range(n)]
    chunks = [c for c in chunks if c]
    procs = []
    for i, part in enumerate(chunks):
        tag = f"{i}of{len(chunks)}"
        qpath = CACHE / f"shard_qids_{tag}.json"
        with qpath.open("w", encoding="utf-8") as f:
            json.dump(part, f)
        cmd = [sys.executable, str(SCRIPT), "--stage", "pool", "article",
               "--only-qids", str(qpath), "--shard-tag", tag,
               "--dev-size", str(args.dev_size),
               "--channels", ",".join(args.channels),
               "--doc-k", str(args.doc_k), "--max-cands", str(args.max_cands),
               "--max-dieu-per-doc", str(args.max_dieu_per_doc),
               "--keep-articles", str(args.keep_articles),
               "--max-length", str(args.max_length),
               "--allow-gpus", args.allow_gpus, "--need-mib", str(args.need_mib)]
        env = {**os.environ, "RUN_QA_TAG": f".{tag}", "RUN_QA_REEXEC": "1"}
        log = OUT / f"shard_{tag}.log"
        OUT.mkdir(parents=True, exist_ok=True)
        procs.append((tag, len(part), subprocess.Popen(
            cmd, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)))
        print(f"  shard {tag}: {len(part)} câu -> {log}")

    print(f"  đang chờ {len(procs)} shard ...", flush=True)
    failed = []
    for tag, cnt, p in procs:
        rc = p.wait()
        print(f"  shard {tag} ({cnt} câu) xong, mã thoát {rc}")
        if rc != 0:
            failed.append((tag, OUT / f"shard_{tag}.log"))
    if failed:
        raise SystemExit("Shard hỏng:\n  - " + "\n  - ".join(
            f"{t}: xem {l}" for t, l in failed))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", nargs="+", default=["all"], choices=STAGES + ["all"],
                   help="mặc định all: chạy hết, tự bỏ qua stage đã có cache")
    p.add_argument("--force", nargs="*", default=[], choices=STAGES + ["all"],
                   help="chạy lại stage dù đã có cache")
    p.add_argument("--dev-size", type=int, default=1000)
    p.add_argument("--no-public", action="store_true",
                   help="chỉ chạy dev, không predict public (dùng khi đang dò cấu hình)")

    g = p.add_argument_group("retrieval")
    g.add_argument("--channels", default="bm25,bge,e5",
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   help="thêm 'sparse' nếu chấp nhận tải BAAI/bge-m3 (~2,2 GB); "
                        "đo được nó chỉ đáng +0,10 điểm recall@5 — xem đầu file")
    g.add_argument("--top-k-bm25", type=int, default=50)
    g.add_argument("--top-k-dense", type=int, default=30)
    g.add_argument("--doc-k", type=int, default=5, help="số document mở ra cắt Điều")
    g.add_argument("--max-dieu-per-doc", type=int, default=40)
    g.add_argument("--max-cands", type=int, default=150)
    g.add_argument("--keep-articles", type=int, default=5)
    g.add_argument("--max-length", type=int, default=512)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--batch-size-rerank", type=int, default=None,
                   help="mặc định: tự chọn theo khe VRAM chiếm được (tốc độ không "
                        "phụ thuộc batch, xem RERANK_PEAK_MIB)")

    g = p.add_argument_group("compose")
    g.add_argument("--top-n", type=int, default=1, help="số Điều đưa vào đáp án")
    g.add_argument("--lead", choices=["cancu", "theo", "none"], default="cancu")
    g.add_argument("--concl", choices=["none", "echo", "echo2", "q"], default="echo2",
                   help="câu kết: echo2 tốt nhất trên dev 501 câu (0,5630 vs 0,5151 khi "
                        "không có) — đã xác nhận bằng paired test và split-half")
    g.add_argument("--strip-dieu", dest="strip_dieu", action="store_true", default=True)
    g.add_argument("--no-strip-dieu", dest="strip_dieu", action="store_false")
    g.add_argument("--drop-deco", dest="drop_deco", action="store_true", default=True)
    g.add_argument("--no-drop-deco", dest="drop_deco", action="store_false")
    g.add_argument("--show", type=int, default=0, help="in ra N đáp án đầu để soi bằng mắt")

    g = p.add_argument_group("máy dùng chung")
    g.add_argument("--gpu", type=int, default=None, help="chỉ định thẳng, bỏ qua bộ chọn")
    g.add_argument("--allow-gpus", default=os.environ.get("ALLOW_GPUS", "4 5 6 7"))
    g.add_argument("--need-mib", default="auto",
                   help="MiB tối thiểu đòi hỏi ở một thẻ; 'auto' = mức đo được thật "
                        "sự cần (~2.832 MiB) thay vì một con số đặt tay")
    g.add_argument("--reserve-gb", type=float, default=0.0,
                   help="giữ chỗ VRAM tới mức đỉnh; 0 = không giữ, chỉ dùng đúng "
                        "phần mình cần (thân thiện hơn với máy dùng chung)")
    g.add_argument("--ballast-gb", type=float, default=1.0)
    g.add_argument("--hf-home", default=None)
    g.add_argument("--tag", default="", help="hậu tố tên file submission")
    g.add_argument("--shards", type=int, default=1,
                   help="chia việc GPU cho N tiến trình con, mỗi con MỘT thẻ riêng. "
                        "Tốc độ tăng gần tuyến tính theo N (rerank bão hoà compute "
                        "trên một thẻ, nên chỉ nhiều thẻ mới nhanh hơn)")
    g.add_argument("--only-qids", default="",
                   help="[nội bộ] chỉ xử lý các qid trong file JSON này")
    g.add_argument("--shard-tag", default="",
                   help="[nội bộ] hậu tố file cache của shard này")
    return p.parse_args()


def main():
    args = parse_args()
    stages = set(STAGES if "all" in args.stage else args.stage)
    force = set(STAGES if "all" in args.force else args.force)
    t0 = time.time()

    print("=== môi trường ===")
    check_interpreter()
    claim_single_instance()
    pin_hf_home(args.hf_home)
    print("=== preflight ===")
    preflight(args)

    with (DATA / "train.json").open(encoding="utf-8") as f:
        train = json.load(f)
    with (DATA / "public-official.json").open(encoding="utf-8") as f:
        public = json.load(f)

    print("=== corpus ===")
    doc_index = load_doc_index()
    dev_qids = build_dev_split(train, doc_index, args.dev_size)

    questions = {q: train[q]["question"] for q in dev_qids}
    public_qids = [] if args.no_public else sorted(public)
    questions.update({q: public[q]["question"] for q in public_qids})
    qids = sorted(questions)
    print(f"  tổng {len(qids)} câu ({len(dev_qids)} dev + {len(public_qids)} public)")

    # ---- GPU: chỉ chiếm khi thật sự có việc cho nó -------------------------
    pool_fp, art_fp = pool_fingerprint(args), article_fingerprint(args)
    if "pool" in force or "article" in force:
        for kind in (["pool"] if "pool" in force else []) + ["articles"]:
            for p in cache_inputs(kind):
                p.unlink()
    have_pool = load_cache_all("pool", pool_fp, "pool")
    have_art = load_cache_all("articles", art_fp, "arts")

    # Shard chỉ làm phần qid được giao; phần còn lại coi như đã có để không tính lại.
    if args.only_qids:
        with open(args.only_qids, encoding="utf-8") as f:
            mine = set(json.load(f))
        qids = [q for q in qids if q in mine]
        print(f"  shard {args.shard_tag}: nhận {len(qids)} câu")

    # Chỉ tính phần CÒN THIẾU. Nhờ vậy chạy thử 300 câu rồi mở lên 1.500 câu chỉ tốn
    # 1.200 câu mới, thay vì làm lại tất cả.
    todo_pool = [q for q in qids if q not in have_pool]
    todo_art = [q for q in qids if q not in have_art]
    need_pool = "pool" in stages and todo_pool
    need_art = "article" in stages and todo_art
    if "pool" in stages and not todo_pool:
        print(f"  pool: đủ {len(qids)} câu trong cache, bỏ qua")
    if "article" in stages and not todo_art:
        print(f"  article: đủ {len(qids)} câu trong cache, bỏ qua")
    if need_pool or need_art:
        print(f"  cần tính: pool {len(todo_pool)} câu · article {len(todo_art)} câu")
    lock_fd = None
    if (need_pool or need_art) and args.shards > 1 and not args.only_qids:
        print(f"=== chia {args.shards} shard, mỗi shard một GPU ===")
        run_shards(qids, sorted(set(todo_pool) | set(todo_art)), args.shards, args)
        have_pool = load_cache_all("pool", pool_fp, "pool")
        have_art = load_cache_all("articles", art_fp, "arts")
        miss = [q for q in qids if q not in have_art]
        if miss:
            raise SystemExit(f"{len(miss)} câu vẫn thiếu sau khi gộp shard "
                             f"(vd {miss[:3]}) — xem legalqa/output/shard_*.log")
        need_pool = need_art = False
    if need_pool or need_art:
        print("=== GPU ===")
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            print(f"  dùng GPU {args.gpu} (chỉ định thẳng)")
        else:
            allow = [int(x) for x in args.allow_gpus.split()]
            need = min_need_mib() if args.need_mib == "auto" else int(args.need_mib)
            _g, lock_fd, free = acquire_gpu(need, allow)
            if args.batch_size_rerank is None:
                # Batch theo khe THẬT SỰ chiếm được, không theo một con số cố định.
                args.batch_size_rerank, used = plan_batch(free)
                print(f"  batch rerank {args.batch_size_rerank} "
                      f"(đỉnh dự kiến ~{used} MiB) — tốc độ không phụ thuộc batch, "
                      f"batch chỉ để vừa khe")
        if args.batch_size_rerank is None:
            args.batch_size_rerank = 32
        assert_single_gpu()

    if need_pool:
        print("=== stage pool ===")
        stage_pool(todo_pool, questions, args, have_pool)
    if need_art:
        print("=== stage article ===")
        pool, aid_of = {}, {}
        for _p in cache_inputs("pool"):
            _d = _load_gz(_p)
            pool.update(_d["pool"]); aid_of.update(_d["aid_of"])
        miss = [q for q in todo_art if q not in pool]
        if miss:
            raise SystemExit(f"{len(miss)} câu không có trong cache pool (vd {miss[:3]}) "
                             f"— chạy stage pool trước, hoặc bỏ `--stage` để chạy cả hai.")
        stage_article(todo_art, questions, pool, aid_of, doc_index, args, have_art)
    release_gpu_lock(lock_fd)

    if "compose" not in stages:
        print(f"\nxong {time.time()-t0:.0f}s")
        return

    print("=== stage compose ===")
    articles = load_cache_all("articles", art_fp, "arts")
    if not articles:
        raise SystemExit("chưa có cache articles — chạy stage article trước.")
    miss = [q for q in qids if q not in articles]
    if miss:
        raise SystemExit(f"{len(miss)} câu không có trong cache article (vd {miss[:3]}).")
    answers = stage_compose(qids, questions, articles, doc_index, args)
    cfg = (f"top_n={args.top_n} lead={args.lead} concl={args.concl} "
           f"strip_dieu={args.strip_dieu} drop_deco={args.drop_deco}")
    lens = sorted(len(a.split()) for a in answers.values())
    print(f"  {cfg} · độ dài trung vị {lens[len(lens)//2]} từ (đáp án train: 312)")
    if args.show:
        # Nhìn tận mắt vẫn bắt được thứ METEOR không nói: câu dẫn sai số hiệu, thân bài
        # cắt cụt, Điều lấy nhầm sang văn bản khác.
        for q in qids[:args.show]:
            print(f"\n  --- [{q}] {questions[q]}")
            print("  " + answers[q][:600].replace("\n", "\n  ")
                  + (" ..." if len(answers[q]) > 600 else ""))
        print()

    if "eval" in stages:
        print("=== stage eval (dev) ===")
        res = stage_eval(dev_qids, train, answers, tag=cfg)
        OUT.mkdir(parents=True, exist_ok=True)
        hist = OUT / "eval_log.jsonl"
        with hist.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **res},
                               ensure_ascii=False) + "\n")
        print(f"  ghi thêm vào {hist} (không ghi đè — mỗi lần chạy một dòng)")

    if "submit" in stages and public_qids:
        print("=== stage submit ===")
        name = f"submission{('_' + args.tag) if args.tag else ''}.zip"
        stage_submit({q: answers[q] for q in public_qids}, set(public), OUT / name)

    print(f"\nxong {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
