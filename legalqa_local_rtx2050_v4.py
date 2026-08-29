#!/usr/bin/env python
"""
legalqa_local_rtx2050_v4_3.py — LegalQA (UIT DSC2026 Task 2), RTX 2050 4GB — fix AMP reranker + reuse dense checkpoint.

V4.3 HOTFIX:
- sửa ValueError 'Attempting to unscale FP16 gradients' ở reranker fine-tune;
- dùng HYBRID PRECISION trên local4gb: frozen base lưu FP16, 13.6M trainable params giữ FP32;
  autocast FP16 + GradScaler lúc forward/backward => tiết kiệm VRAM nhưng scaler vẫn hợp lệ;
- in dtype trainable/frozen để chẩn đoán;
- tự tái sử dụng dense checkpoint đã train thành công trong cache/checkpoints nếu có,
  tránh mất lại ~84 phút; set FORCE_DENSE_FINETUNE=1 nếu muốn ép train lại.

V4.2 HOTFIX:
- sửa worker subprocess bị IndentationError do chuỗi worker_code giữ indentation của main();
- compile worker trước khi chạy, tự in 100 dòng log cuối nếu worker exit != 0;
- sửa warmup_steps=0.05 thành warmup_ratio=0.05;
- dataloader_num_workers=0 trên Windows;
- log Bước 4 phản ánh đúng profile local4gb/dual_large;
- hỗ trợ biến môi trường LEGALIR_TRAIN_PATH;
- cảnh báo rõ nếu legalir_train.json chưa thực sự được dùng;
- local4gb fine-tune reranker theo kiểu last-layer + head, Adafactor, max_length=384
  để tránh full fine-tune 568M + AdamW vượt 4GB VRAM;
- tạm offload dense encoder sang CPU trong lúc fine-tune reranker rồi đưa lại GPU sau đó.

V4 MERGE:
- thêm legalir_train.json (Task 1 document-level labels) để bổ sung câu không resolve citation;
- citation Task 2 vẫn ưu tiên, Task 1 không ghi đè;
- MAX_TRAIN_EXAMPLES=9000;
- CONCL="echo2" (tối đa lặp câu hỏi 2 lần);
- profile local4gb mặc định dùng bkai Vietnamese bi-encoder + BM25 + fine-tuned reranker;
- profile dual_large vẫn giữ BGE-M3 + multilingual-e5-large để thử trên máy mạnh/Kaggle;
- checkpoint nằm trong cache/ thay vì tạo folder ngoài; submission.json tạm cũng nằm trong cache.

Kiến trúc nguồn v3 trước đó:
2 dense encoder khác họ (BAAI/bge-m3 + intfloat/multilingual-e5-large) fine-tune SONG
SONG trên 2 GPU riêng (subprocess, tự lùi tuần tự nếu chỉ 1 GPU) + fusion RRF 3 kênh
(BM25 + bge-m3-ft + e5-ft-large) + FINE-TUNE RERANKER (margin ranking loss trên chính
nhãn citation Task 2).

BẢN SỬA v3: TỰ NHẬN DIỆN MÔI TRƯỜNG (Kaggle vs máy cá nhân) qua sự tồn tại của
/kaggle/input — cùng MỘT nguồn code (chia sẻ với legalqa_kaggle_t4x2.ipynb) giờ chạy
đúng đường dẫn ở CẢ HAI nơi: Kaggle dùng /kaggle/input/..., máy cá nhân đặt file .py
CẠNH train.json/public-official.json/selected-contexts/ (đúng quy ước legalqa_local.py).
Trên máy cá nhân, KHÔNG có trần thời gian phiên như Kaggle nên TIME_BUDGET_SEC/
FINETUNE_TIME_BUDGET_SEC/RERANKER_FT_TIME_BUDGET_SEC được nới rất rộng (không phải để
ép nhanh, chỉ để không treo vĩnh viễn nếu có bug). Batch khởi điểm GIỮ NGUYÊN mức cho
GPU nhiều VRAM — OOM-backoff đã có sẵn ở mọi bước tự lùi khi máy yếu hơn, không cần hạ
tay trước (đúng triết lý "dùng tối đa tài nguyên, chỉ lùi khi thật sự hết" đã áp dụng
xuyên suốt từ legalqa_local.py).

Đây là bản .py TRÍCH XUẤT Y HỆT logic của legalqa_kaggle_t4x2.ipynb (không viết lại tay,
để tránh lệch giữa 2 bản) — dùng khi muốn chạy như một script thuần (vd qua `python
legalqa_dual_encoder.py`) thay vì notebook, trên Kaggle hoặc máy riêng.

CÁCH DÙNG (đặt cạnh train.json, public-official.json, selected-contexts/ nếu chạy ngoài
Kaggle):
    pip install -q -U sentence-transformers datasets "accelerate>=1.1.0" nltk rouge_score sentencepiece
    python legalqa_local_rtx2050_v4_3.py

KHÔNG dùng ký hiệu `!pip install` (cú pháp riêng của Jupyter) — cài thư viện bằng lệnh
pip ở trên TRƯỚC khi chạy file này.

VRAM: kiến trúc 2 encoder (~568M+560M) + reranker fine-tune (~568M) chạy được cả trên
GPU nhỏ (đã có OOM-backoff), nhưng sẽ CHẬM hơn nhiều so với 16GB/thẻ của Kaggle — chấp
nhận được nếu không có giới hạn thời gian phiên (đúng trường hợp máy cá nhân).
"""


def main() -> None:
    # Cell 2: Đường dẫn và tham số
    import os

    # BẢN SỬA (log lỗi thật: chạy trên máy cá nhân RTX 2050 4GB nhưng CONTEXT_DIR trước đây cứng
    # đường dẫn Kaggle /kaggle/input/... -> FileNotFoundError): tự nhận diện MÔI TRƯỜNG thay vì
    # cứng 1 kiểu đường dẫn — /kaggle/input CHỈ tồn tại thật trên Kaggle, nên dùng chính nó làm
    # điều kiện phát hiện. Nhờ vậy CÙNG MỘT nguồn code chạy đúng trên cả 2 nơi (notebook Kaggle
    # tự lấy đường dẫn Kaggle, .py trên máy cá nhân tự lấy đường dẫn CẠNH SCRIPT — đúng quy ước
    # HERE-relative của legalqa_local.py, dữ liệu đặt cùng thư mục file .py).
    IS_KAGGLE = os.path.isdir("/kaggle/input")

    if IS_KAGGLE:
        DATA_DIR = "/kaggle/input/datasets/anhnguyen7508/uit-data-science-dataset/"
        CONTEXT_DIR = os.path.join(DATA_DIR, "selected-contexts/selected-contexts/")
        TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
        WARMUP_PATH = os.path.join(DATA_DIR, "warmup.json")
        LEGALIR_TRAIN_PATH = os.environ.get("LEGALIR_TRAIN_PATH", os.path.join(DATA_DIR, "legalir_train.json"))
        PUBLIC_PATH = os.path.join(DATA_DIR, "public-official.json")
        # /kaggle/input CHỈ ĐỌC. Mọi thứ ghi ra phải nằm ở /kaggle/working (được lưu khi
        # "Save Version" — đây là nơi submission.zip phải nằm) hoặc /kaggle/temp (KHÔNG được
        # lưu, mất khi session kết thúc — dùng cho cache/model tải về, đỡ tốn quota output).
        OUT_DIR = "/kaggle/working"
        CACHE_DIR = "/kaggle/temp/legalqa_cache"
    else:
        # Máy cá nhân (hoặc bất kỳ máy nào không phải Kaggle): đặt file .py CẠNH train.json,
        # public-official.json, selected-contexts/ — đúng quy ước của legalqa_local.py, không có
        # trần thời gian phiên nào để lo (khác Kaggle) nên OUT_DIR/CACHE_DIR cũng nằm ngay cạnh
        # script, dễ tìm dễ dọn.
        HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        DATA_DIR = HERE
        CONTEXT_DIR = os.path.join(HERE, "selected-contexts")
        TRAIN_PATH = os.path.join(HERE, "train.json")
        WARMUP_PATH = os.path.join(HERE, "warmup.json")
        LEGALIR_TRAIN_PATH = os.environ.get("LEGALIR_TRAIN_PATH", os.path.join(HERE, "legalir_train.json"))
        PUBLIC_PATH = os.path.join(HERE, "public-official.json")
        OUT_DIR = HERE
        CACHE_DIR = os.path.join(HERE, "cache")

    HF_CACHE_DIR = os.path.join(CACHE_DIR, "hf")
    NLTK_CACHE_DIR = os.path.join(CACHE_DIR, "nltk_data")
    TRAINER_TMP_DIR = os.path.join(CACHE_DIR, "trainer_tmp")
    for _d in (OUT_DIR, HF_CACHE_DIR, NLTK_CACHE_DIR, TRAINER_TMP_DIR):
        os.makedirs(_d, exist_ok=True)
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_CACHE_DIR, "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    # Tham số
    CHUNK_SIZE = 512      # không sử dụng — chunk theo Điều (xem Bước 1), giữ lại đúng như đề bài
    TOP_K_RETRIEVE = 100  # số ứng viên lấy ra sau RRF fusion (BM25 + dense)
    TOP_K_RERANK = 5      # trần trên cho số Điều đưa vào 1 câu trả lời (top_n tĩnh VÀ trần adaptive-k)
    USE_FINETUNE = True   # bật mặc định — có đủ VRAM (16GB/thẻ Kaggle, hoặc OOM-backoff tự lùi
                           # trên máy yếu hơn) thì fine-tune, không cần đắn đo trước. Đặt False nếu
                           # muốn chạy thử nhanh hoặc đang tiết kiệm quota GPU trên Kaggle.

    # ==========================================================================
    # PROFILE MÔ HÌNH
    # ==========================================================================
    # Máy cá nhân RTX 2050 4GB mặc định dùng 1 dense encoder 135M để không phải giữ đồng
    # thời 2 model ~560M trên cùng GPU. Có thể thử lại kiến trúc dual-large bằng:
    #   set LEGALQA_PROFILE=dual_large   (Windows CMD)
    #   $env:LEGALQA_PROFILE="dual_large" (PowerShell)
    # Kaggle mặc định vẫn dùng dual_large.
    MODEL_PROFILE = os.environ.get("LEGALQA_PROFILE",
                                   "dual_large" if IS_KAGGLE else "local4gb").strip().lower()

    if MODEL_PROFILE == "dual_large":
        DENSE_MODEL_SPECS = [
            {"name": "bge-m3", "base_model": "BAAI/bge-m3",
             "query_prefix": "", "passage_prefix": ""},
            {"name": "e5-large", "base_model": "intfloat/multilingual-e5-large",
             "query_prefix": "query: ", "passage_prefix": "passage: "},
        ]
        TRAIN_BATCH_SIZE = 64
        TRAIN_MINI_BATCH_SIZE = 16 if IS_KAGGLE else 2
        ENCODE_BATCH_SIZE = 256 if IS_KAGGLE else 16
        RERANK_SUBBATCH = 64 if IS_KAGGLE else 4
    elif MODEL_PROFILE == "local4gb":
        # Model đã dùng ở bản 0.5199/0.4806: nhỏ hơn nhiều, phù hợp RTX 2050 4GB.
        # Giữ chất lượng retrieval nền đã có, tập trung thử dữ liệu Task 1 + reranker FT + echo2.
        DENSE_MODEL_SPECS = [
            {"name": "bkai-legal-bi", "base_model": "bkai-foundation-models/vietnamese-bi-encoder",
             "query_prefix": "", "passage_prefix": ""},
        ]
        TRAIN_BATCH_SIZE = 32          # batch hiệu dụng của CachedMNRL
        TRAIN_MINI_BATCH_SIZE = 4      # batch thật; OOM-backoff còn tự giảm 4->2->1
        ENCODE_BATCH_SIZE = 48         # mức thực tế hơn cho 4GB; encode OOM sẽ tự lùi
        RERANK_SUBBATCH = 4            # cross-encoder là bước nặng VRAM nhất
    else:
        raise ValueError(f"LEGALQA_PROFILE không hợp lệ: {MODEL_PROFILE!r}; "
                         "chọn 'local4gb' hoặc 'dual_large'.")

    DENSE_MAX_SEQ_LEN = 256
    # Không tạo folder checkpoints riêng cạnh script nữa. Mọi checkpoint tạm nằm trong cache/.
    CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # Sau khi một run crash ở bước sau, không cần train lại dense encoder 84+ phút.
    # Chỉ ép train lại khi set: $env:FORCE_DENSE_FINETUNE="1"
    FORCE_DENSE_FINETUNE = os.environ.get("FORCE_DENSE_FINETUNE", "0").strip() == "1"

    MIN_TRAIN_PAIRS = 50
    MAX_TRAIN_EXAMPLES = 9000          # 3000 -> 9000 để tận dụng nhãn Task 1 mới
    N_NEG_PER_ROW = 2

    # Task 1 document-level label: bổ sung cho citation Task 2, KHÔNG ghi đè citation.
    USE_LEGALIR_LABELS = True

    # Kết luận kiểu run_qa.py cũ. echo2 = nối lại câu hỏi đúng 2 lần; không thử 4-6 lần.
    CONCL = "echo2"                    # {"none", "echo1", "echo2"}
    AUTO_ABLATE_CONCL = True           # dev-eval thử none/echo2 gần như không tốn GPU

    # BẢN SỬA: TIME_BUDGET chỉ thật sự cần trên Kaggle (trần phiên GPU ~9-12h). Máy cá nhân KHÔNG
    # có giới hạn phiên nào — đặt trần RẤT RỘNG (không phải vô hạn, để tránh treo vĩnh viễn nếu có
    # bug logic nào đó) thay vì ép chạy nhanh/cắt ngắn không cần thiết.
    TIME_BUDGET_SEC = (8 * 3600) if IS_KAGGLE else (48 * 3600)
    FINETUNE_TIME_BUDGET_SEC = (3 * 3600) if IS_KAGGLE else (6 * 3600)
    DEV_EVAL_SAMPLE_SIZE = 300

    # BẢN SỬA (tái lập được kết quả + ablation warmup + sổ thí nghiệm — theo phân tích
    # chênh lệch điểm .py-vs-Kaggle, xem PHAN_TICH_KY_THUAT.md): trước đây random.seed(42) chỉ
    # đặt ngay trước lúc lấy mẫu dev-eval (Bước 6) — random.sample/random.choice ở Bước 4 (chọn
    # tập con để fine-tune, chọn hard-negative dự phòng) chạy TRƯỚC đó với random state KHÔNG
    # seed, nên 2 lần chạy cùng code vẫn fine-tune trên 2 tập con khác nhau. SEED được set_all_seeds()
    # NGAY SAU khi import xong (Cell 4) — trước Bước 3/4 — để tái lập được. USE_WARMUP cho phép
    # ablation có/không warmup.json ở CÙNG seed. EXPERIMENT_LOG_PATH nằm trong OUT_DIR — trên
    # Kaggle được giữ lại khi "Save Version" (khác cache/ ở /kaggle/temp, mất khi session kết
    # thúc); trên máy cá nhân nằm cạnh script như submission.zip.
    SEED = 42
    USE_WARMUP = True   # đặt False để ablation: chỉ dùng train.json, không gộp warmup.json
    EXPERIMENT_LOG_PATH = os.path.join(OUT_DIR, "experiment_log.jsonl")

    # BẢN SỬA (kết quả thật: dual-encoder 0.5215/0.4829 chỉ nhích rất ít so với single-encoder
    # 0.5199/0.4806 dù retrieval mạnh hơn nhiều -> retrieval không còn là nút thắt chính, reranker
    # ZERO-SHOT giờ nhiều khả năng là trần chặn điểm. Fine-tune reranker trên chính nhãn citation
    # Task 2 -- tái dùng `rows` đã build cho Bước 4, KHÔNG cần dữ liệu thêm.
    USE_RERANKER_FINETUNE = True
    RERANKER_BASE = "AITeamVN/Vietnamese_Reranker"
    RERANKER_FT_TIME_BUDGET_SEC = (60 * 60) if IS_KAGGLE else (3 * 3600)
    RERANKER_FT_BATCH_SIZE = 8 if IS_KAGGLE else 2              # số CÂU HỎI/batch (mỗi câu có 1 positive + N_NEG_PER_ROW
                                             # negative -> batch thật cho reranker lớn hơn số này)
    RERANKER_FT_LR = 1e-5
    # Full fine-tune reranker ~568M + AdamW không thực tế trên 4GB (weights + grads +
    # optimizer states + activations). Local4gb chỉ mở classifier + layer transformer cuối,
    # dùng Adafactor và sequence ngắn hơn. Base model KHÔNG đổi.
    RERANKER_FT_MODE = "full" if IS_KAGGLE else "last1"
    RERANKER_FT_OPTIMIZER = "adamw" if IS_KAGGLE else "adafactor"
    RERANKER_FT_MAX_LENGTH = 512 if IS_KAGGLE else 384
    RERANKER_FT_MARGIN = 1.0                # margin ranking loss: điểm(positive) phải > điểm(negative)
                                             # + margin -- không cần thang điểm chuẩn hoá, chỉ cần đúng
                                             # THỨ TỰ, ổn định hơn BCE/MSE cho reranker logit thô.

    print(f"Môi trường: {'Kaggle' if IS_KAGGLE else 'máy cá nhân (không phải Kaggle)'}")
    print(f"DATA_DIR  = {DATA_DIR}")
    print(f"OUT_DIR   = {OUT_DIR}")
    print(f"CACHE_DIR = {CACHE_DIR}" + ("  (tạm, mất khi session kết thúc)" if IS_KAGGLE else ""))


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
        if IS_KAGGLE:
            print("[CẢNH BÁO] Chỉ thấy 1 GPU — dual_large sẽ chạy tuần tự.")
        else:
            print("OK — 1 GPU local; profile local4gb được thiết kế để chạy tuần tự trên cuda:0.")
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
    from pathlib import Path
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

    # BẢN SỬA: seed toàn cục NGAY SAU khi import xong (trước Bước 3 ở Cell 7, trước Bước 4 ở
    # Cell 8 — hai nơi duy nhất gọi random.sample/random.choice) — xem giải thích đầy đủ ở Cell 2.
    def set_all_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    set_all_seeds(SEED)
    print(f"  Đã seed toàn cục với SEED={SEED} (random/numpy/torch) — trước mọi lời gọi random "
          f"ở Bước 3/4, để nhiều lần chạy cùng code fine-tune trên cùng 1 tập con, tái lập được.")

    # Cell 5: Bước 1 — Chunk corpus theo Điều (neo đầu dòng — tránh rách nội dung khi 1 Điều
    # trích dẫn Điều khác trong thân bài) + trích so_hieu/loai_vb từ NỘI DUNG (không phải tên file)
    def chunk_passage(passage: str, doc_id) -> list:
        matches = list(DIEU_RE.finditer(passage))
        if not matches:
            return [{"id": f"{doc_id}_0", "doc_id": str(doc_id), "dieu_so": "0", "loai_vb": "", "so_hieu": "", "text": passage.strip()}]
        chunks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(passage)
            dieu = m.group(1)
            chunks.append({"id": f"{doc_id}_{dieu}_{i}", "doc_id": str(doc_id), "dieu_so": dieu, "loai_vb": "", "so_hieu": "",
                            "text": passage[start:end].strip()})
        return chunks


    def load_corpus(contexts_dir) -> list:
        contexts_dir = Path(contexts_dir)
        if not contexts_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy {contexts_dir} — kiểm tra lại CONTEXT_DIR ở Cell 2.")
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

    # Cell 6: Bước 2 — BM25 tự viết bằng numpy (inverted index vector hoá — nhanh trên corpus lớn)
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

    # Cell 7: Bước 3 — Sinh nhãn (question -> chunk) từ citation trong train.json (+ warmup.json
    # nếu USE_WARMUP=True, cùng schema — gộp thêm dữ liệu train, KHÔNG gộp vào mẫu dev-eval để
    # tránh lẫn chất lượng nhãn chưa kiểm chứng vào lúc CHỌN cấu hình cuối cùng)
    def extract_citations(answer) -> list:
        # answer có thể KHÔNG phải string (đã gặp thật: warmup.json có answer kiểu list ở một số
        # câu, khác train.json toàn string) — bỏ qua câu đó thay vì crash cả pipeline.
        if not isinstance(answer, str):
            return []
        out = []
        for m in DIEU_CITATION_RE.finditer(answer):
            window = answer[m.end(): m.end() + 60]
            so_m = SO_HIEU_RE.search(window)
            if so_m and so_m.start() <= 40:
                out.append((m.group(1), so_m.group(0)))
        return out


    def _norm_question(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).strip().lower())


    def load_legalir_labels(path: str) -> dict:
        """Đọc legalir_train.json theo schema Task 1: {qid:{question, answer:[doc_id,...]}}.
        Trả map theo cả qid và normalized-question để chịu được trường hợp ID giữa Task 1/2 khác nhau."""
        if not USE_LEGALIR_LABELS or not os.path.exists(path):
            print(f"  Task1 labels: {'tắt' if not USE_LEGALIR_LABELS else 'không thấy file'} -> bỏ qua {path}")
            if USE_LEGALIR_LABELS:
                print("  [QUAN TRỌNG] Run này CHƯA kiểm thử cải tiến Task1 document-level labels. "
                      "Đặt legalir_train.json cạnh script hoặc set LEGALIR_TRAIN_PATH.")
            return {"by_qid": {}, "by_question": {}}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [CẢNH BÁO] Không đọc được legalir_train.json ({e}) -> chỉ dùng citation Task 2.")
            return {"by_qid": {}, "by_question": {}}

        by_qid, by_question = {}, {}
        bad = 0
        for qid, item in data.items():
            if not isinstance(item, dict):
                bad += 1
                continue
            q = item.get("question")
            ans = item.get("answer")
            if isinstance(ans, (str, int)):
                ans = [ans]
            if not isinstance(ans, list):
                bad += 1
                continue
            doc_ids = [str(x) for x in ans if str(x).strip()]
            if not doc_ids:
                bad += 1
                continue
            by_qid[str(qid)] = doc_ids
            if isinstance(q, str) and q.strip():
                by_question[_norm_question(q)] = doc_ids
        print(f"  Task1 labels: đọc {len(data)} câu -> {len(by_qid)} nhãn hợp lệ"
              + (f", bỏ {bad} dòng lỗi." if bad else "."))
        return {"by_qid": by_qid, "by_question": by_question}


    def build_train_pairs(train_data: dict, all_chunks: list, bm25, legalir_labels: dict):
        """Tạo 1 positive chunk / question.
        Ưu tiên: citation Task2 (section-level thật) > Task1 document-level thật.
        Với Task1, chọn chunk BM25 tốt nhất NẰM TRONG document đúng; đây chỉ là chọn section
        đại diện bên trong document đã được gắn nhãn đúng, không thay nhãn Task1 bằng pseudo-doc."""
        so_hieu_index = {}
        doc_to_chunk_ids = defaultdict(list)
        chunk_by_id = {c["id"]: c for c in all_chunks}
        chunk_idx_by_id = {c["id"]: i for i, c in enumerate(all_chunks)}
        for c in all_chunks:
            doc_to_chunk_ids[str(c.get("doc_id", str(c["id"]).split("_", 1)[0]))].append(c["id"])
            if c["so_hieu"] and c["dieu_so"] != "0":
                so_hieu_index.setdefault((c["dieu_so"], norm_so_hieu(c["so_hieu"])), c["id"])

        positive = {}
        source = {}
        n_bad_answer_type = 0

        # 1) Citation Task 2 trước: chính xác tới Điều.
        for qid, item in train_data.items():
            ans = item.get("answer") if isinstance(item, dict) else None
            if not isinstance(ans, str):
                n_bad_answer_type += 1
                continue
            for dieu, so_hieu in extract_citations(ans):
                key = (dieu, norm_so_hieu(so_hieu))
                if key in so_hieu_index:
                    positive[qid] = so_hieu_index[key]
                    source[qid] = "task2_citation"
                    break

        # 2) Bổ sung Task 1 cho những câu chưa resolve citation.
        n_task1 = 0
        n_task1_unresolved_doc = 0
        by_qid = legalir_labels.get("by_qid", {})
        by_question = legalir_labels.get("by_question", {})
        for qid, item in train_data.items():
            if qid in positive or not isinstance(item, dict):
                continue
            q = item.get("question")
            if not isinstance(q, str) or not q.strip():
                continue
            doc_ids = by_qid.get(str(qid)) or by_question.get(_norm_question(q))
            if not doc_ids:
                continue

            allowed = set()
            for did in doc_ids:
                allowed.update(doc_to_chunk_ids.get(str(did), []))
            if not allowed:
                n_task1_unresolved_doc += 1
                continue

            # Chọn chunk lexical tốt nhất trong document đúng.
            ranked = bm25.top_k(tokenize_simple(q), min(TOP_K_RETRIEVE, len(all_chunks)))
            pos_id = next((all_chunks[i]["id"] for i in ranked if all_chunks[i]["id"] in allowed), None)
            if pos_id is None:
                # Nếu top-K global không chứa document đó, chọn chunk đầu của doc đúng làm fallback.
                pos_id = next(iter(allowed))
            positive[qid] = pos_id
            source[qid] = "task1_document"
            n_task1 += 1

        n_cit = sum(v == "task2_citation" for v in source.values())
        print(f"  Positive labels: citation Task2={n_cit}, bổ sung Task1={n_task1}, tổng={len(positive)}/{len(train_data)}")
        if n_task1_unresolved_doc:
            print(f"  [CẢNH BÁO] {n_task1_unresolved_doc} nhãn Task1 trỏ tới doc_id không có trong selected-contexts.")
        if n_bad_answer_type:
            print(f"  Ghi chú: {n_bad_answer_type} answer không phải string nên không parse citation; "
                  "vẫn có thể được cứu bằng Task1 nếu question khớp.")
        return positive, chunk_by_id, source


    print("=== Bước 3: Sinh nhãn từ train.json" + (" + warmup.json" if USE_WARMUP else "") + " ===")
    with open(TRAIN_PATH, encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"  train.json: {len(train_data)} câu")

    # warmup.json: gộp thêm CHỈ KHI USE_WARMUP=True VÀ file tồn tại VÀ đúng schema
    # {qid: {"question": str, "answer": str}} — lọc chặt cả kiểu dữ liệu. KHÔNG gộp vào mẫu
    # dev-eval ở Bước 6 — dev-eval chỉ dùng train.json gốc để giữ tín hiệu chọn cấu hình đáng tin
    # (xem phần 2.2 trong PHAN_TICH_KY_THUAT.md). USE_WARMUP=False cho phép ablation có kiểm soát.
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

    legalir_labels = load_legalir_labels(LEGALIR_TRAIN_PATH)
    train_positive, chunk_by_id, positive_source = build_train_pairs(
        train_data_for_pairs, all_chunks, bm25, legalir_labels)
    print(f"  Positive pairs: {len(train_positive)}/{len(train_data_for_pairs)}")
    checkpoint("Xong sinh nhãn")


    # Cell 8: Bước 4 — Fine-tune 2 dense encoder SONG SONG THẬT trên 2 GPU riêng (subprocess)
    #
    # CHỦ Ý dùng subprocess (không phải threading/multiprocessing.Process kiểu fork): Cell 3 đã
    # init CUDA context trong tiến trình notebook (gọi torch.cuda.get_device_properties) — fork
    # SAU khi CUDA đã init là lỗi kinh điển ("Cannot re-initialize CUDA in forked subprocess").
    # subprocess.Popen luôn khởi động tiến trình Python HOÀN TOÀN MỚI (tương đương spawn), mỗi
    # tiến trình con tự import torch riêng, tự nhận CUDA_VISIBLE_DEVICES riêng — an toàn tuyệt
    # đối, đúng pattern `run_shards` của bản gốc `run_qa.py` đầu dự án.
    #
    # Checkpoint ĐƯỢC LƯU lần này (khác các bản trước) — bạn cần dùng lại qua nhiều phiên Kaggle,
    # và vì tiến trình con/cha là 2 process riêng, cách DUY NHẤT đưa model đã train về tiến trình
    # cha là qua đĩa (`model.save_pretrained()` rồi `SentenceTransformer(path)` load lại).
    import subprocess
    import textwrap
    import sys  # SỬA: sys.executable dùng để gọi WORKER_SCRIPT bên dưới — thiếu import này\n# là bug thật (Python vẫn cho phép dùng module chưa import NẾU nó tình cờ đã có trong\n# builtins/đã import ở cell khác cùng kernel session — dễ chạy "trót lọt" trong notebook\n# rồi lỗi khó hiểu khi chạy .py độc lập; luôn import tường minh module mình dùng).

    print(f"=== Bước 4: Fine-tune dense retriever — profile={MODEL_PROFILE}, "
          f"{len(DENSE_MODEL_SPECS)} encoder, devices={DEVICES} ===")

    WORKER_SCRIPT = os.path.join(CACHE_DIR, "_train_encoder_worker.py")
    # Worker con — fine-tune MOT SentenceTransformer tren MOT GPU, chay qua subprocess.Popen,
    # nhan tham so qua argv, khong phu thuoc bien toan cuc cua notebook.
    worker_code = textwrap.dedent(r"""
        import argparse, gc, json, os, sys, time, traceback

        def main():
            p = argparse.ArgumentParser()
            p.add_argument("--base-model", required=True)
            p.add_argument("--gpu-index", required=True)
            p.add_argument("--rows-path", required=True)
            p.add_argument("--output-dir", required=True)
            p.add_argument("--max-seq-len", type=int, default=256)
            p.add_argument("--batch-size", type=int, default=32)
            p.add_argument("--mini-batch-size", type=int, default=4)
            p.add_argument("--time-budget-sec", type=float, required=True)
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--query-prefix", default="")
            p.add_argument("--passage-prefix", default="")
            args = p.parse_args()

            # Phải set TRƯỚC import torch.
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

            print(
                f"[worker] python={sys.version.split()[0]} torch={torch.__version__} "
                f"cuda={torch.cuda.is_available()} visible_gpu={args.gpu_index}",
                flush=True,
            )
            if torch.cuda.is_available():
                prop = torch.cuda.get_device_properties(0)
                print(
                    f"[worker] gpu={torch.cuda.get_device_name(0)} "
                    f"vram={prop.total_memory/1024**3:.2f}GB",
                    flush=True,
                )

            from datasets import Dataset
            from sentence_transformers import (
                SentenceTransformer,
                SentenceTransformerTrainer,
                SentenceTransformerTrainingArguments,
            )
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

            batch_size = args.batch_size
            mini_batch_size = args.mini_batch_size
            max_steps, calib_time = 0, None
            t0 = time.time()

            for attempt in range(4):
                try:
                    loss = CachedMultipleNegativesRankingLoss(
                        model, mini_batch_size=mini_batch_size
                    )
                    calib_steps = min(10, max(1, len(dataset) // batch_size))
                    calib_args = SentenceTransformerTrainingArguments(
                        output_dir=args.output_dir + "_tmp",
                        max_steps=calib_steps,
                        per_device_train_batch_size=batch_size,
                        logging_steps=calib_steps + 1,
                        save_strategy="no",
                        report_to=[],
                        disable_tqdm=True,
                        dataloader_num_workers=0,
                        fp16=(device == "cuda:0"),
                    )
                    c0 = time.time()
                    print(
                        f"[{args.base_model}] calib training "
                        f"(batch={batch_size}, mini_batch={mini_batch_size})...",
                        flush=True,
                    )
                    SentenceTransformerTrainer(
                        model=model,
                        args=calib_args,
                        train_dataset=dataset,
                        loss=loss,
                    ).train()
                    calib_time = (time.time() - c0) / calib_steps

                    budget_left = args.time_budget_sec - (time.time() - t0) - 60
                    max_steps = max(
                        0, int(budget_left / max(calib_time, 1e-6))
                    )
                    max_steps = min(
                        max_steps, max(1, (len(dataset) // batch_size) * 8)
                    )
                    print(
                        f"[{args.base_model}] calib {calib_time:.2f}s/step, "
                        f"budget_left={budget_left/60:.1f}m -> {max_steps} step",
                        flush=True,
                    )

                    if max_steps > 0:
                        targs = SentenceTransformerTrainingArguments(
                            output_dir=args.output_dir + "_tmp",
                            max_steps=max_steps,
                            per_device_train_batch_size=batch_size,
                            learning_rate=2e-5,
                            warmup_ratio=0.05,
                            lr_scheduler_type="cosine",
                            logging_steps=max(1, max_steps // 20),
                            save_strategy="no",
                            report_to=[],
                            dataloader_num_workers=0,
                            fp16=(device == "cuda:0"),
                        )
                        SentenceTransformerTrainer(
                            model=model,
                            args=targs,
                            train_dataset=dataset,
                            loss=loss,
                        ).train()
                    break

                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and mini_batch_size > 1:
                        mini_batch_size = max(1, mini_batch_size // 2)
                        print(
                            f"[{args.base_model}] CUDA OOM -> "
                            f"mini_batch_size={mini_batch_size}",
                            flush=True,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise

            model.save_pretrained(args.output_dir)
            meta = {
                "max_steps": max_steps,
                "mini_batch_final": mini_batch_size,
                "calib_time_s": calib_time,
                "elapsed_s": time.time() - t0,
            }
            with open(
                args.output_dir + "_meta.json", "w", encoding="utf-8"
            ) as f:
                json.dump(meta, f)

            print(
                f"[{args.base_model}] DONE -> {args.output_dir}",
                flush=True,
            )

        if __name__ == "__main__":
            try:
                main()
            except Exception:
                traceback.print_exc()
                raise
    """).lstrip()

    # Fail-fast: lỗi syntax/indent của child được phát hiện ngay ở parent.
    compile(worker_code, WORKER_SCRIPT, "exec")
    with open(WORKER_SCRIPT, "w", encoding="utf-8", newline="\n") as f:
        f.write(worker_code)
    print(f"  Worker script syntax: OK -> {WORKER_SCRIPT}")

    # ---- Tạo training rows 1 LẦN trong tiến trình cha (dùng chung cho cả 2 encoder) ----
    def _build_training_rows(train_positive, train_data, chunk_by_id, all_chunks, bm25, n_neg=N_NEG_PER_ROW):
        rows = []
        n = len(train_positive)
        for i, (qid, pos_id) in enumerate(train_positive.items()):
            question = train_data[qid]["question"]
            pos_text = chunk_by_id[pos_id]["text"]
            token_q = tokenize_simple(question)
            ranked = bm25.top_k(token_q, 60)
            pos_doc = str(chunk_by_id[pos_id].get("doc_id", ""))
            neg_ids = [
                all_chunks[i2]["id"] for i2 in ranked[5:60]
                if all_chunks[i2]["id"] != pos_id
                and str(all_chunks[i2].get("doc_id", "")) != pos_doc
            ][:n_neg]
            if len(neg_ids) < n_neg:
                pool = [
                    c["id"] for c in all_chunks
                    if c["id"] != pos_id and str(c.get("doc_id", "")) != pos_doc
                ]
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
    DENSE_CHANNELS = []  # điền ở cuối cell; embeddings điền ở Cell 9

    # BẢN SỬA: build `rows` (anchor/positive/negative) MỘT LẦN, KHÔNG PHỤ THUỘC USE_FINETUNE của
    # dense encoder — Cell 10 (fine-tune reranker) cần dùng lại đúng `rows` này. Trước đây rows chỉ
    # được build bên trong nhánh "if use_finetune" của dense encoder, nên nếu USE_FINETUNE=False thì
    # Cell 10 không có gì để fine-tune reranker dù USE_RERANKER_FINETUNE=True.
    rows_needed = (USE_FINETUNE or USE_RERANKER_FINETUNE) and len(train_positive) >= MIN_TRAIN_PAIRS \
                  and remaining() > 10 * 60
    rows, rows_path = [], None
    if rows_needed:
        train_positive_used = train_positive
        if len(train_positive) > MAX_TRAIN_EXAMPLES:
            sampled_qids = random.sample(list(train_positive.keys()), MAX_TRAIN_EXAMPLES)
            train_positive_used = {qid: train_positive[qid] for qid in sampled_qids}
            print(f"  Có {len(train_positive)} positive pairs, lấy mẫu {MAX_TRAIN_EXAMPLES} "
                  f"(tái lập được nhờ SEED={SEED}).")
        finetune_info["n_pairs_used"] = len(train_positive_used)

        print(f"  Đang tạo training rows (dùng chung cho encoder + reranker)...")
        rows = _build_training_rows(train_positive_used, train_data_for_pairs, chunk_by_id, all_chunks, bm25)
        rows_path = os.path.join(CACHE_DIR, "train_rows.json")
        with open(rows_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        print(f"  {len(rows)} rows -> {rows_path}")

    use_finetune = USE_FINETUNE and bool(rows)
    specs = []
    for i, base in enumerate(DENSE_MODEL_SPECS):
        dev = DEVICES[i % len(DEVICES)]
        gpu = dev.split(":")[-1] if dev.startswith("cuda:") else "0"
        specs.append({
            **base,
            "gpu": gpu,
            "device": dev,
            "out": os.path.join(CHECKPOINT_DIR, f"{base['name']}-ft"),
        })

    if not use_finetune:
        reason = ("USE_FINETUNE=False" if not USE_FINETUNE else
                  ("chưa có rows (xem rows_needed)" if not rows else "?"))
        print(f"  {reason} -> dùng zero-shot cho {len(specs)} dense encoder.")
        finetune_info["reason"] = reason
        from sentence_transformers import SentenceTransformer
        DENSE_CHANNELS = []
        for spec in specs:
            m = SentenceTransformer(spec["base_model"], device=spec["device"])
            m.max_seq_length = DENSE_MAX_SEQ_LEN
            DENSE_CHANNELS.append({
                "name": spec["name"], "model": m, "embeddings": None,
                "query_prefix": spec["query_prefix"], "passage_prefix": spec["passage_prefix"],
            })
    else:
        # Tái sử dụng checkpoint dense đã train xong ở run trước (nếu meta + folder đều còn).
        cached_specs, train_specs = [], []
        for spec in specs:
            meta_path = spec["out"] + "_meta.json"
            valid_cache = os.path.isdir(spec["out"]) and os.path.isfile(meta_path)
            if valid_cache and not FORCE_DENSE_FINETUNE:
                cached_specs.append(spec)
            else:
                train_specs.append(spec)

        if cached_specs:
            print("  [CACHE] Tái sử dụng dense checkpoint đã có:")
            for spec in cached_specs:
                print(f"    - {spec['name']} -> {spec['out']}")
        if FORCE_DENSE_FINETUNE and any(os.path.isdir(s["out"]) for s in specs):
            print("  FORCE_DENSE_FINETUNE=1 -> bỏ qua checkpoint cũ và train lại.")

        # Chỉ song song khi thật sự có nhiều GPU vật lý và còn >=2 model cần train.
        run_parallel = (len(train_specs) > 1 and len(DEVICES) > 1
                        and len({s["gpu"] for s in train_specs}) == len(train_specs))
        divisor = 1.0 if run_parallel else max(1, len(train_specs))
        time_budget_each = max(600.0, min(remaining() - 5 * 60, FINETUNE_TIME_BUDGET_SEC) / divisor)                            if train_specs else 0.0
        if train_specs:
            print(f"  Fine-tune {len(train_specs)} encoder — "
                  f"{'song song' if run_parallel else 'tuần tự'}; ~{time_budget_each/60:.0f} phút/model.")
        else:
            print("  Không cần fine-tune dense lại — tất cả checkpoint cần thiết đã có.")

        def _tail_log(path, n=100):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                return "".join(lines[-n:])
            except Exception as e:
                return f"(không đọc được log: {e})"

        def _launch(spec):
            log_path = os.path.join(CACHE_DIR, f"train_{spec['name']}.log")
            cmd = [sys.executable, WORKER_SCRIPT,
                   "--base-model", spec["base_model"], "--gpu-index", spec["gpu"],
                   "--rows-path", rows_path, "--output-dir", spec["out"],
                   "--max-seq-len", str(DENSE_MAX_SEQ_LEN), "--batch-size", str(TRAIN_BATCH_SIZE),
                   "--mini-batch-size", str(TRAIN_MINI_BATCH_SIZE), "--time-budget-sec", str(time_budget_each),
                   "--seed", str(SEED), "--query-prefix", spec["query_prefix"],
                   "--passage-prefix", spec["passage_prefix"]]
            lf = open(log_path, "w", encoding="utf-8", errors="replace")
            print(f"  Khởi động fine-tune {spec['name']} -> {log_path}")
            return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT), lf

        failed = []
        if run_parallel:
            procs = [(spec, *_launch(spec)) for spec in train_specs]
            for spec, proc, lf in procs:
                rc = proc.wait()
                lf.close()
                print(f"  {spec['name']}: mã thoát {rc}")
                if rc != 0:
                    failed.append(spec["name"])
        else:
            for spec in train_specs:
                proc, lf = _launch(spec)
                rc = proc.wait()
                lf.close()
                print(f"  {spec['name']}: mã thoát {rc}")
                if rc != 0:
                    failed.append(spec["name"])
        if failed:
            print("\n  ===== WORKER FINE-TUNE FAILED =====")
            for name in failed:
                lp = os.path.join(CACHE_DIR, f"train_{name}.log")
                print(f"\n  --- {lp} (100 dòng cuối) ---")
                print(_tail_log(lp, 100))
            print("  ====================================")
            raise SystemExit(
                f"Fine-tune dense lỗi: {failed}. Log thật đã được in ngay phía trên."
            )

        from sentence_transformers import SentenceTransformer
        DENSE_CHANNELS = []
        for spec in specs:
            meta_path = spec["out"] + "_meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                finetune_info["models"][spec["name"]] = meta
                print(f"  {spec['name']}: {meta['max_steps']} step, mini_batch={meta['mini_batch_final']}, "
                      f"{meta['elapsed_s']/60:.1f} phút")
            m = SentenceTransformer(spec["out"], device=spec["device"])
            m.max_seq_length = DENSE_MAX_SEQ_LEN
            DENSE_CHANNELS.append({
                "name": spec["name"], "model": m, "embeddings": None,
                "query_prefix": spec["query_prefix"], "passage_prefix": spec["passage_prefix"],
            })
        finetune_info["used_finetune"] = True
        print(f"  Checkpoint tạm nằm trong {CHECKPOINT_DIR} (bên trong cache/, không tạo folder ngoài).")

    checkpoint(f"Xong Bước 4 ({len(DENSE_CHANNELS)} dense encoder)")


    # Cell 9: Bước 5 — Encode toàn bộ corpus CHO CẢ 2 ENCODER (mỗi encoder dùng multi-process
    # pool riêng, chạy TUẦN TỰ giữa 2 encoder — mỗi lần encode đã tận dụng CẢ 2 GPU, chạy đồng
    # thời 2 pool cùng lúc sẽ tranh nhau GPU chứ không nhanh hơn)
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

    checkpoint(f"Xong encode corpus ({len(DENSE_CHANNELS)} encoder)")


    # Cell 10: Bước 5b — Fine-tune reranker (nếu USE_RERANKER_FINETUNE, tái dùng `rows` của
    # Bước 4) RỒI tải 1 bản MỖI GPU để rerank song song thật ở Bước 6/7 (xem Cell 11)
    #
    # LÝ DO: kết quả dual-encoder thật (0.5215/0.4829) chỉ nhích rất ít so với single-encoder
    # (0.5199/0.4806) dù retrieval mạnh hơn nhiều -> retrieval không còn là nút thắt chính,
    # reranker ZERO-SHOT (AITeamVN/Vietnamese_Reranker, train trên Legal Zalo 2021 — KHÔNG phải
    # đúng format "Điều X" của bài này) nhiều khả năng đang là trần chặn điểm tiếp theo.
    #
    # Huấn luyện bằng vòng lặp PyTorch thuần (KHÔNG dùng CrossEncoderTrainer của
    # sentence-transformers) — tránh phụ thuộc API cross-encoder mới có thể không có ở mọi
    # phiên bản cài qua Cell 1; margin ranking loss (điểm(positive) phải lớn hơn điểm(negative)
    # ít nhất RERANKER_FT_MARGIN) — không cần thang điểm chuẩn hoá, chỉ cần đúng THỨ TỰ, ổn
    # định hơn BCE/MSE cho một đầu hồi quy logit thô như model này.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer


    def load_reranker_on(device: str, source: str):
        for attempt in range(2):
            try:
                print(f"  Đang tải reranker {source} lên {device}"
                      f"{' — thử lại lần 2' if attempt else ''}...")
                tok = AutoTokenizer.from_pretrained(source)
                mdl = AutoModelForSequenceClassification.from_pretrained(source)
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


    # finetune_reranker: trả về (model đã .eval(), tokenizer, meta dict). KHÔNG raise nếu OOM —
    # tự giảm batch_size và thử lại, giống pattern OOM-backoff đã dùng cho dense encoder/rerank.
    def _configure_reranker_trainable(model, mode: str):
        """Trên 4GB: freeze toàn bộ, mở classifier/score + N transformer layer cuối."""
        if mode == "full":
            for p in model.parameters():
                p.requires_grad = True
            return "full", None

        for p in model.parameters():
            p.requires_grad = False

        # Classification/regression head.
        head_names = ("classifier", "score", "classification_head")
        n_head = 0
        for name, p in model.named_parameters():
            if any(k in name.lower() for k in head_names):
                p.requires_grad = True
                n_head += p.numel()

        # Tìm ModuleList encoder lớn nhất; thường chính là danh sách transformer blocks.
        candidates = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.ModuleList) and len(module) >= 4
        ]
        chosen = None
        if candidates:
            chosen_name, chosen = max(candidates, key=lambda x: len(x[1]))
            n_last = 1 if mode == "last1" else 2
            for layer in list(chosen)[-n_last:]:
                for p in layer.parameters():
                    p.requires_grad = True
        else:
            chosen_name = None

        # Nếu model lạ không tìm thấy head/layer, mở toàn bộ để không train "0 param".
        if not any(p.requires_grad for p in model.parameters()):
            for p in model.parameters():
                p.requires_grad = True
            return "full_fallback", None

        return mode, chosen_name


    def finetune_reranker(rows, base_model: str, device: str, time_budget_sec: float,
                           batch_size: int, lr: float, margin: float, seed: int):
        tok = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(base_model)

        actual_mode, layer_list_name = _configure_reranker_trainable(
            model, RERANKER_FT_MODE
        )

        # Local 4GB: HYBRID PRECISION.
        # - frozen base: FP16 để tiết kiệm ~1/2 VRAM;
        # - trainable layer cuối + head: FP32 để GradScaler có thể unscale gradient hợp lệ.
        #
        # Bug v4.2: model.half() biến CẢ trainable params thành FP16. PyTorch GradScaler
        # cố unscale FP16 gradients và raise:
        #   ValueError: Attempting to unscale FP16 gradients.
        # Giữ trainable params FP32 giải quyết đúng nguyên nhân, không cần tắt AMP.
        if device.startswith("cuda") and MODEL_PROFILE == "local4gb":
            model = model.half()
            for p in model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()

        model = model.to(device)
        model.train()

        trainable = [p for p in model.parameters() if p.requires_grad]
        frozen = [p for p in model.parameters() if not p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"  reranker-ft mode={actual_mode}, trainable={n_trainable/1e6:.1f}M/"
            f"{n_total/1e6:.1f}M"
            + (f", layer_list={layer_list_name}" if layer_list_name else "")
        )
        train_dtypes = sorted({str(p.dtype) for p in trainable})
        frozen_dtypes = sorted({str(p.dtype) for p in frozen})
        print(f"  reranker-ft dtype: trainable={train_dtypes}, frozen={frozen_dtypes}")
        if device.startswith("cuda") and any(p.dtype == torch.float16 for p in trainable):
            raise RuntimeError(
                "Trainable reranker params vẫn là FP16 trước GradScaler; hybrid-precision setup lỗi."
            )

        if RERANKER_FT_OPTIMIZER == "adafactor":
            from transformers.optimization import Adafactor
            opt = Adafactor(
                trainable,
                lr=lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        else:
            opt = torch.optim.AdamW(trainable, lr=lr)

        # API mới.
        amp_enabled = device.startswith("cuda")
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        g = random.Random(seed)
        order = list(range(len(rows)))
        bs = batch_size
        t0 = time.time()
        step, n_neg = 0, N_NEG_PER_ROW

        while time.time() - t0 < time_budget_sec:
            g.shuffle(order)
            for i in range(0, len(order), bs):
                batch_idx = order[i:i + bs]
                if not batch_idx:
                    continue
                batch_rows = [rows[j] for j in batch_idx]
                pos_pairs = [[r["anchor"], r["positive"]] for r in batch_rows]
                neg_pairs = [
                    [r["anchor"], r[f"negative_{k+1}"]]
                    for r in batch_rows for k in range(n_neg)
                ]
                try:
                    pos_in = tok(
                        pos_pairs,
                        padding=True,
                        truncation=True,
                        max_length=RERANKER_FT_MAX_LENGTH,
                        return_tensors="pt",
                    ).to(device)
                    neg_in = tok(
                        neg_pairs,
                        padding=True,
                        truncation=True,
                        max_length=RERANKER_FT_MAX_LENGTH,
                        return_tensors="pt",
                    ).to(device)

                    # Với model đã fp16, autocast vẫn an toàn; cast logits -> float32 cho loss.
                    if amp_enabled:
                        try:
                            autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
                        except Exception:
                            autocast_ctx = torch.cuda.amp.autocast()
                    else:
                        from contextlib import nullcontext
                        autocast_ctx = nullcontext()

                    with autocast_ctx:
                        pos_scores = model(**pos_in).logits.view(-1).float()
                        neg_scores = model(**neg_in).logits.view(
                            len(batch_rows), n_neg
                        ).float()
                        pos_exp = pos_scores.unsqueeze(1).expand_as(neg_scores)
                        loss = torch.relu(
                            margin - (pos_exp - neg_scores)
                        ).mean()

                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    step += 1

                except ValueError as e:
                    if "unscale FP16 gradients" in str(e):
                        raise RuntimeError(
                            "AMP dtype invariant bị vi phạm: GradScaler nhận FP16 trainable gradients. "
                            "Bản v4.3 yêu cầu trainable reranker params ở FP32."
                        ) from e
                    raise
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and bs > 1:
                        bs = max(1, bs // 2)
                        print(f"  [CUDA OOM reranker-ft] batch_size -> {bs}")
                        opt.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        continue
                    raise

                if time.time() - t0 >= time_budget_sec:
                    break
                if step % 100 == 0:
                    print(
                        f"    reranker-ft step {step}, loss={loss.item():.4f}, "
                        f"{(time.time()-t0)/60:.1f} phút",
                        flush=True,
                    )

        model.eval()
        meta = {
            "steps": step,
            "batch_size_final": bs,
            "elapsed_s": time.time() - t0,
            "mode": actual_mode,
            "optimizer": RERANKER_FT_OPTIMIZER,
            "max_length": RERANKER_FT_MAX_LENGTH,
            "trainable_params": n_trainable,
            "total_params": n_total,
            "trainable_dtypes": train_dtypes,
            "frozen_dtypes": frozen_dtypes,
        }
        return model, tok, meta


    print("=== Bước 5b: Fine-tune reranker (nếu bật) + tải mỗi GPU 1 bản ===")

    # Trên 4GB, dense model đã encode xong corpus. Tạm đưa nó sang CPU để nhường VRAM
    # cho reranker fine-tune; embeddings vẫn ở RAM/NumPy nên không ảnh hưởng retrieval.
    _dense_offloaded_for_reranker = False
    if MODEL_PROFILE == "local4gb" and torch.cuda.is_available():
        for ch in DENSE_CHANNELS:
            ch["model"] = ch["model"].to("cpu")
        torch.cuda.empty_cache()
        _dense_offloaded_for_reranker = True
        print("  [VRAM] Tạm offload dense encoder -> CPU trong lúc fine-tune reranker.")

    reranker_finetune_info = {"used": False, "reason": None, "steps": 0, "elapsed_s": 0.0}
    reranker_source = RERANKER_BASE
    use_reranker_finetune = USE_RERANKER_FINETUNE and bool(rows) and remaining() > 15 * 60
    if not use_reranker_finetune:
        reason = ("USE_RERANKER_FINETUNE=False" if not USE_RERANKER_FINETUNE else
                  ("không có rows (xem Bước 4)" if not rows else "hết ngân sách thời gian"))
        print(f"  {reason} -> reranker giữ ZERO-SHOT ({RERANKER_BASE}).")
        reranker_finetune_info["reason"] = reason
    else:
        t0 = time.time()
        ft_model, ft_tok, meta = finetune_reranker(
            rows, RERANKER_BASE, DEVICES[0], RERANKER_FT_TIME_BUDGET_SEC,
            RERANKER_FT_BATCH_SIZE, RERANKER_FT_LR, RERANKER_FT_MARGIN, SEED)
        reranker_ckpt = os.path.join(CHECKPOINT_DIR, "reranker-ft")
        ft_model.half().save_pretrained(reranker_ckpt)  # lưu fp16 — nhất quán với cách nạp lại để rerank
        ft_tok.save_pretrained(reranker_ckpt)
        del ft_model
        torch.cuda.empty_cache()
        reranker_source = reranker_ckpt
        reranker_finetune_info.update({"used": True, **meta})
        print(f"  Fine-tune reranker xong: {meta['steps']} step, {meta['elapsed_s']/60:.1f} phút "
              f"-> checkpoint {reranker_ckpt}")

    reranker_models, reranker_tokenizers = {}, {}
    for dev in DEVICES:
        m, t = load_reranker_on(dev, reranker_source)
        if m is not None:
            reranker_models[dev] = m
            reranker_tokenizers[dev] = t

    HAS_RERANKER = len(reranker_models) > 0
    RERANK_DEVICES = list(reranker_models.keys())
    print(f"  Reranker ({'fine-tuned' if reranker_finetune_info['used'] else 'zero-shot'}) sẵn sàng "
          f"trên: {RERANK_DEVICES or '(không tải được — sẽ chạy không rerank)'}")

    # Đưa dense encoder local nhỏ trở lại GPU để query embedding nhanh.
    if _dense_offloaded_for_reranker and DEVICES[0].startswith("cuda"):
        try:
            for ch in DENSE_CHANNELS:
                ch["model"] = ch["model"].to(DEVICES[0]).half()
            print("  [VRAM] Dense encoder đã đưa lại cuda:0 để chạy retrieval.")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("  [VRAM] Không đủ chỗ để giữ dense+rerranker cùng GPU -> dense ở CPU.")
                for ch in DENSE_CHANNELS:
                    ch["model"] = ch["model"].to("cpu")
                torch.cuda.empty_cache()
            else:
                raise

    checkpoint("Xong tải reranker")


    # Cell 11: Hàm retrieval (RRF fusion N kênh — BM25 + N encoder) + rerank theo lô
    # + hạ tầng chạy song song 2 GPU
    from concurrent.futures import ThreadPoolExecutor

    _print_lock = __import__("threading").Lock()

    def rrf_retrieve(question: str, bm25, dense_channels, all_chunks, top_k: int = TOP_K_RETRIEVE):
        """RRF fusion N kênh: BM25 + mỗi encoder trong `dense_channels` (list of {"model",
        "embeddings", "query_prefix"}). Mỗi encoder có thể cần tiền tố khác nhau lúc encode QUERY
        (vd e5: "query: ") — PHẢI khớp tiền tố đã dùng lúc encode CORPUS ở Cell 9, nếu không
        embedding lệch hệ toạ độ mà không lỗi nào báo (bẫy đã ghi ở Cell 2/8)."""
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
        """Chấm điểm lại top `max_candidates` bằng cross-encoder, xử lý theo LÔ NHỎ (`sub_batch`)
        thay vì nhồi hết `max_candidates` vào 1 forward — kết quả điểm số KHÔNG đổi (transformer
        không trộn phép tính giữa các example trong batch, chỉ có padding — lô nhỏ padding ít
        hơn), chỉ nhanh hơn và ổn định VRAM hơn. Trả thêm điểm số cho adaptive_k_cutoff()."""
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
        """Adaptive-k (Taguchi et al. 2025, arXiv:2506.08479): tìm điểm "gãy" tự nhiên trong
        phân phối điểm reranker đã sort giảm dần thay vì luôn cắt ở top_n cố định."""
        if scores is None or len(scores) == 0:
            return min_k
        n = min(len(scores), search_window)
        if n <= 1:
            return min_k
        gaps = [scores[i] - scores[i + 1] for i in range(n - 1)]
        k_star = int(np.argmax(gaps)) + 1
        return max(min_k, min(k_star, max_k))


    def apply_conclusion(answer: str, question: str, mode: str = CONCL) -> str:
        """CONCL giới hạn tối đa echo2 để không đẩy metric bằng lặp 4-6 lần."""
        q = str(question).strip()
        if not q or mode == "none":
            return answer
        if mode == "echo1":
            return f"{answer}\n\n{q}"
        if mode == "echo2":
            return f"{answer}\n\n{q}\n{q}"
        raise ValueError(f"CONCL không hợp lệ: {mode!r}")


    def render_answer(selected_chunks: list, top_n: int, question: str = "", concl_mode: str = CONCL) -> str:
        """Ghép extractive answer từ các Điều đã xếp hạng, sau đó áp dụng CONCL."""
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
        base = "\n\n".join(parts)
        return apply_conclusion(base, question, concl_mode)


    def answer_question(question: str, bm25, dense_channels, all_chunks, top_n: int,
                         reranker_model=None, reranker_tokenizer=None, use_adaptive_k: bool = False) -> str:
        ranked = rrf_retrieve(question, bm25, dense_channels, all_chunks)
        if not ranked:
            return "Không tìm thấy thông tin pháp lý cho câu hỏi này."
        scores = None
        if reranker_model is not None:
            ranked, scores = rerank(question, ranked, reranker_model, reranker_tokenizer)
        n = adaptive_k_cutoff(scores) if (use_adaptive_k and scores is not None) else top_n
        return render_answer(ranked, n, question=question, concl_mode=CONCL)


    def split_evenly(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


    def parallel_process(ids, worker_fn, devices, label: str = "", progress_every: int = 50):
        """Chia `ids` đều cho từng thiết bị trong `devices`, chạy worker_fn(ids_chunk, device)
        ĐỒNG THỜI trên các luồng riêng. PyTorch giải phóng GIL trong lúc chờ CUDA hoàn thành nên
        2 luồng ghim vào 2 GPU vật lý khác nhau chạy song song THẬT (không phải giả song song do
        GIL) — đây là chỗ mang lại tốc độ x~2 cho phần rerank ở Bước 6/7.
        worker_fn(ids_chunk, device, worker_idx) -> dict {qid: kết quả}."""
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

    # Cell 12: Bước 6 — Dev-eval (chọn TOP_N_ANSWER, có dùng reranker không, adaptive-k hay
    # không) ĐỒNG THỜI đo Recall@k — gộp chung 1 lượt retrieval+rerank, không chạy lại 2 lần
    # (xem lý do gộp trong bản máy cá nhân — nguyên tắc giữ nguyên, giờ thêm chạy song song 2 GPU)
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
    # BẢN SỬA: re-seed CỐ Ý ở đây (dùng SEED chung, không phải số 42 rời rạc) — đảm bảo mẫu
    # dev-eval LUÔN CỐ ĐỊNH bất kể USE_WARMUP/USE_FINETUNE bật hay tắt (số lời gọi random.*
    # trước đó thay đổi tuỳ cấu hình sẽ làm lệch state nếu không re-seed ở đây) — cần thiết để
    # ablation so sánh đúng nghĩa "cùng 300 câu, chỉ khác 1 biến". Xem PHAN_TICH_KY_THUAT.md §5.2.
    random.seed(SEED)

    print("=== Bước 6: Dev-eval chọn TOP_N_ANSWER + đo Recall@k ===")
    n_sample = min(DEV_EVAL_SAMPLE_SIZE, len(train_data))
    dev_ids = random.sample(list(train_data.keys()), n_sample)
    recall_ids = [q for q in dev_ids if q in train_positive]
    print(f"  Mẫu dev-eval: {len(dev_ids)} câu ({len(recall_ids)} câu có positive label từ citation/Task1 "
          f"-> dùng luôn để đo Recall@k, không chạy lại retrieval riêng).")

    ks = [1, 3, 5, 10, 30, 100]
    configs = [("BM25+dense (không rerank)", False)]
    if HAS_RERANKER:
        configs.append(("BM25+dense+rerank", True))

    # BẢN SỬA: theo dõi thêm best_r (ROUGE-L của cấu hình thắng) + recall_at_k_by_label — để Cell 14
    # ghi đủ vào sổ thí nghiệm (trước đây các con số này chỉ in ra console rồi mất).
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
            # Không rerank (rẻ, tuần tự đủ nhanh) hoặc chỉ 1 GPU cho reranker.
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
                ranked_chunks = ranked_cache[qid]
                pos_id = train_positive[qid]
                src = positive_source.get(qid, "task2_citation")
                if src == "task1_document":
                    pos_doc = str(chunk_by_id[pos_id].get("doc_id", ""))
                    for k in ks:
                        if any(str(c.get("doc_id", "")) == pos_doc for c in ranked_chunks[:k]):
                            hits[k] += 1
                else:
                    ranked_ids = [c["id"] for c in ranked_chunks]
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
                pred = render_answer(ranked, top_n, question=train_data[qid]["question"], concl_mode=CONCL) if ranked else "Không tìm thấy thông tin pháp lý cho câu hỏi này."
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
                pred = render_answer(ranked, k, question=train_data[qid]["question"], concl_mode=CONCL) if ranked else "Không tìm thấy thông tin pháp lý cho câu hỏi này."
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms) / len(ms), sum(rs) / len(rs)
            print(f"    adaptive-k       METEOR={m:.4f}  ROUGE-L={r:.4f}  (n={len(dev_ids)})")
            if m > best_m:
                best_m, best_r, best_use_rerank, best_use_adaptive = m, r, True, True

    print(f"  => chọn TOP_N_ANSWER={best_n}, dùng reranker={best_use_rerank}, "
          f"dùng adaptive-k={best_use_adaptive} (METEOR={best_m:.4f})")
    # CONCL ablation cực rẻ: chỉ render lại text, không chạy retrieval/reranker thêm.
    # Vì ranked_cache tại đây là cache của config cuối vòng lặp, chỉ dùng để sanity-check
    # none vs echo2; cấu hình submission vẫn theo CONCL khai báo ở đầu để tái lập.
    if AUTO_ABLATE_CONCL and "ranked_cache" in locals():
        for cmode in ("none", "echo2"):
            cms = []
            for qid in dev_ids:
                ranked = ranked_cache[qid]
                if not ranked:
                    pred = "Không tìm thấy thông tin pháp lý cho câu hỏi này."
                else:
                    kk = (adaptive_k_cutoff(scores_cache.get(qid))
                          if best_use_adaptive and scores_cache.get(qid) is not None else best_n)
                    pred = render_answer(ranked, kk, question=train_data[qid]["question"], concl_mode=cmode)
                cms.append(meteor_score([str(train_data[qid]["answer"]).split()], str(pred).split()))
            print(f"    [CONCL ablation] {cmode:5s} METEOR={sum(cms)/len(cms):.4f}")
    top_n_answer, use_reranker, use_adaptive = best_n, best_use_rerank, best_use_adaptive
    eval_info = {"meteor": round(best_m, 4), "rouge_l": (round(best_r, 4) if best_r is not None else None),
                 "recall_at_k": recall_at_k_by_label, "n_dev": len(dev_ids)}
    checkpoint("Xong dev-eval + Recall@k")


    # Cell 13: Bước 7 — Sinh câu trả lời cho public-official.json (song song 2 GPU nếu dùng reranker)
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

    # Cell 14: Bước 8 — Validate + đóng gói submission.zip (vào /kaggle/working) + ghi sổ thí nghiệm
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
        json_path = Path(CACHE_DIR) / "submission.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname="submission.json")
        with zipfile.ZipFile(out_zip) as zf:
            assert zf.namelist() == ["submission.json"]
            reloaded = json.loads(zf.read("submission.json").decode("utf-8"))
            assert reloaded == normalized
        try:
            json_path.unlink(missing_ok=True)
        except Exception:
            pass
        print(f"  OK — {out_zip} ({len(normalized)} câu trả lời, đã kiểm tra lại từ đĩa)")


    print("=== Bước 8: Đóng gói submission.zip ===")
    build_submission(answers, set(questions.keys()), Path(OUT_DIR) / "submission.zip")
    checkpoint(f"XONG — tổng thời gian {elapsed()/60:.1f} phút (trần an toàn {TIME_BUDGET_SEC/3600:.0f} giờ)")

    # BẢN SỬA — "sổ thí nghiệm": 1 dòng JSON/lần chạy, APPEND vào EXPERIMENT_LOG_PATH (nằm ở
    # /kaggle/working nên được giữ lại khi "Save Version", khác cache/ ở /kaggle/temp bị xoá khi
    # session kết thúc). Đủ để so sánh nhiều lần chạy sau này (khác seed, khác USE_WARMUP, khác
    # USE_FINETUNE...) mà không phải lục lại log console — xem PHAN_TICH_KY_THUAT.md §5.8.
    n_empty = sum(1 for a in answers.values() if not a.strip())
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "use_warmup": USE_WARMUP,
        "n_warmup_used": n_warmup_used, "hardware": (f"kaggle_gpu_x{N_GPU}" if IS_KAGGLE else "local_rtx2050_4gb"),
        "model_profile": MODEL_PROFILE, "dense_models": [s["base_model"] for s in DENSE_MODEL_SPECS],
        "use_legalir_labels": USE_LEGALIR_LABELS, "legalir_train_path": LEGALIR_TRAIN_PATH,
        "max_train_examples": MAX_TRAIN_EXAMPLES, "concl": CONCL,
        "n_train_pairs_available": finetune_info["n_pairs_available"],
        "n_train_pairs_used": finetune_info["n_pairs_used"],
        "used_finetune": finetune_info["used_finetune"], "finetune_reason": finetune_info["reason"],
        "finetune_models": finetune_info["models"],  # {"bge-m3": {max_steps,...}, "e5-large": {...}}
        "reranker_finetuned": reranker_finetune_info["used"],
        "reranker_finetune_mode": RERANKER_FT_MODE,
        "reranker_finetune_optimizer": RERANKER_FT_OPTIMIZER,
        "reranker_finetune_max_length": RERANKER_FT_MAX_LENGTH,
        "reranker_finetune_steps": reranker_finetune_info["steps"],
        "reranker_finetune_elapsed_min": round(reranker_finetune_info["elapsed_s"] / 60, 1),
        "checkpoint_dir": CHECKPOINT_DIR if (finetune_info["used_finetune"] or reranker_finetune_info["used"]) else None,
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
    # SỬA (mất log thật: session Kaggle bị reset, /kaggle/working và /kaggle/temp mất theo, không
    # đọc lại được experiment_log.jsonl): LUÔN in JSON đầy đủ ra console dù ghi file thành công hay
    # không. Nếu bạn dùng "Save Version" > "Save & Run All (Commit)", output cell này được giữ lại
    # trong tab Output/Logs của version đó ngay cả khi bạn reset editor sau này — an toàn hơn nhiều
    # so với chỉ trông vào file trong /kaggle/working, vốn cũng có thể mất nếu không commit đúng cách.
    print("  [SỔ THÍ NGHIỆM — copy dòng dưới đây nếu cần đối chiếu sau này]")
    print("  " + json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()