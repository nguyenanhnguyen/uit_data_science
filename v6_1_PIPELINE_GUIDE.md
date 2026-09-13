# v6_1 Pipeline — Hướng dẫn chi tiết

**Ngày soạn:** 2026-09-13  
**Bản cho:** Task 2 (LegalQA) — Kaggle T4×2  
**Phiên bản notebook:** `legalqa_kaggle_v6_1.ipynb` (197,777 bytes, 17 cells)

---

## 📊 Sơ đồ luồng tổng quát

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        INPUT LAYER: Dữ liệu đầu vào                          ║
╠═════════════════════════════════╦═════════════════════════════════════════════╣
║  train.json                     ║  public.json      ║  Corpus files           ║
║  3.436 Q (với citation)         ║  1.000 Q (nộp)    ║  184K chunks (450-word) ║
╚═════════════════════════════════╩═════════════════════════════════════════════╝
                                        ⬇
╔═══════════════════════════════════════════════════════════════════════════════╗
║               PREPROCESSING: Cell 5-7 (0 GPU, ~7 phút)                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Cell 5: Chunking 2-tầng                                                      ║
║    train.json → 184K chunks (tầng 1) + 450K Điều (tầng 2)                    ║
║    ├─ Tầng 1: 450-word documents (tìm kiếm nhanh)                             ║
║    └─ Tầng 2: Articles/Điều (độ chính xác cao)                               ║
║                                                                                ║
║  Cell 6: BM25 Index (2 tầng)                                                  ║
║    184K chunks + 450K Điều → 2 CSR matrices (TF-IDF)                         ║
║    Dùng: Tầng 1 retrieval (nhanh, không GPU)                                 ║
║                                                                                ║
║  Cell 7: Sinh nhãn & Phân tập                                                 ║
║    train.json (3.436 Q) → 3 tập RỜI NHAU:                                    ║
║    ├─ train_positive: 1.707 Q (có citation) → fine-tune encoder             ║
║    ├─ ltr_pool: 1.729 Q (không citation) → train LTR                        ║
║    └─ dev_ids: 143 Q (phẩm chất cao) → test split-half                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                        ⬇
╔═══════════════════════════════════════════════════════════════════════════════╗
║        TRAINING: Cell 8-10 (GPU intensive, ~302 phút)                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Cell 8: Fine-tune 2 Dense Encoder (SONG SONG)     ◄─── 105 phút              ║
║    GPU:0 (Vietnamese_Embedding_v2)   │   GPU:1 (vietlegal-harrier)          ║
║    424 steps, batch 16               │   152 steps, batch 4 (OOM)            ║
║    ~99 phút                          │   ~99 phút                             ║
║    Đầu vào: train_positive (1.707 cặp Q-Điều)                               ║
║    Đầu ra: encoder_a_ft.pth + encoder_b_ft.pth                              ║
║                                                                                ║
║  Cell 9: Encode toàn bộ Corpus          ◄─── 195 phút                        ║
║    184K chunks × 2 encoder fine-tuned                                         ║
║    → dense_vectors_a.npy (184K×768) + dense_vectors_b.npy (184K×768)        ║
║    Peak VRAM: ~3 GB/GPU                                                       ║
║                                                                                ║
║  Cell 10: Nạp Reranker                 ◄─── 2 phút                           ║
║    AITeamVN/Vietnamese_Reranker (zero-shot)                                   ║
║    Nạp trên cả 2 GPU (1 bản mỗi GPU)                                         ║
║    Bug fix: load_reranker_on(device, base_model) thay vì "zero-shot"        ║
║    Chốt chặn: lọc false-negative (bình thường fine-tune được bật ở v6,      ║
║              nhưng v6_1 tắt vì -3.3 METEOR)                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                        ⬇
╔═══════════════════════════════════════════════════════════════════════════════╗
║         RETRIEVAL & RANKING: Cell 11-13 (GPU + CPU, ~77 phút)                 ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Cell 11: Retrieval + Feature Extraction (hàm, không cell riêng)             ║
║    ┌─ Tầng 1 Retrieval: RRF fusion (3 kênh)                                  ║
║    │   ├─ BM25 score                                                          ║
║    │   ├─ Dense score từ encoder A                                            ║
║    │   └─ Dense score từ encoder B                                            ║
║    │   → Top-5 chunks                                                         ║
║    │                                                                           ║
║    ├─ Tầng 2 Reranking: Cross-encoder                                        ║
║    │   ├─ Cắt top-5 chunks thành Điều (tầng 2)                             ║
║    │   └─ Reranker chấm từng Điều → CE scores                               ║
║    │                                                                           ║
║    └─ Feature Extraction: 13 đặc trưng                                       ║
║        ├─ CE score, CE rank, CE gap to top, CE gap to next, CE z-score      ║
║        ├─ Độ dài Điều, unit type (Điều/Tiết/...)                           ║
║        ├─ Khớp câu hỏi (overlap, ratio, số hiệu, Điều số)                  ║
║        └─ Cấu trúc (doc rank, is_first_in_doc)                             ║
║                                                                                ║
║  Cell 12: Harvest Pha A + Dev-eval      ◄─── 74 phút                         ║
║    Xử lý: 900 câu hỏi từ train (27% tập train)                              ║
║    Mỗi câu: retrieve_two_tier() → cache kết quả                             ║
║    Cache gồm:                                                                 ║
║      ├─ Top-5 ứng viên (Điều + metadata)                                    ║
║      ├─ CE score của từng ứng viên                                           ║
║      ├─ 13 đặc trưng (LTR features)                                         ║
║      └─ METEOR của từng ứng viên (chấm thực tế)                             ║
║    Song song: 2 GPU → ~4.52 giây/câu                                        ║
║    Đồng thời: dev-eval 300 câu dev → chọn TOP_N_ANSWER                     ║
║                                                                                ║
║  Cell 13: LTR Training + Split-half Gate   ◄─── 3 phút (CPU)                 ║
║    Train set: ltr_pool (1.729 Q, không citation) → 599 groups sau lọc       ║
║    Nhãn: METEOR rời rạc (0-4) trong từng group (LambdaRank cần thứ tự)      ║
║    Model: LightGBM LambdaRank                                                 ║
║      ├─ 13 features → NDCG@1 (chỉ rank 1 nộp)                              ║
║      ├─ 50 trees, learning_rate 0.05                                        ║
║      └─ min_child_samples 20 (ngăn overfitting)                            ║
║    Cổng split-half:                                                          ║
║      ├─ Chia dev_ids: nửa A (71 Q) + nửa B (72 Q)                         ║
║      ├─ Đo baseline (chỉ reranker) + LTR resort                            ║
║      └─ Nhận LTR iff delta_a > 0 AND delta_b > 0 (CỘNG hai nửa)            ║
║    **Kết quả v6_1:** delta_a = -0.0089, delta_b = -0.0059 → BỎ LTR         ║
║    → Bộ chọn zero-shot reranker (không LTR) được dùng cho submission      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                        ⬇
╔═══════════════════════════════════════════════════════════════════════════════╗
║              OUTPUT & SUBMISSION: Cell 14-15 (GPU + CPU, ~196 phút)          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Cell 14: BUILD SUBMISSION ⭐ CRITICAL                      ◄─── ~81 phút    ║
║    Xử lý: 1.000 câu từ public.json                                           ║
║    Mỗi câu:                                                                   ║
║      ├─ retrieve_two_tier(question)                                          ║
║      ├─ Nếu LTR nhận: apply_ltr_resort() (không áp ở v6_1)                ║
║      └─ render_answer() → câu trả lời dạng text                            ║
║    Kiểm tra:                                                                  ║
║      ├─ Key khớp public.json?                                                ║
║      └─ Answer không rỗng?                                                   ║
║    Ghi: submission.json → submission.zip                                    ║
║    Kiểm lại từ đĩa: đọc ngược .zip để chứng minh nó hợp lệ                ║
║    ✅ BÀI NỘP AN TOÀN TRÊN ĐĨA — từ đây mất cũng không sao                 ║
║                                                                                ║
║  Cell 15: Harvest Mở rộng + Phân tích   ◄─── 196 phút (dư thời gian)        ║
║    Harvest Pha C: câu còn lại (~2.600 Q) → tổng 3.500 Q                    ║
║    Phân loại lỗi 4 nhóm:                                                     ║
║      ├─ ok: METEOR ≥ 0.60 (không phải lỗi) — 41.7%                       ║
║      ├─ ranking_fail: oracle − chosen ≥ 0.05 (đáp án tốt mà không chọn)  ║
║      │   → Đây là lỗi LTR hướng tới — chiếm 21.9%                         ║
║      ├─ retrieval_fail: oracle < 0.40 (không ứng viên tốt) — 16.7%       ║
║      │   → Tầng 1 retrieval vấn đề (chunking/dense)                        ║
║      └─ extraction_weak: chọn đúng nhưng điểm kém — 19.7%                 ║
║          → Template/độ dài/ranh giới Điều vấn đề                           ║
║    Ghi output:                                                               ║
║      ├─ eval_harvest_summary.json (tổng hợp)                               ║
║      └─ eval_harvest_full.json (chi tiết từng câu)                         ║
║    ⚠️ Điểm này KHÔNG dùng để dò tham số (không nộp lại sau Cell 14)       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## 📋 Bảng tóm tắt từng Cell

| Cell | Tên | Thời gian | GPU | Input | Output | Chú thích |
|------|-----|----------|-----|-------|--------|-----------|
| 1 | Setup | 1 phút | Không | — | thư viện | pip install, import |
| 2 | Config | <1 phút | Không | — | biến toàn cục | AGG_MODE, SEED, HARVEST_TARGET_TOTAL |
| 3 | GPU check + utils | 1 phút | Không | — | hàm helper | checkpoint(), resource_snapshot() |
| 4 | Import + regex | <1 phút | Không | — | hằng số | CHUNK_PATTERN, token limits |
| 5 | Chunking 2-tầng | 1 phút | Không | Corpus files | chunks_t1.json, chunks_t2.json | 450-word chunks + Điều entities |
| 6 | BM25 Index | 2 phút | Không | chunks_t1, chunks_t2 | bm25_t1.pkl, bm25_t2.pkl | Inverted index, CSR matrix |
| 7 | Mine Labels | 2 phút | Không | train.json | train_positive, ltr_pool, dev_ids | Phân tập (3 tập rời) |
| 8 | Fine-tune Encoders | 105 phút | ✅ GPU0 + GPU1 (song song) | train_positive | encoder_a_ft.pth, encoder_b_ft.pth | LoRA, batch fallback 32→4 |
| 9 | Encode Corpus | 195 phút | ✅ GPU0 + GPU1 | chunks_t1, encoder_a_ft, encoder_b_ft | dense_vectors_a.npy, dense_vectors_b.npy | 184K×768 mỗi file |
| 10 | Load Reranker | 2 phút | ✅ GPU0 + GPU1 | HF repo ID | reranker_models[dev] | Zero-shot (fine-tune tắt) |
| 11 | Retrieval + Features | Bộ phận của 12/15 | ✅ GPU0 + GPU1 | query, BM25, dense, reranker | (top-5 Điều, 13 features) | RRF 3-kênh + CE rerank |
| 12 | Harvest A + Dev-eval | 74 phút | ✅ GPU0 + GPU1 | 900 Q + resources | harvest_records (cache) | Cache: top-5 + features + METEOR |
| 13 | LTR Training + Gate | 3 phút | CPU | ltr_pool + dev_ids | ltr_model (hoặc None) | LightGBM, split-half gate |
| 14 | Build Submission | 81 phút | ✅ GPU0 + GPU1 | 1000 Q + resources | submission.zip | Answer generation + zip |
| 15 | Harvest C + Analysis | 196 phút | ✅ GPU0 + GPU1 | ~2600 Q + resources | eval_harvest_summary.json, eval_harvest_full.json | Error classification (4 nhóm) |
| 16 | Log | <1 phút | Không | tất cả metrics | experiment_log.jsonl (append) | Resource log: VRAM/RAM peak |

---

## 🔄 Luồng dữ liệu chi tiết

### Input → Preprocessing

```
train.json (3436 Q)           Corpus files                public.json (1000 Q)
        ⬇                              ⬇                          ⬇
   [Cell 7]                      [Cell 5]                      ┌─────┐
   phân tập                     chunk                         │(dùng│
        ⬇                              ⬇                       │Cell │
  ┌──────┴──────┬─────────┐    184K chunks + 450K Điều       │14)  │
  ⬇             ⬇         ⬇              ⬇                      └─────┘
train_pos   ltr_pool  dev_ids        [Cell 6]
(1707)      (1729)    (143)          BM25 index
                                           ⬇
                                    bm25_t1.pkl
```

### Training → Encoding

```
train_positive (1707 Q)              Corpus (184K chunks)
        ⬇                                    ⬇
   [Cell 8]◄──────────── Chạy song song trên 2 GPU
   Fine-tune 2 encoder               [Cell 9]
        ⬇                            Encode corpus
  encoder_a_ft.pth                        ⬇
  encoder_b_ft.pth            dense_vectors_a.npy (184K×768)
                              dense_vectors_b.npy (184K×768)
```

### Retrieval → Harvest → LTR

```
900 Q từ train                BM25      Dense       Reranker
        ⬇                      ⬇          ⬇            ⬇
        └─────────────────────[Cell 11]◄─────────────────
                          retrieve_two_tier()
                                  ⬇
                          Top-5 Điều + CE scores
                                  ⬇
                          [Cell 11]
                          extract_ltr_features()
                                  ⬇
                          13 đặc trưng
                                  ⬇
                          [Cell 12]
                          harvest_records ← CACHE KEY
                          {qid: {cands, feat, ...}}
                                  ⬇
                    ┌─────────────┼─────────────┐
                    ⬇             ⬇             ⬇
                [Cell 13]     [Cell 14]    [Cell 15]
                LTR train     Submission   Analysis
                (từ cache)    (từ cache)   (từ cache)
```

### Output → Submission

```
harvest_records (cache từ Cell 12)
        ⬇
   [Cell 14]
   answer_question_v61() × 1000 Q
        ⬇
   submission.json
        ⬇
   [Cell 14]
   zipfile.ZipFile()
        ⬇
   submission.zip ✅ AN TOÀN TRÊN ĐĨA
        ⬇
   [Cell 15] (phần ăn thêm, không ảnh hưởng nộp)
   Harvest C + phân tích lỗi
        ⬇
   eval_harvest_summary.json
   eval_harvest_full.json
```

---

## 🎯 Dữ liệu trung gian chính

### Tế bào **Cell 12: harvest_records** (PIVOT)

```python
harvest_records = {
    "Q0001": {
        "question": "Điều 12 quy định...",
        "cands": [
            {
                "id": "111_1",              # Điều ID
                "dieu_so": "12",
                "so_hieu": "100/2019/NĐ-CP",
                "n_words": 245,
                "ce_score": 8.10,           # Cross-encoder score
                "meteor": 0.8234            # METEOR nếu chọn cái này
            },
            # ... 4 ứng viên khác (rank 2-5)
        ],
        "feat": [                           # 13 đặc trưng cho LTR
            [8.10, 0, 0.01, ...],          # Ứng viên 1
            [8.09, 1, 0.0, ...],           # Ứng viên 2
            # ... 3 hàng tiếp
        ],
        "n_cands": 5,
        "meteor_chosen": 0.8234             # METEOR của ứng viên rank 1
    },
    # ... 899 câu khác
}
```

**Tại sao quan trọng?**
- Cell 13 dùng `feat` + METEOR của mỗi `cands` → train LTR
- Cell 14 dùng `cands` → lấy ứng viên rank 1 (hoặc LTR resort) → dựng answer
- Cell 15 dùng `meteor_chosen` + `cands[0].meteor` (oracle) → phân loại lỗi

---

## 📊 13 đặc trưng LTR

| Nhóm | Đặc trưng | Ý nghĩa | Ví dụ |
|------|-----------|---------|-------|
| **CE Score** | `ce_score` | Cross-encoder score thô | 8.10 |
| | `ce_rank` | Hạng trong nhóm (0-4) | 0 (rank 1) |
| | `ce_gap_to_top` | Khoảng cách tới top-1 | 0.01 |
| | `ce_gap_to_next` | Khoảng cách tới rank 2 | 0.01 |
| | `ce_z_in_group` | Z-score chuẩn hóa | 1.23 |
| **Text** | `n_words` | Số từ của Điều | 245 |
| | `unit_is_dieu` | 1 nếu là Điều, 0 nếu Tiết | 1.0 |
| **Query Match** | `q_overlap` | Số từ gợi ý khớp | 3 |
| | `q_overlap_ratio` | Tỉ lệ khớp (Q words) | 0.30 |
| | `so_hieu_match` | 1 nếu số hiệu giống câu hỏi | 1.0 |
| | `dieu_no_match` | 1 nếu "Điều X" trong Q khớp | 1.0 |
| **Structure** | `doc_rank` | Hạng văn bản (tầng 1) | 1 |
| | `is_first_in_doc` | 1 nếu Điều đầu tiên trong văn bản | 1.0 |

---

## 🔐 Ba cơ chế bảo vệ Data Contamination

### 1. **Phân tập rời (Cell 7)**

```
train.json (3.436 Q)
    ⬇
    ├─ train_positive (1.707) ──→ fine-tune encoder (Cell 8)
    ├─ ltr_pool (1.729) ──→ train LTR (Cell 13)
    └─ dev_ids (143) ──→ test split-half (Cell 13)

Bảo đảm: Ba tập này không chồng lấp
→ Model fine-tune KHÔNG bao giờ thấy câu train LTR
→ LTR không train trên câu test
```

### 2. **Split-half gate (Cell 13)**

```
dev_ids (143 Q) chia làm 2:
    Nửa A (71 Q) ──────────→ Chọn LTR tham số
    Nửa B (72 Q) ──────────→ ĐỘC LẬP kiểm chứng

Quy tắc: Nhận LTR iff Δ_A > 0 AND Δ_B > 0

→ Ngăn overfitting trên dev, tránh "dev cao public thấp"
```

### 3. **Lệnh sau-submission (Cell 14 → Cell 15)**

```
[Cell 14] Sinh submission.zip
     ⬇ (✅ nộp, an toàn)
[Cell 15] Phân tích (dùng contaminated harvest)

→ Nộp submission trước chẩn đoán
→ Nếu Kaggle ngắt Cell 15, bài nộp vẫn có
→ Điểm harvest không ảnh hưởng nộp
```

---

## ⏱️ Phân bổ thời gian (tổng 637 phút = 10.6 giờ)

```
Khâu                            Thời gian    Phần trăm
─────────────────────────────────────────────────────
Fine-tune 2 encoder (Cell 8)    105 min      16.5%  ◄── Song song GPU0+GPU1
Encode corpus (Cell 9)          195 min      30.6%  ◄── CPU-bound, batch load
Harvest A + dev-eval (Cell 12)   74 min      11.6%
Build submission (Cell 14)       81 min      12.7%
Harvest C + analysis (Cell 15)  196 min      30.8%  ◄── Thời gian dư (tuỳ CPU)
─────────────────────────────────────────────────────
TỔNG                            637 min     100.0%
```

**Mốc quan trọng:**
- Trước Cell 14 hoàn thành: **~441 phút** (7.4 giờ) — submission.zip an toàn
- Sau Cell 14: phần dư để chạy Cell 15 (không ảnh hưởng nộp)

---

## 🔍 Bốn loại lỗi (Error Classification)

Từ harvest_records, mỗi câu được phân loại dựa vào hai con số:
- `chosen_meteor`: METEOR của ứng viên rank 1 (pipeline chọn)
- `oracle_meteor`: METEOR của ứng viên tốt nhất (top-5)

```
┌─────────────────────────────────────────────────────────┐
│  if chosen ≥ 0.60 ──→ "ok"                              │
│     (không phải lỗi, đạt mục tiêu)                      │
│                                                         │
│  elif oracle < 0.40 ──→ "retrieval_fail"               │
│     (tầng 1 không lọc được ứng viên tốt)              │
│     → cần sửa: chunking, BM25, dense encoder            │
│                                                         │
│  elif oracle − chosen ≥ 0.05 ──→ "ranking_fail"        │
│     (đáp án tốt trong tay nhưng không chọn)            │
│     → cần sửa: LTR model, reranker scoring              │
│                                                         │
│  else ──→ "extraction_weak"                             │
│     (chọn đúng Điều nhưng điểm vẫn kém)               │
│     → cần sửa: template, độ dài câu trả lời             │
└─────────────────────────────────────────────────────────┘

v6_1 kết quả (3.500 Q):
  ok: 41.7% (1.458 Q)          ✅
  ranking_fail: 21.9% (766 Q)  ⚠️ Chỗ LTR hướng tới
  extraction_weak: 19.7%
  retrieval_fail: 16.7%         ⚠️ Chunking/dense hạn chế
```

---

## ✅ Checklist hiểu pipeline

Khi bạn đã hiểu v6_1, bạn sẽ có thể trả lời:

- [ ] Cell 5-7 dùng bao lâu? Có GPU không? (0 GPU, ~7 phút)
- [ ] Cell 8 chạy trên mấy GPU? Song song hay tuần tự? (2 GPU, song song)
- [ ] Cell 9 mất bao lâu? Tại sao nó không song song được? (195 phút, phải load encoder lần lượt)
- [ ] `harvest_records` chứa gì? Tại sao gọi là "pivot"? (top-5, features, METEOR — ba cell dùng nó)
- [ ] Cell 12 chạy bao lâu? Vì sao không phải 3.500 câu? (74 phút, 900 câu pha A; thời gian còn lại → pha C)
- [ ] Split-half gate là gì? Tại sao cần? (test LTR riêng nửa A, nửa B → ngăn overfitting)
- [ ] Cell 14 làm gì? Tại sao quan trọng? (sinh submission.zip, phải hoàn thành trước Cell 15)
- [ ] Bốn loại lỗi từ Cell 15 dùng để làm gì? (quyết định hướng sửa: retrieval, ranking, hay extraction)
- [ ] LTR bị cổng loại ở v6_1 có ý nghĩa gì? (13 đặc trưng không bắt dư địa, cần đặc trưng khác)

---

## 🚀 Hướng tiếp theo (dựa trên v6_1 results)

### Nếu `ranking_fail` ↑ (21.9%) và LTR được nhận

→ Fine-tune reranker (bật `USE_RERANKER_FINETUNE=True`) với chốt chặn đã sửa

### Nếu `ranking_fail` ↑ và LTR **bị cổng loại**

→ Thêm đặc trưng mới:
- Khớp tiêu đề Điều với câu hỏi
- Vị trí Điều trong văn bản (gần hay xa)
- Semantic similarity từ query encoder

### Nếu `retrieval_fail` ↑ (>20%)

→ Sửa tầng 1 (chunking, BM25, dense encoder):
- Thay dense encoder (test Qwen, BGE-M3)
- Query expansion (PRF)
- Dual-tầng re-ranking (tầng 1 cắt tốt hơn)

### Nếu `extraction_weak` ↑ (>25%)

→ Sửa template `render_answer()`:
- Điều chỉnh độ dài câu trả lời
- Ranh giới cắt Điều (prefix/suffix padding)
- Khớp văn phong train.json
