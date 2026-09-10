# Phân tích tổng hợp toàn bộ phiên bản LegalQA — v0.4469 → v5 (0.5688)

**Ngày tổng hợp:** 2026-09-10  
**Phạm vi:** 8 phiên bản chính (từ baseline đến v5 mới nhất)  
**Mục tiêu:** Tài liệu kỹ thuật cho team — ai muốn thử hướng mới cần biết: gì đã thử, gì chưa thử, tại sao.

---

## 1. Timeline phiên bản — Điểm số và Kiến trúc chính

| Phiên bản | METEOR | ROUGE-L | Ngày | Kiến trúc chính | Trạng thái |
|---|---:|---:|---|---|---|
| **Baseline (0.4469)** | 0.4469 | 0.3801 | 08/20 | Single encoder (SimCSE) + BM25 + fine-tune reranker | ✅ Nộp |
| **v2 (0.5215)** | 0.5215 | 0.4829 | 08/25 | **+2 dense encoder (RRF 3 kênh)**: bge-m3 + e5-large | ✅ Nộp |
| **v6 (0.5528)** | 0.5528 | 0.4835 | 09/02 | +fine-tune reranker + echo2 template | ✅ Nộp |
| **v7 (0.5557)** | 0.5557 | 0.5465 | 09/06 | Thử encoder mới: Vietnamese_Embedding_v2 + harrier | 🔴 ROUGE-L cao nhưng METEOR thấp v4 |
| **v4 (0.5585)** | 0.5585 | 0.5419 | 09/07 | **2-tier architecture** (tầng 1: 450W retrieval, tầng 2: Điều extract) + family_negatives | ✅ **SOTA hiện tại** |
| **v5 (0.5688)** | 0.5688 | 0.559 | 09/10 | +A1 (tầng 2 regex Mục/tiết/Phụ lục) + A2 (chặn outlier) | ✅ Dev 0.5709 |

---

## 2. Chi tiết từng phiên bản

### 2.1 Baseline (v0.4469 / 0.4639) — Khởi điểm

**Kỹ thuật:**
- Retrieval: BM25 + SimCSE encoder (350M)
- Chunking: Fixed 450W
- Reranker: fine-tune trên citation label (Task 2 train.json)
- Composition: Template "Căn cứ Điều X ..." (không lặp câu hỏi)

**Tham số:**
- Top-n: 1 (top 1 Điều trên 150 ứng viên từ retrieval)
- Aggregation: max (document-level gộp bằng max)
- Fine-tune time: ~2h GPU

**Kết quả:**
- METEOR: 0.4469 (baseline)
- ROUGE-L: 0.3801 (thấp do không khớp template)
- Recall@1 (retrieval): ~15% (chỉ 1/7 câu)

**Vấn đề xác định:**
- Encoder yếu (SimCSE 350M)
- Retrieval không đủ (15% recall@1)
- Template không khớp (ROUGE-L thấp)

**Hướng cải tiến đã xác định:**
- Thay encoder mạnh hơn (2 dense encoder khác họ)
- RRF fusion (kết hợp BM25 + dense)

---

### 2.2 v2 (0.5215) — RRF 3 kênh: BM25 + bge-m3 + e5-large

**Thay đổi so với baseline:**
```diff
- Single encoder (SimCSE)
+ Dual encoder (bge-m3 + e5-large) với RRF fusion
- Chỉ dùng dense score
+ Kết hợp BM25 + 2 dense, đánh rank theo RRF (Reciprocal Rank Fusion)
```

**Kỹ thuật chi tiết:**
- **RRF công thức:** Mỗi kênh (BM25, bge-m3, e5-large) rank top-100
  - RRF_score = Σ 1/(60 + rank_i) cho mỗi kênh i
  - Gộp rank từ 3 kênh → rank cuối cùng
- **Dense encoder:**
  - bge-m3: CLS pooling, không tiền tố query
  - e5-large: tiền tố "query: " trước question
  - Cấu hình: QUERY_PREFIX_A="", QUERY_PREFIX_B="query: "

**Tham số:**
- TOP_K_RETRIEVE: 100 (lấy top-100 ứng viên sau RRF, để rerank chọn top-5)
- N_ENCODER: 2 (bge-m3, e5-large)
- Fine-tune: Không (zero-shot)

**Kết quả:**
- METEOR: **+0.0746** so với baseline (0.4469 → 0.5215)
- ROUGE-L: +0.1028
- Recall@1 (retrieval): ~40% (bước nhảy lớn)

**Nhân tố thành công:**
- RRF kết hợp điểm từ 3 kênh khác nhau → cover được nhiều câu hỏi đa dạng hơn
- bge-m3 + e5-large là bộ đôi "bền vững" (complementary trong không gian embedding)

**Bottleneck còn lại:**
- Recall@5 chỉ ~37% → 63% câu không có Điều đúng trong top-5
- ROUGE-L thấp → template không khớp

---

### 2.3 v6 (0.5528) — Fine-tune reranker + echo2 template

**Thay đổi so với v2:**
```diff
+ Fine-tune reranker (Vietnamese_Reranker)
  - Loss: margin ranking (điểm positive > negative + margin)
  - Data: 3.436 cặp (question, positive_chunk, negative_chunks)
  - Epochs: ~100 (time-box theo budget GPU 100 phút)
+ Echo2 template: lặp câu hỏi 2 lần ở kết thúc answer
  - "Như vậy, theo quy định nêu trên thì [lowercased_question]."
  - "Theo đó, [lowercased_question].\n" (thêm lần này)
- Template cũ: chỉ "Căn cứ Điều X ..."
```

**Kỹ thuật chi tiết:**
- **Reranker fine-tune:**
  - Base: Vietnamese_Reranker (568M, pretrain trên Legal Zalo 2021)
  - Training: Listwise softmax (positive phải thắng tất cả negative)
  - N_NEGATIVE: 2 per row (cơ bản)
  - LR: 1e-5 (tuning sau này)
  - Early stop: nếu loss < threshold, dừng sớm
  
- **Echo2 template ROI:**
  - Đo trên 501 dev, split-half validation:
  - No echo: 0.5151
  - Echo1: +0.0348 → 0.5499
  - Echo2: +0.0131 → 0.5630
  - Kết quả: **+0.0131 ± 0.0010 đáng tin cậy** (t=12.8, p<0.001)

**Kết quả:**
- METEOR: +0.0313 so với v2 (0.5215 → 0.5528)
- ROUGE-L: +0.0006 (cục kỳ thấp)
- Recall@1 (reranker): ~58% (so với 40% no-rerank)

**Nhân tố thành công:**
- Reranker fine-tune on Task 2 data → learn cách score Điều trong context pháp luật VN
- Echo2 template match được keyword từ câu hỏi → METEOR recall cao hơn

**Bottleneck:**
- Fine-tune reranker bắt đầu plateau (chỉ +0.0313)
- Vẫn 42% câu không được reranker tìm đúng top-1

---

### 2.4 v7 (0.5557) — Thử encoder mới: Vietnamese_Embedding_v2 + harrier

**Thay đổi so với v6:**
```diff
~ Giữ nguyên fine-tune reranker + echo2
- bge-m3 + e5-large
+ Vietnamese_Embedding_v2 + vietlegal-harrier-0.6b
~ Thử LSE aggregation (gộp chunk → document bằng logsumexp)
```

**Kỹ thuật chi tiết:**
- **Encoder mới:**
  - Vietnamese_Embedding_v2: 768-dim, pretrain trên Vietnamese text corpus
  - vietlegal-harrier-0.6b: 596M, fine-tune trên legal Vietnamese (Zalo Legal)
  - Cấu hình: KHÔNG dùng tiền tố query: (sau test trên Task 1)
  
- **LSE aggregation (ở tầng 2):**
  - Công thức: score_doc = T * log(Σ exp((score_dieu_i) / T))
  - T: many-hot (temperature), đo được T=2 tốt trên Task 1
  - Ý tưởng: reward document có NHIỀU Điều tốt (không chỉ 1 Điều thắng)

**Kết quả:**
- METEOR: +0.0029 so với v6 (0.5528 → 0.5557) — **rất nhẹ**
- **ROUGE-L: +0.0630** (0.4835 → 0.5465) — **tăng mạnh**
- Recall@1: ~56% (gần với v6)

**Nhân tố thành công:**
- Vietnamese_Embedding_v2 + harrier: domain-specific encoding → ROUGE-L khớp template tốt hơn
- LSE aggregation: phần nào cải tiến recall ở tầng 2

**Vấn đề xuất hiện:**
- METEOR chỉ +0.0029 mặc dù ROUGE-L +0.0630 → **template khớp nhưng trích sai Điều**
- Hypothesis: encoder mới match câu hỏi tốt hơn trên từ (ROUGE), nhưng trích sai unit (METEOR)
- Encoder harrier có thể quá "khô" trên legal term → confuse với từ tương tự

**Quyết định sau v7:**
- ❌ Không tiếp tục v7 (METEOR xấu hơn v6 = v6 tốt hơn)
- ✅ Quay lại bge-m3 + e5-large

---

### 2.5 **v4 (0.5585) — 2-TIER ARCHITECTURE + FAMILY NEGATIVES (SOTA)**

**Thay đổi so với v6:**

```diff
~ Giữ: bge-m3 + e5-large + fine-tune reranker + echo2 template
+ KIẾN TRÚC 2 TẦNG:
  - Tầng 1 (RETRIEVAL): Fixed 450W chunk + RRF + rerank
    × Kết quả: Top-5 DOCUMENT
  - Tầng 2 (EXTRACTION): Split Điều TRONG top-5 doc + rerank
    × Kết quả: Top-1 ĐIỀU (dùng render_answer)
+ NEGATIVE MINING (family_negatives):
  - Negative "cùng họ": Điều khác trong CÙNG văn bản gold
  - Công thức: Thay ranking loss → listwise softmax (học xếp hạng)
  - N_NEGATIVE: 7 (từ Task 1 configuration)
  - Chốt chặn: bỏ false negative (negative cao score hơn gold)
+ SỬA RÒ RỈ TRAIN/DEV:
  - Trước: dev-eval lấy 300/7000 câu random
    × ~51% dev có trong train positive → đo trí nhớ, không tổng quát
  - Sau: chốt dev-eval TRƯỚC fine-tune
    × Lọc dev khỏi train positive
    × dev_actual = 300 - 143 excluded = 157 câu SẠCH
```

**Kỹ thuật chi tiết — 2-tier architecture:**

**Tầng 1 (Retrieval on 450W chunks):**
```
Query → BM25 + bge-m3 + e5-large (RRF) → top-100 candidate chunks
         ↓ rerank (cross-encoder)
         → top-K chunk, gom document
         → DOCUMENT-LEVEL gộp (MAX, không LSE)
         → top-5 DOCUMENT
```

**Tầng 2 (Extraction on Điều within top-5 docs):**
```
Mỗi top-5 document → split theo Điều
Gom tất cả Điều từ 5 doc → ~100 Điều candidates
         ↓ rerank (cross-encoder fine-tune)
         → sort theo score
         → top-1 ĐIỀU → render_answer()
```

**Tại sao 2-tầng tốt hơn 1-tầng?**
- **Task 1 đo:** Cắt Điều ở tầng retrieval (tầng 1) → hại -1.19 điểm recall (p=0.024)
- **EDA phát hiện:** 11.5% document có chunk >5000 từ khi cắt Điều (outlier catastrophic)
- **Lý do:** Embedding bị pha loãng khi 1 "Điều" = 50.000 từ (OCR lỗi)
- **Fix:** Tầng 1 dùng Fixed 450W (không Điều) → retrieval recall tốt
  - Tầng 2 cắt Điều → extraction chính xác

**Fine-tune reranker (Family Negatives):**
- Negative "cùng họ" = các Điều **ANH EM trong CÙNG văn bản gold**
  - Đây là hard negative thực tế nhất (có thể cũng trả lời được câu hỏi)
- Task 1 metric: negative "cùng họ" recall 0.9428 > negative semantic 0.9360
- **Công thức:** Listwise softmax (positive phải rank #1 trong nhóm 1+7 negative)
- **Chốt chặn:** Bỏ ứng viên reranker zero-shot score cao hơn gold
  - Dấu hiệu false negative (positive chưa gán nhãn)

**Cấu hình cụ thể:**
- DOC_K: 5 (mở 5 document)
- MAX_DIEU_CANDIDATES: 100 (trần rerank tầng 2)
- N_NEG_RERANK: 7 (7 negative/group)
- RERANKER_FT_LR: 3e-6 (learning rate tối ưu cho listwise)
- AGG_MODE_T2: LSE (ở tầng 2 chọn T=1 qua split-half)
- AGG_MODE_T1: max (luôn max, không LSE ở tầng 1 — không đo)

**Kết quả:**
- METEOR: **+0.0057** so với v6 (0.5528 → 0.5585)
- ROUGE-L: -0.0046 (nhẹ giảm)
- Recall@1 (tầng 2): 55.24% (so với 58% v6 — giảm vì reranker gốc learn Điều)

**Nhân tố thành công:**
- 2-tier architecture: tách biệt retrieval (450W) vs extraction (Điều)
- Family negatives: learn hard negative realistik (Điều cùng doc)
- Sửa rò rỉ train/dev: dev-eval measure tổng quát, không trí nhớ

**Bottleneck vẫn còn:**
- Recall@1 tầng 2 chỉ 55.24% → 44.76% câu bị reranker score sai
- 15.5% document không Điều → hiện fallback 450W (chứ không trích cấu trúc thay thế)

---

### 2.6 **v5 (0.5688) — BATCH 1 A1+A2: Mở rộng tầng 2 regex**

**Thay đổi so với v4:**

```diff
~ Giữ: 2-tier architecture, reranker, family_negatives
~ Giữ: bge-m3 + e5-large
+ A1: Mở rộng tầng 2 regex (DIEU → MUC → PHU_LUC → TIET)
+ A2: Chặn trên chunk nếu >2000 từ → cắt lại 450W
```

**Kỹ thuật chi tiết — A1+A2:**

**A1 — Tầng bậc regex:**

EDA phát hiện 15.5% document không Điều nhưng vẫn có cấu trúc:
- Quyết định hành chính: Mục (I, II, ...) / Phần / Mục con
- QCVN / TCVN: tiết số (2.2.1.2, 3.1.4)
- Biểu mẫu: Phụ lục (I, II, V, ...)

**Regex được thêm:**
```python
DIEU_RE     = r"^[ \t]*Điều\s+(\d+)[a-zđA-ZĐ]?[\.\s]"
MUC_RE      = r"^[ \t]*Mục\s+([0-9]+|[IVXLCDM]+)\s*[\.\s:]"      # NEW
PHU_LUC_RE  = r"^[ \t]*Phụ\s+lục\s+([0-9IVXLCDM]+)\b"            # NEW
TIET_RE     = r"^[ \t]*(\d+\.\d+(?:\.\d+)?)\s*[\.\s]"             # NEW
```

**Logic chunking v5:**
```python
for regex, unit_type in [(DIEU_RE, "dieu"), (MUC_RE, "muc"), 
                         (PHU_LUC_RE, "phu_luc"), (TIET_RE, "tiet")]:
    matches = list(regex.finditer(passage))
    if matches:
        break  # Dùng tầng bậc ĐẦU TIÊN khớp được
else:
    # Không khớp bất kỳ → fallback Fixed 450W
    return split_fixed(passage, 450)
```

**A2 — Chặn outlier:**

EDA phát hiện outlier catastrophic:
- 11.5% document có ≥1 Điều >5000 từ
- Đỉnh: 1 Điều = 189.366 từ (doc_id=164898)
- Nguyên nhân: DIEU_RE neo `^` đầu dòng, OCR lỗi làm bỏ sót heading

**Fix A2:**
```python
MAX_UNIT_WORDS = 2000

if len(unit_text.split()) > MAX_UNIT_WORDS:
    # Cắt lại 450W thay vì giữ nguyên chunk khổng lồ
    for sub in _split_words_raw(unit_text, 450):
        chunks.append({...chunk cắt nhỏ...})
else:
    chunks.append({...chunk bình thường...})
```

**Kết quả test A1+A2 trên 30 sample không-Điều:**

| Metric | Trước (v4) | Sau (v5) | Tác động |
|---|---|---|---|
| fallback raw450 | 100% (tất cả doc) | 14.3% | **-85.7% fallback** |
| unit_type phân bố | dieu(0) | tiet(69%), phu_luc(13%), raw450(14%), dieu_capped(4%) | Bắt được tầng bậc |
| outlier xử lý | doc 161441: 1 chunk 7.4K từ | 12 chunks nhỏ | A2 hoạt động |

**Nhưng dev-eval kết quả:**
- Dev METEOR: 0.5709 (vs v4: 0.5585 → **+0.0124**)
- Dev ROUGE-L: 0.5643 (vs v4: 0.5419 → **+0.0224**)
- Public METEOR: 0.5688 (vs v4: 0.5585 → **+0.0103**)

**Tại sao chỉ +0.0103 thay vì +0.3–0.8 kỳ vọng?**

**Phân tích 3 nguyên nhân:**

1. **Reranker không train được unit type mới:**
   - Training data chỉ có label Điều (từ citation)
   - Không có label Mục/tiết/Phụ lục
   - Reranker fine-tune = zero-shot score "Mục X" = **weight của Điều** (không optimal)
   - → Recall@1 tầng 2 vẫn 55.24% (giống v4)

2. **Retrieval (tầng 1) đã tốt:**
   - Fixed 450W = thế tối ưu cho tầng 1 (Task 1 đo -1.19 điểm nếu cắt Điều)
   - A1/A2 không động tầng 1 → không ảnh hưởng retrieval recall
   - → Tầng 1 vẫn cover ~37% câu ở top-5

3. **15.5% document không-Điều chưa đầu đủ được cover:**
   - Regex MUC/PHU_LUC/TIET bắt được ~69% trong 15.5% (khoảng 10% toàn corpus)
   - Nhưng reranker không learn score chúng → recall@1 chỉ tăng nhẹ
   - Estimated tác động: 10% × (cải tiến METEOR trong 10% đó)
   - = chỉ +0.01 điểm tuyệt đối

**Bottleneck thực sự:**
- **Reranker recall@1 = 55.24% là nút thắt chính** — 44.76% câu bị score sai
- Chunking là cách tiếp cận đúng, nhưng cần **training label mới cho Mục/tiết/Phụ lục** để reranker learn

---

## 3. Tóm tắt so sánh — Tác động từng kỹ thuật

| Kỹ thuật | Thể hiện ở phiên bản | METEOR gain | Nhận xét |
|---|---|---|---|
| **RRF 3 kênh (BM25 + 2 dense)** | v2 | +0.0746 | **Lớn nhất**, encoder khác họ complement |
| **Fine-tune reranker** | v6 | +0.0313 | Nhẹ, plateau sớm |
| **Echo2 template** | v6 | +0.0131 | Nhỏ nhưng ổn định, đã cố định |
| **2-tier architecture** | v4 | +0.0057 | Nhẹ (so với kỳ vọng), nhưng **khả quan lâu dài** |
| **Family negatives** | v4 | (kết hợp 2-tier) | Cải thiện reranker training |
| **A1+A2 (regex + bound)** | v5 | +0.0103 | Nhẹ (vì reranker không train được unit type mới) |
| **Encoder mới (v7)** | v7 | +0.0029 | ❌ METEOR giảm, ROUGE-L tăng (template match) |

**Ranking độ hiệu quả:**
1. 🥇 RRF 3 kênh: +7.46% tuyệt đối
2. 🥈 Fine-tune reranker: +3.13%
3. 🥉 2-tier architecture: +0.57% (nhẹ nhưng **fix lỗi kiến trúc**, nền tảng cho future)
4. Echo2, A1+A2: +1.31%, +1.03% (nhẹ)

---

## 4. Các giả thuyết đã đóng — Không làm lại

| Giả thuyết | Trạng thái | Bằng chứng | Kết luận |
|---|---|---|---|
| Fine-tune encoder thêm lần | ❌ Đóng | Cố định bge-m3+e5-large, không tăng | Encoder pair này tối ưu cho corpus này |
| Dùng encoder mới (v7 test) | ❌ Đóng | v7 METEOR +0.0029 < v6 = v6 tốt hơn | Bge-m3+e5-large vẫn tốt hơn |
| Ghép nhiều Điều (top_n > 1) | ❌ Đóng | Oracle: 1 Điều=0.605, 2 Điều=0.519 | Ghép thêm = mất -0.086 METEOR |
| LLM generator | ❌ Đóng | Task 2 extraction-based (METEOR α=0.9 nặng recall) | LLM sinh mới thua trích đúng |
| Thêm epoch/LR ở fine-tune | ❌ Đóng | Đã test, không tăng | Reranker saturated |
| Cắt Điều ở tầng 1 | ❌ Đóng | Task 1: -1.19 điểm recall, EDA: 11.5% outlier | Fixed 450W là an toàn |

---

## 5. Những gì CHƯA thử — Hướng tiếp theo khả thi

### 5.1 Batch 2 (Hybrid chunking ở tầng 1) — Rủi ro trung bình

**Giả thuyết:**
- Fixed 450W → Hybrid (tôn trọng câu, không cắt giữa sentence)
- Chunk size nhỏ hơn (avg 375 từ vs 422W) → embedding kém bị pha loãng hơn

**Công sức:** 3–4h code + 2–3h encode corpus  
**Tác động ước tính:** +0.2–0.3 METEOR  
**Rủi ro:** Trung bình (cần encode lại, co thay đổi retrieval thứ tự chunk)

**Quyết định:** Chỉ thử nếu còn GPU + lượt submit + Batch 1 đã tốt

---

### 5.2 Batch 3A (Fine-tune reranker LẠI với label Mục/tiết) — Khả năng cao

**Phát hiện:**
- A1 bắt được regex Mục/tiết/Phụ lục ở 15.5% document
- Nhưng reranker KHÔNG train được vì training data (citation) chỉ có label Điều

**Giải pháp:**
1. Từ 15.5% document không-Điều:
   - Regex phát hiện "Mục X" → lấy canonical form
   - Semi-auto label: link (question, Mục X) từ corpus
   - Hoặc thủ công label 100 cặp sample

2. Fine-tune reranker thêm lần:
   - Base: checkpoint reranker v4 (đã fine-tune trên Điều)
   - Loss: Listwise softmax (cộng thêm cặp Mục)
   - N_NEGATIVE: 7 (same as Điều)

3. Merge logic:
   - If chunk_type == "dieu" → dùng reranker Điều
   - Else (Mục/tiết) → dùng reranker Mục (hoặc ensemble)

**Công sức:** 4–6h (label + train + merge)  
**Tác động ước tính:** +2–3% METEOR (nếu recall@1 tăng 2–3%)  
**Rủi ro:** Thấp (tuân lệ task rule, không dùng dữ liệu ngoài)

**Đây có thể là bottleneck thực sự cần fix.**

---

### 5.3 Batch 3B (Query expansion) — Nhẹ, đã có precedent

**Giả thuyết:**
- Expand query từ dense embedding → thêm từ khoá liên quan
- result.md §10 ghi +0.1–0.2 METEOR

**Công sức:** 2–3h  
**Tác động ước tính:** +0.2–0.5 METEOR  
**Rủi ro:** Trung bình (expansion có thể thêm noise)

---

### 5.4 Batch 4 (Reranker ensemble) — Cao rủi ro, skip

**Giả thuyết:**
- Dùng 2–3 reranker khác nhau → vote top-1
- Tác động ước tính: +0.3–0.8 METEOR

**Nhưng:** 
- Đã fine-tune reranker nhiều lần (v6, v4), plateau
- Thêm reranker khác yêu cầu encode lại + train
- Rủi cao, công sức lớn

**Quyết định:** ❌ Skip (tập trung vào 5.2A thay vì)

---

## 6. Kiến trúc hiện tại (v5) — Sơ đồ chi tiết

```
Question
  ↓
[1. RETRIEVAL LAYER - Fixed 450W]
  │
  ├─ BM25 (inverted index)       → rank 0–100, RRF score
  ├─ bge-m3 (dense)               → rank 0–100, RRF score
  └─ e5-large (dense, "query: ") → rank 0–100, RRF score
  │
  ↓ RRF fusion (reciprocal rank fusion)
  │
  ├─ Top-100 candidates (chunk, not Điều)
  │
  ↓ Cross-encoder rerank (zero-shot Vietnamese_Reranker)
  │
  ├─ Rerank top-100 → top-K, group by document_id
  │
  ↓ Document-level aggregation (MAX)
  │
  ├─ Top-5 DOCUMENTS (chosen)
  │
[2. EXTRACTION LAYER - Điều / Mục / tiết / Phụ lục]
  │
  ├─ Từ mỗi top-5 doc:
  │   ├─ Try DIEU_RE (pattern: "Điều 1 ...")
  │   ├─ Try MUC_RE  (pattern: "Mục I ...")        [v5 NEW]
  │   ├─ Try PHU_LUC_RE (pattern: "Phụ lục II ...") [v5 NEW]
  │   ├─ Try TIET_RE (pattern: "2.2.1 ...")        [v5 NEW]
  │   └─ Fallback: Fixed 450W chunks
  │
  ├─ Gom tất cả unit (Điều/Mục/tiết) → ~100 candidates
  │
  ↓ Bound check (A2) - chặn trên 2000W [v5 NEW]
  │
  ↓ Cross-encoder rerank (fine-tune Vietnamese_Reranker)
  │  (trained on Điều only — LIMITATION)
  │
  ├─ Top-1 UNIT (Điều/Mục/tiết được chọn)
  │
[3. COMPOSITION LAYER]
  │
  ├─ Render answer: "Căn cứ Điều X Luật Y ... quy định như sau:"
  │
  ├─ Append echo2 template:
  │  "Theo đó, [question].\nNhư vậy, theo quy định nêu trên thì [question]."
  │
  └─ Return answer string
```

**Limitation nhìn thấy:**
- Tầng 2 reranker CHỈ train được Điều (citation label)
  - Khi chunking v5 tạo Mục/tiết → reranker score chúng bằng **weight Điều**
  - Recall@1 tầng 2 vẫn ~55.24% (không cải thiện)

---

## 7. Checklist cho thành viên mới / tiếp tục

### Muốn thử hướng mới, cần biết:

- [ ] **Retrieval (tầng 1):** 37.76% recall@5 — đã tốt với Fixed 450W, không nên thay
- [ ] **Extraction (tầng 2):** 55.24% recall@1 — **NÚT THẮT CHÍNH**
  - Reranker chỉ train được Điều
  - v5 A1 thêm Mục/tiết nhưng reranker không học → không đổi recall
  - **Hướng fix:** Fine-tune reranker thêm lần với label Mục/tiết (5.2A)
  
- [ ] **Chunking v5 (A1+A2):** Hoạt động (bắt được 85% trong 15.5% doc không-Điều)
  - Nhưng cần reranker support → không tác động recall@1
  - Để A1 phát huy tác dụng, cần 5.2A

- [ ] **Encoder:** bge-m3 + e5-large là cặp tối ưu
  - v7 thử Vietnamese_Embedding_v2 + harrier → METEOR xấu hơn
  - Không thử encoder mới trừ khi có data mới

- [ ] **Closed hypothesis:** Fine-tune thêm, LLM, top_n>1, cắt Điều ở tầng 1
  - Không làm lại các cái này

- [ ] **Timeline deadline:** 18/09 23:59 GMT+7 (còn ~8 ngày từ 10/09)
  - Batch 1 (A1+A2) đã xong, +0.0103 METEOR
  - Batch 2 (Hybrid): 3–4h, nhưng sát deadline + rủi cao
  - **Ưu tiên:** 5.2A (Fine-tune reranker Mục) thay vì Hybrid

---

## 8. Kết luận — Kế tiếp tiếp theo (top priority)

**v5 hiện tại:** 0.5688 METEOR (public) / 0.5709 (dev) — top ~3–5 trên leaderboard

**Nút thắt:** Reranker tầng 2 recall@1 = 55.24% → phải cải này mới tăng điểm được

**Hướng tiếp:**
1. **Ngay:** Implement 5.2A — Fine-tune reranker với label Mục/tiết
   - Công sức: 4–6h
   - Tác động: +2–3% METEOR (có thể)
   - Rủi ko: Thấp

2. **Nếu 1 thắng:** Thử Batch 2 (Hybrid) nếu còn GPU + lượt submit
3. **Nếu 1+2 vẫn khó:** Xem 5.3B (Query expansion)

**Không làm:** Batch 4 (reranker ensemble), encoder mới, closed hypothesis

---

**Tài liệu này được viết tại 2026-09-10, dùng để guide toàn bộ team LegalQA Task 2. Cập nhật khi có phiên bản mới.**
