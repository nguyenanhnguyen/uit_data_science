# Tổng hợp EDA + Đánh giá Khả thi Cải tiến — LegalQA

**Ngày:** 2026-09-09 · **Context:** EDA chạy trên 7.000 QA + 8.474 doc, v4 baseline 0.5585 METEOR · **Mục tiêu:** định hướng thử nghiệm tiếp theo

---

## 1. Quá trình suy luận từ EDA (tóm tắt điểm then chốt)

### 1.1 Tầng 1 (Retrieval): Fixed 450W là lựa chọn tối ưu cho lúc này

**Dữ liệu:**
- Articles (cắt theo Điều) có **11,5% document** tạo chunk >5.000 từ, **2,3%** >20.000 từ
- Outlier cực: 189.366 từ (doc 164898) — gần toàn bộ văn bản thành 1 "chunk"
- Nguyên nhân: `DIEU_RE` chỉ khớp `^\s*Điều` ở đầu dòng, lỗi OCR/format làm bỏ sót
- Hệ quả: embedding bị pha loãng, recall giảm (thỏa số liệu `result.md §3`: -1,19 điểm)

**Kết luận:** Tầng 1 phải dùng **Fixed 450W hoặc Hybrid** (chặn cứng max=450) để tránh outlier.
- Fixed 450W: an toàn 100%, không cần thay đổi
- Hybrid: an toàn 100%, tôn trọng câu tốt hơn (chunk avg 374,9 vs Fixed 422,1) → **đáng thử A/B nhẹ, rủi ro thấp**

**Khuyến nghị:** GIỮ nguyên Fixed 450W ở tầng 1 (không cần encode lại), hoặc thử Hybrid nếu còn GPU + lượt submit.

### 1.2 Tầng 2 (Extraction): Có **15,5% document "không Điều" thực ra vẫn có cấu trúc**

**Dữ liệu từ sample thật (§4 report):**
- Câu "Hồ sơ thành lập trường" → cấu trúc `điểm 20.3 khoản 20 tiểu mục I Mục B Phần II`
- Câu "Chất lượng đậu bắp" → cấu trúc `tiết 2.2.3 tiểu mục 2.2 Mục 2` (TCVN tiêu chuẩn)
- Câu "Mẫu sổ thống kê" → cấu trúc `Phụ lục I` (biểu mẫu)

**Hiện tại làm gì:** tầng 2 chỉ cắt theo `DIEU_RE` → rỗng → fallback Fixed 450W → trả lời bằng chunk thô 450 từ.

**Tại sao quan trọng:** Nhóm 826 câu (11,8%) không Điều hiện đang **ăn penalty** vì không được trích đúng đơn vị ngữ nghĩa (như 72,2% câu có Điều được trích 1–2 Điều → METEOR 0,605). Nếu cải tiến tầng 2 bắt được cấu trúc thay thế (Mục/Phần/tiết), có thể **tái lập lợi ích của extraction** cho nhóm này.

**Ước tính tác động:** 
- Nhóm 826 câu (11,8%) không Điều hiện có METEOR ~? (chưa đo)
- Nếu cải tiến tầng 2 bắt được cấu trúc → METEOR của nhóm này tăng từ ~0,50 lên ~0,57 (ước lượng lỏng lẻo dựa trên tỉ lệ extraction lợi ích từ nhóm có Điều)
- Tác động toàn cục: `(0,57 - 0,50) × 0,118 ≈ +0,008` trên METEOR 0–1 scale = **+0,8 điểm trên scale 0–1000 = +8 điểm bách phân nếu có bao nhiêu điểm**
  - Vd: từ 0,5585 → 0,5665 nếu estimate đúng

**Khuyến nghị:** CÓ GIÁ TRỊ thử, nhưng cần đo thật trên dev-eval trước khi tin.

### 1.3 Tầng 3 (Compose): Echo2 đã tối ưu, không cần đổi

Từ `legalqa_local.py` docstring:
- Echo2 (+0,0131 ± 0,0010 điểm so với Echo, trên 501 câu dev) → lợi ích nhỏ nhưng ổn định
- Lặp 2 lần câu hỏi là giới hạn (lần 4 chỉ +0,5 điểm, nhìn đã "bẩn")
- Đã cố định ở v4 baseline → **không cần động**

---

## 2. Đánh giá tính khả thi — Chunking vs Các hướng khác

### Scenario A: Cải tiến tầng 2 (mở rộng regex)

**Việc làm:**
```
1. Thêm MUC_RE: r"(?m)^\s*Mục\s+(\d+|[IVXLCDM]+)\s*[.．:]"
2. Thêm PHU_LUC_RE: r"(?m)^\s*Phụ\s+lục\s+([IVXLCDM\d]+)"
3. Thêm fallback tiết (TCVN): r"(?m)^\s*(\d+\.\d+(\.\d+)?)\s"
4. Nếu tất cả rỗng → fallback Fixed 450W (như hiện tại)
```

**Công sức:** 2–3 giờ (chỉ code, không encode lại corpus)

**Rủi ro:** Rất thấp — thêm heuristic cho nhóm 15,5% đang bỏ ngỏ, không chạm vào 84,5% nhóm Điều

**Tác động ước tính:**
- Best case: +0,8 điểm (nếu bắt được cấu trúc thay thế đủ tốt cho nhóm 826 câu)
- Likely case: +0,3–0,5 điểm (một số cấu trúc bị bỏ sót do regex không hoàn toàn chính xác)
- Worst case: 0 (nếu regex không khớp hoặc cấu trúc Mục/tiết không có mối liên hệ rõ ràng với METEOR)

**Khuyến nghị:** **NÊN làm, ưu tiên cao** — rủi ro thấp, rủi ro có tiền lệ (extraction + finetuned reranker đã đạt 0,5528 ở v6).

---

### Scenario B: Thử Hybrid thay Fixed 450W ở tầng 1

**Việc làm:** Thay `split_fixed()` bằng `split_hybrid()` ở tầng retrieval

**Công sức:** 1 giờ code + **phải encode lại corpus ~2–3 giờ GPU** (RTX 2080 Ti)

**Rủi ro:** Trung bình — thay đổi thứ tự chunk sẽ làm thay đổi embedding dense, có thể ảnh hưởng retrieval

**Tác động ước tính:**
- Best case: +0,2–0,3 điểm (tôn trọng câu tốt hơn, embedding kém bị pha loãng hơn)
- Likely case: ±0 (thứ tự chunk khác nhưng kích thước tương tự, dense không thay đổi nhiều)
- Worst case: -0,1–0,2 điểm (nếu 21,3 chunk/doc vs 19,5 hiện tại làm retrieval kém hơn do chunk nhỏ hơn)

**Khuyến nghị:** **CÓ THỂ thử nếu còn GPU + lượt submit**, nhưng ưu tiên thấp hơn Scenario A vì:
1. Công sức cao hơn (cần encode lại)
2. Tác động không chắc chắn (có thể 0)
3. Rủi ro thay đổi retrieval (có tiền lệ âm từ Task 1 — cắt theo Điều làm hại -1,19 điểm)

---

### Scenario C: Các hướng **KHÔNG nên** làm (closed based on result.md)

Từ `legalqa/result.md`, những thứ đã đóng (không tăng điểm hoặc hại):

1. **Fine-tune encoder/reranker thêm nữa:** đã cố định `e5-large + bge-m3 + Vietnamese_Reranker` → thêm epoch/LR/negatives không tăng điểm
2. **Dùng LLM generator:** Task 2 là extraction-based (METEOR α=0.9 nặng recall) → LLM sinh mới câu sẽ thua trích xuất từ Điều/Mục chính xác
3. **Ghép nhiều Điều (top_n > 1):** Oracle đo 2 Điều ghép = 0,519 < 1 Điều = 0,605 → penalty cộng dồn
4. **Đổi template câu dẫn:** "Căn cứ" vs "Theo" là tối ưu, echo2 là cố định (§1.3)

---

### Scenario D: Cách làm khác NGOÀI chunking (để reference)

Nếu Scenario A/B không mang lại đủ tác động, có thể xem xét:

| Hướng | Công sức | Rủi ro | Tác động ước tính | Trạng thái trong result.md |
|---|---|---|---|---|
| **Query expansion** (tăng thêm từ khoá) | 2–3h | Trung | +0,2–0,5 | §10: đo được +0,1–0,2, có hiệu lực nhẹ |
| **Reranker ensemble** (thêm reranker khác) | 4–6h + encode | Cao | +0,3–0,8 | đã thử cross-encoder + Vietnamese_Reranker, thêm nữa có giảm thiểu rủi ro? |
| **Post-processing: entity linking** | 3–4h | Trung | +0,1–0,3 | chưa thử, xem §research_paper xem SOTA có ghi |
| **Cắt câu trước khi trích** | 1h | Thấp | +0,05–0,15 | chưa thử, có thể tế nhị |

---

## 3. Kết luận: EDA có đủ để quyết định chưa?

### 3.1 Những gì đã biết (từ EDA + baseline v4)

✅ **Đủ để quyết định làm Scenario A (mở rộng tầng 2)**
- Biết 15,5% document không Điều nhưng có cấu trúc khác (Mục/Phần/tiết/Phụ lục)
- Biết nhóm này hiện đang bỏ ngỏ (fallback Fixed 450W)
- Biết extraction từ cấu trúc tương tự (Điều) cho METEOR 0,605 trong oracle
- Rủi ro thấp (chỉ thêm regex cho nhóm bỏ ngỏ, không chạm nhóm Điều)

⚠️ **Chưa đủ để confident dự đoán Scenario B (Hybrid)**
- Biết Hybrid có lợi thế "tôn trọng câu" nhưng không biết lợi ích thực tế trên retrieval
- Lịch sử: Task 1 cắt theo Điều ở tầng retrieval → hại -1,19 điểm (nhiều rủi ro tương tự)
- Cần: đo A/B `eval/score_dev_v2.py` Fixed vs Hybrid để tin

### 3.2 Những gì chưa biết (cần đo thêm)

1. **METEOR thật của nhóm 826 câu không Điều** (dùng `eval/score_dev_v2.py` với filter)
   - Nếu đã ~0,56 (gần nhóm Điều), lợi ích cải tiến nhỏ
   - Nếu ~0,48 (thấp), lợi ích lớn

2. **Cấu trúc Mục/Phần/tiết trong sample từ corpus thật**
   - Regex `MUC_RE` / `PHU_LUC_RE` có khớp được không?
   - Có fallback case không — cấu trúc lạ không khớp 3 regex trên?

3. **Recall@1 của mỗi strategy** ở tầng 2 (nếu có thời gian)
   - Giữa "cắt Điều" vs "cắt Mục" vs "fallback 450W", cái nào cover được >70% câu?

---

## 4. Khuyến nghị: Lộ trình triển khai

### Nếu còn **<5 ngày + <30 lượt submit**:

**Tuần tự:**
1. **(Ngay)** Chạy Scenario A (mở rộng tầng 2):
   - Code: 2h
   - Đo dev-eval: 30 phút
   - Quyết định: nộp hay không (dựa kết quả dev)

2. **(Nếu A tốt)** Thử Scenario B (Hybrid) nếu còn GPU + lượt submit
   - Code + encode: 3–4h
   - Đo dev-eval: 30 phút
   - Quyết định nộp

3. **(Nếu A+B vẫn không vừa điểm)** Xem Scenario D (query expansion / reranker ensemble)

### Nếu còn **<2 ngày + <5 lượt submit** (tình cảnh hiện tại):

**Ưu tiên:**
1. ✅ Scenario A (mở rộng tầng 2) — công sức nhỏ, rủi ro rất thấp, tác động ước tính +0,3–0,8
2. ❌ Scenario B (Hybrid) — quá sát deadline, encode lại corpus mất 2–3h, rủi ro cao nếu không đo kỹ A/B

---

## 5. Chunking có phải cách duy nhất?

**Không.** Nhưng **là cách rẻ + hiệu quả nhất lúc này:**

| Hướng | Công sức | Rủi ro | Tác động | Khả thi ngay |
|---|---|---|---|---|
| **Chunking (A)** | 2–3h | Rất thấp | +0,3–0,8 | ✅ Có |
| Chunking (B) | 4h + encode | Trung | +0,2–0,3 | ⚠️ Sát deadline |
| Query expansion | 2–3h | Trung | +0,2–0,5 | ✅ Có (nhưng chưa thử) |
| Reranker ensemble | 4–6h | Cao | +0,3–0,8 | ❌ Quá công sức |

**Khuyến nghị cuối:** Làm **Scenario A (mở rộng tầng 2)** ngay hôm nay/mai. Nó là cách tối ưu được chứng minh bằng dữ liệu (15,5% document có cấu trúc nhưng bỏ ngỏ), rủi ro rất thấp, công sức nhỏ, tác động có tiềm năng +0,3–0,8 trên public METEOR.

---

## 6. Checklist triển khai Scenario A

- [ ] Viết `MUC_RE`, `PHU_LUC_RE`, fallback tiết (regex test trước trên 5–10 sample từ `eda_samples.json`)
- [ ] Thêm chặn trên cứng cho tầng 2 nếu 1 "chunk" >2.000 từ → fallback Fixed 450W
- [ ] Chạy `eval/score_dev_v2.py` so sánh (cùng pipeline, chỉ đổi tầng 2)
- [ ] Nếu +0,1 trở lên → nộp public test (dùng 1 lượt)
- [ ] Nếu ≤0 → roll back, xem Scenario D

**Thời gian ước tính:** 4–5 giờ tổng (code + đo + quyết định)
