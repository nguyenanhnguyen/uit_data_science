# Phân tích EDA — Tính chất Data & Chiến thuật Chunking LegalQA

`2026-09-09` · nguồn: `eda_server_output/` (chạy trên 7.000 QA pairs + 8.474 document,
toàn bộ corpus, không sample) · script sinh ra: `eda_all_in_one.py`

---

## 1. Tóm tắt số liệu cốt lõi

| Đối tượng | Tổng | Có Điều | Không Điều |
|---|---:|---:|---:|
| QA pairs (`train.json`) | 7.000 | 6.174 (88,2%) | 826 (11,8%) |
| Document (`selected-contexts/`) | 8.474 | 7.161 (84,5%) | 1.313 (15,5%) |

| Answer (`train.json`) | Giá trị |
|---|---:|
| Độ dài trung bình | 347,4 từ |
| Độ dài median | 312,0 từ |
| Số câu trung bình | 9,66 |
| p10 / p50 / p90 / p99 | 147 / 312 / 577 / 1.028 từ |

| Document (`selected-contexts/`) | Giá trị |
|---|---:|
| Độ dài trung bình | 8.562 từ |
| Độ dài median | 4.830,5 từ |
| p10 / p50 / p90 / p99 | 1.395 / 4.831 / 18.440 / 59.251 từ |
| Số Điều/doc (khi có cấu trúc) | trung bình 21,9 · median 15 · max **838** |

**Nhận xét đầu tiên:** corpus rất lệch phải (long-tail) — 10% document dài trên 18.440 từ,
1% dài trên 59.251 từ. Đây là gốc rễ của mọi vấn đề chunking bên dưới.

---

## 2. Phân bố số Điều cần trích trong answer

| Số Điều trong answer | Số câu | Tỷ lệ |
|---:|---:|---:|
| 1 | 2.954 | 47,8% |
| 2 | 1.504 | 24,4% |
| 3 | 782 | 12,7% |
| 4 | 413 | 6,7% |
| 5 | 204 | 3,3% |
| ≥6 | 317 | 5,1% |

**72,2% câu chỉ cần 1–2 Điều.** Khớp hoàn toàn với oracle đã đo trong `result.md` §3
(1 Điều → METEOR 0,605; 2 Điều ghép → 0,519 — **giảm**). Số liệu này củng cố thêm quyết
định `top_n=1` hiện tại trong `compose()`: đa số câu hỏi vốn dĩ chỉ cần đúng 1 Điều, ghép
thêm là pha loãng câu trả lời đúng bằng nội dung không liên quan.

⚠️ **Bug phát hiện trong chính script EDA**: `KHOANG_RE = r"\b\((\d+)\)\s"` chỉ khớp
format `"(1) "` — nhưng văn bản luật Việt Nam dùng `"1. "` không ngoặc. Kết quả: chỉ
**1/7.000** câu được ghi nhận "has_khoang". Đây là artifact của regex sai, **không phải**
đặc điểm thật của data — không dùng con số này để kết luận gì về cấu trúc khoản.

---

## 3. Loại văn bản chính trong answer

| Loại | Số câu | Tỷ lệ |
|---|---:|---:|
| Nghị định | 1.802 | 25,7% |
| Luật | 1.632 | 23,3% |
| Thông tư | 1.621 | 23,2% |
| Quyết định | 1.080 | 15,4% |
| Khác (không khớp 4 loại trên) | 865 | 12,4% |

Bốn loại văn bản chính (Nghị định/Luật/Thông tư/Quyết định) chiếm 87,6% — đều **có khả
năng** cấu trúc theo Điều. Nhóm "Khác" 12,4% là nơi cần soi kỹ nhất (xem §4).

---

## 4. Edge case: answer/document KHÔNG có Điều — soi bằng sample thật

Lấy mẫu thật từ `eda_samples.json` (không suy diễn):

| Câu hỏi | Cấu trúc trích dẫn thật |
|---|---|
| "Hồ sơ thành lập trường cao đẳng tư thục…" | `điểm 20.3 khoản 20 tiểu mục I Mục B Phần II` (Quyết định 445/QĐ-LĐTBXH) |
| "Phẫu thuật kết hợp xương gãy xương gót…" | `Mục III Quy trình kỹ thuật` (Quyết định 5728/QĐ-BYT) |
| "Diện tích tối đa bảng quảng cáo…" | `mục 2.2.1.2` (QCVN 17:2018, núp dưới Thông tư 04/2018/TT-BXD) |
| "Chất lượng đậu bắp quả tươi hạng II…" | `tiết 2.2.3 tiểu mục 2.2 Mục 2` (TCVN 12995:2020) |
| "Mẫu sổ thống kê tai nạn lao động…" | `Phụ lục I` (Thông tư 13/2020/TT-BLĐTBXH) |

Document tương ứng — "Nghị quyết 33/NQ-CP-2023" (7.461 từ) — không có "Điều" nào, chỉ có
mục La Mã/số.

**Kết luận quan trọng: 11,8–15,5% "không Điều" KHÔNG đồng nghĩa với "không cấu trúc".**
Đây là văn bản dùng hệ phân cấp khác:

```
Nghị định/Luật/Thông tư (dạng chính)  →  Điều → Khoản → Điểm
Quyết định hành chính, Quy trình kỹ thuật  →  Phần → Mục → tiểu mục → tiết
Quy chuẩn/Tiêu chuẩn (QCVN/TCVN)  →  Mục → tiểu mục → tiết (số thập phân kiểu 2.2.1.2)
Biểu mẫu, Phụ lục  →  không có cấu trúc phân cấp, chỉ có mã "Phụ lục I/II/III"
```

Tức là "fallback" đúng nghĩa không phải là "cắt 450 từ cho xong", mà là **một bộ regex
thứ hai bắt `Mục/Phần/tiểu mục/tiết/Phụ lục`**, chạy song song với `DIEU_RE`.

---

## 5. So sánh 4 chiến lược chunking (đo trên toàn bộ 8.474 document)

| Strategy | Chunks/doc (avg) | Chunk length avg | Stdev | **Max chunk (từ toàn corpus)** |
|---|---:|---:|---:|---:|
| **Fixed 450W** (hiện tại) | 19,5 | 422,1 | 71,4 | **450** (bị chặn cứng) |
| Sentences | 20,9 | 410,1 | 95,2 | **11.657** |
| Articles | 20,8 | 723,4 | **1.011,8** | **189.366** ⚠️ |
| Hybrid (450W + né cắt câu) | 21,3 | 374,9 | 106,4 | **450** (bị chặn cứng) |

### 5.1 Phát hiện quan trọng nhất: Articles strategy có outlier thảm hoạ

- **973 document (11,5%)** có ít nhất 1 chunk-Điều dài **>5.000 từ**
- **197 document (2,3%)** có ít nhất 1 chunk dài **>20.000 từ**
- Outlier cực đoan nhất: doc 164898 — chỉ tách được **4 "Điều"** cho toàn bộ 189.366 từ

**Nguyên nhân**: `DIEU_HEADING_RE` neo `^\s*Điều\s+\d+` ở đầu dòng. Với văn bản có OCR/format
lỗi (dòng bị nối, không xuống dòng đúng chỗ), phần lớn "Điều X." nằm giữa dòng thay vì đầu
dòng → regex bỏ sót → khoảng cách giữa 2 lần khớp kéo dài hàng chục nghìn từ → "chunk" đó
trở thành gần như toàn bộ văn bản.

**Đây chính là cơ chế đứng sau con số `result.md §3`: "chunk theo Điều ở tầng retrieval làm
document recall tệ đi 1,19 điểm"** — không phải vì chunk theo Điều sai về nguyên tắc, mà vì
**một tỷ lệ nhỏ nhưng không hiếm (11,5%) document tạo ra chunk khổng lồ**, làm hỏng biểu diễn
dense/BM25 của toàn bộ phần văn bản còn lại trong chunk đó (bị pha loãng, embedding không
còn phản ánh đúng nội dung cục bộ).

### 5.2 Sentences strategy cũng có vấn đề tương tự (nhẹ hơn)

Max chunk 11.657 từ — văn bản có đoạn dùng dấu `;` hoặc xuống dòng thay vì `.` để ngăn cách
ý, khiến bộ tách câu bằng `[.!?]` gộp nhầm một khối lớn thành "một câu".

### 5.3 Fixed 450W và Hybrid là hai chiến lược DUY NHẤT có chặn trên cứng

Cả hai đều bị giới hạn max=450 vì logic chia theo **số từ**, không phụ thuộc cấu trúc nội
dung — nên miễn nhiễm với lỗi định dạng của văn bản nguồn. Đây là lý do chúng "an toàn"
hơn về mặt retrieval dù có vẻ "thô" hơn về ngữ nghĩa.

### 5.4 Vì sao Hybrid vẫn đáng dùng dù Fixed đã an toàn

Hybrid giữ cùng chặn trên (450) như Fixed nhưng:
- Chunk length avg thấp hơn (374,9 vs 422,1 từ) — cắt sớm hơn khi gặp dấu câu ở 70%+ chunk
- Không cắt ngang giữa câu trong đa số trường hợp (trade-off: nhiều chunk hơn — 21,3 vs 19,5)
- Vẫn hoạt động đồng nhất trên **100% document**, không cần biết trước document có Điều
  hay không — khác với Articles phải fallback có điều kiện

---

## 6. Đề xuất chiến thuật chunking

### 6.1 KHÔNG đổi chunking ở tầng 1 (retrieval)

Giữ nguyên **Fixed 450W** hoặc nâng cấp sang **Hybrid** cho tầng 1. Dữ liệu ở §5.1 xác nhận
lại — bằng bằng chứng cụ thể hơn `result.md §3` — rằng **bất kỳ chunking nào phụ thuộc cấu
trúc nội dung (Điều, câu) đều có đuôi phân bố dài (long-tail outlier)** do văn bản nguồn có
lỗi định dạng không đồng nhất trên 8.474 file. Tầng retrieval cần chunk kích thước **ổn định
và có chặn trên cứng** để embedding không bị pha loãng.

**Khuyến nghị cụ thể**: nếu muốn tối ưu thêm ở tầng 1, chuyển từ Fixed 450W sang **Hybrid**
— cùng an toàn (max=450) nhưng tôn trọng ranh giới câu tốt hơn, có thể cải thiện nhẹ chất
lượng embedding vì chunk không bị cắt ngang ý. Đây là thay đổi **rẻ, rủi ro thấp, đáng thử
nghiệm A/B trên `eval/score_dev_v2.py`** trước khi quyết định.

### 6.2 CẢI TIẾN tầng 2 (extraction) — đây là nơi có ROI thật

Tầng 2 hiện tại (`split_dieu()` trong `legalqa_local.py`) chỉ bắt được cấu trúc Điều, bỏ
sót đúng nhóm 15,5% document dùng hệ phân cấp khác (§4). Đề xuất **mở rộng regex tầng 2
theo tầng bậc, thử lần lượt**:

```
1. Thử DIEU_RE trước (đã có) — bắt 84,5% document
2. Nếu rỗng → thử MUC_RE: r"(?m)^\s*Mục\s+(\d+|[IVXLCDM]+)\s*[.．:]"
3. Nếu vẫn rỗng → thử PHU_LUC_RE: r"(?m)^\s*Phụ\s+lục\s+([IVXLCDM\d]+)"
4. Nếu vẫn rỗng → fallback cắt theo khoản/tiết dạng số thập phân:
   r"(?m)^\s*(\d+\.\d+(\.\d+)?)\s"  (bắt "2.2.1.2", "2.2.3" như sample TCVN ở §4)
5. Cuối cùng mới fallback Fixed 450W (như hiện tại)
```

**Lý do đặt đúng thứ tự này**: mẫu thật ở §4 cho thấy văn bản không-Điều vẫn có cấu trúc
phân cấp rõ ràng (Mục → tiểu mục → tiết), chỉ là tên gọi khác. Bắt đúng đơn vị nhỏ nhất
tương đương "Điều" của loại văn bản đó sẽ tái lập được lợi ích mà `result.md §3` đã đo cho
Điều (0,605 METEOR) trên nhóm 15,5% hiện đang bị bỏ ngỏ.

**Ước tính tác động**: nếu 826 câu (11,8% answer không Điều) đang được trả lời bằng chunk
450 từ thô thay vì đơn vị đúng (Mục/tiết), và nếu METEOR của nhóm này thấp hơn nhóm có
Điều theo cùng tỷ lệ đã đo ở Task 1 (0,605 vs baseline thấp hơn), cải thiện đúng nhóm này
có thể mang lại **+0,5 đến +1,0 điểm METEOR toàn cục** — nhưng con số này CẦN đo thật trên
dev-eval trước khi tin, theo đúng nguyên tắc "Đo trước khi tin" của CLAUDE.md §6.

### 6.3 KHÔNG dùng Sentences hoặc Articles thô cho bất kỳ tầng nào

Cả hai đều có outlier catastrophic (max 11.657 và 189.366 từ) do phụ thuộc dấu câu/định
dạng của nguồn — không đáng tin cậy trên quy mô 8.474 document với chất lượng OCR/format
không đồng đều. Nếu dùng Articles ở tầng 2 (đã đúng kiến trúc hiện tại — chỉ cắt trong
K=5 document đã lọc), cần thêm **chặn trên cứng** (vd: nếu 1 "Điều" bị tách ra dài hơn
2.000 từ, coi là lỗi tách, fallback cắt 450 từ cho riêng "Điều" đó) để tránh outlier như
doc 164898 lọt vào top-5 document được chọn.

### 6.4 Thứ tự triển khai đề xuất (từ rẻ → đắt)

| # | Việc làm | Công sức | Cần encode lại? | Rủi ro |
|---|---|---|---|---|
| 1 | Thêm chặn trên cứng cho `split_dieu()` ở tầng 2 (fallback nếu Điều >2.000 từ) | 30 phút | Không | Rất thấp |
| 2 | Mở rộng tầng 2: thêm `MUC_RE`, `PHU_LUC_RE`, fallback thập phân theo §6.2 | 2–3 giờ | Không | Thấp |
| 3 | Đo dev-eval (`score_dev_v2.py`) so sánh có/không mở rộng tầng 2 | 30 phút | Không | — |
| 4 | Thử Hybrid thay Fixed 450W ở tầng 1 | 1 giờ code + encode lại corpus | **Có** (encode lại dense) | Trung bình (đổi thứ tự chunk) |
| 5 | Đo dev-eval so sánh Fixed vs Hybrid ở tầng 1 | 30 phút | Không | — |

**Khuyến nghị làm #1–#3 trước** (không cần encode lại, rủi ro thấp nhất, đúng nhóm 15,5%
document đang bị bỏ ngỏ). Chỉ làm #4–#5 nếu #1–#3 đã hết dư địa và còn lượt submit để thử
nghiệm (nhắc lại ràng buộc `rule.md §3`: public chỉ còn đến 18/09, 10 lượt/ngày).

---

## 7. Giới hạn của phân tích này

- Regex `KHOANG_RE` trong script EDA bị lỗi (§2) — không dùng số liệu "has_khoang" để kết luận.
- Phân tích document dựa trên `count_words()` bằng `str.split()` trần, giống hệt cách BTC
  tokenize khi chấm METEOR (`scoring/legalqa/scoring.py`) — số liệu độ dài có thể so sánh
  trực tiếp với ngưỡng chấm điểm thật.
- Chưa đo METEOR thật của nhóm "không Điều" (826 câu) so với nhóm "có Điều" — đây là bước
  tiếp theo bắt buộc trước khi tin ước tính tác động ở §6.2 (theo CLAUDE.md §6 "Đo trước
  khi tin").
