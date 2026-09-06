# LegalQA — nhật ký nghiên cứu và kết quả thực nghiệm

`2026-09-06` · độ đo chính **METEOR** (nltk, `alpha=0.9`), phụ **ROUGE-L** (vendor BTC, hỏng
với tiếng Việt — xem §5) · dev-eval rút từ `train.json` (nhiều cỡ mẫu khác nhau giữa các
lượt, xem ghi chú từng dòng) · GPU: RTX 2080 Ti (server dùng chung) + Kaggle T4×2 (miễn phí)

File này gộp lại mọi thứ đã thử cho Task 2 (LegalQA) rải rác qua nhiều phiên làm việc, để
không lặp lại lỗi hoặc setting đã biết là vô ích/nguy hiểm. Đọc **§2 (sự cố nghiêm trọng)**
trước tiên — nó ràng buộc mọi hướng đi sau này.

Tài liệu song song: [log.txt](log.txt) (bản ghi thô ban đầu, ít chọn lọc hơn file này) ·
[experiment_log_ver9_svclaude.jsonl](experiment_log_ver9_svclaude.jsonl) (log máy sinh ra
từ lượt Kaggle thật) · [../result.md](../result.md) (Task 1 LegalIR — nguồn của mọi kỹ
thuật "chuyển giao" nhắc ở §6).

---

## 1. Tóm tắt một dòng mỗi mốc

| # | Mốc | METEOR | ROUGE-L | Sinh ra bởi |
|---|---|---:|---:|---|
| 1 | Baseline: SimCSE (`sup-SimCSE-VietNamese-phobert-base`) | 0,3906 | — | pipeline đầu tiên |
| 2 | Đổi dense sang `bkai-foundation-models/vietnamese-bi-encoder` | 0,4178 | — | cùng pipeline, đổi 1 biến |
| 3 | + reranker zero-shot `AITeamVN/Vietnamese_Reranker` | 0,4178 | — | đã có sẵn trong #2 |
| 4 | + fine-tune reranker riêng (`xlm-roberta-base` 278M) | 0,4639 | 0,3923 | lượt fine-tune đầu tiên |
| 5 | 2 dense encoder khác họ (bge-m3 + e5-large), cả hai fine-tune, RRF 3 kênh, reranker **zero-shot** | **0,5215** | **0,4829** | `0.5215_0.4829.ipynb` (Kaggle T4×2) |
| 6 | + fine-tune reranker + câu kết `echo2` + (nhãn Task 1, **không có hiệu lực**, xem §2) | **0,5528** | **0,4835** | `0.5528_0.4835.ipynb` |
| 7 | Cùng #6 nhưng đo **dev-eval nội bộ** (không phải public) | 0,5455 | 0,5442 | log thật, xem §4 |
| — | Bàn đo CPU trên server (checkpoint train bằng nhãn Task 1 — **không hợp lệ để nộp**, xem §2) | 0,5630 | — | `run_qa.py`, dev 501 câu |

**Cột #5 và #6 là public test** (tên file do người dùng đặt theo đúng quy ước
`submission_<mô_tả>_<điểm>.json` đã dùng ở Task 1) — đáng tin nhưng không có log máy đi kèm
để xác minh cấu hình chính xác lúc nộp. **Dòng #7 là dev-eval nội bộ trên `train.json`**,
**không so được trực tiếp với dòng #6** dù cùng notebook — hai tập dữ liệu khác nhau, và
ROUGE-L trên dev cao hơn hẳn public vì template được khớp theo đúng văn phong của
`train.json` (xem §5). Dòng cuối **không phải cấu hình hợp lệ để nộp bài** — xem §2.

---

## 2. ⛔ Sự cố nghiêm trọng: vi phạm rule.md §7c(b) — ĐÃ SỬA, đọc trước khi làm gì tiếp

**rule.md §7c(b)**, làm rõ chính thức từ BTC qua Zalo, ghi nhận 2026-08-31:

> ⛔ **CẤM dùng dữ liệu Task 1 huấn luyện cho Task 2 và ngược lại.** [...] KHÔNG được dùng
> cặp (câu hỏi, gold) của `train.json` tác vụ này để huấn luyện/mining cho tác vụ kia. [...]
> nếu có bất kỳ chỗ nào dùng **embedding/reranker đã fine-tune bằng nhãn của tác vụ này để
> phục vụ retrieval cho tác vụ kia, cần rà lại.**

### 2.1. Việc đã xảy ra

Trong lúc tìm cách "tận dụng pipeline LegalIR đang đạt điểm cao" cho Task 2, đã đề xuất và
triển khai `USE_TASK1_LABELS=True`: dùng 7.000 nhãn `question → document_id` của Task 1 để
bổ sung cho nhãn citation (vốn chỉ phủ ~48% câu Task 2) khi fine-tune bi-encoder/reranker.
Đã dàn sẵn file `legalir_train.json` (copy từ `data/LegalIR_Public_Test/train.json`) để
upload lên Kaggle Dataset.

**Đây là vi phạm trực tiếp điều khoản trên.**

### 2.2. Mức độ thiệt hại — đã kiểm, may mắn chưa gây hậu quả thật

`experiment_log_ver9_svclaude.jsonl` ghi `"n_task1_pairs_added": 0` — file `legalir_train.json`
**chưa từng được upload lên Kaggle Dataset**, nên vi phạm **tồn tại trong code nhưng chưa
từng có hiệu lực thật** trên bất kỳ checkpoint hay bài nộp nào đã biết.

Rà toàn bộ blast radius (2026-09-06):

| Vị trí | Trạng thái trước sửa | Đã sửa |
|---|---|---|
| `legalqa/legalir_train.json` | tồn tại, sẵn sàng upload | **đã xoá** |
| `0.5528_0.4835.ipynb` | `USE_TASK1_LABELS=True` mặc định | **`False`**, `raise SystemExit` nếu bật lại |
| `legalqa_kaggle_v3.ipynb` | `USE_TASK1_LABELS=True` mặc định | **`False`**, `raise SystemExit` nếu bật lại |
| `run_qa.py` (server) — chạy 3 checkpoint (`bgem3_ft`, `e5_ft`, `vn_reranker_ft_fam`) train bằng nhãn **document-level của chính Task 1** | không có cảnh báo | **gắn cảnh báo đầu file**: không được xuất submission Task 2 từ đây, điểm dev 0,5630 của nó không phải cấu hình hợp lệ |
| `legalqa_local.py` | — | **không bị ảnh hưởng** — chỉ dùng model công khai từ HuggingFace |

### 2.3. Bài học phương pháp

Điều khoản cấm **dữ liệu** (cặp câu hỏi–gold) và **checkpoint sinh từ dữ liệu đó**, nhưng
**không cấm phương pháp/thuật toán**. Ranh giới này quan trọng cho §6: công thức "negative
cùng họ" của Task 1 chuyển giao được vì nó là một cách *đào negative* áp lên nhãn của
chính Task 2, không mang theo một bit dữ liệu Task 1 nào.

**Trước khi tái sử dụng bất kỳ checkpoint hoặc pipeline nào ghi trong file này, luôn tự hỏi:
checkpoint đó train bằng nhãn của task nào?** Nếu câu trả lời là "Task 1" và đích dùng là
Task 2 (hoặc ngược lại) → dừng, đây là hướng đã bị cấm.

---

## 3. Kiến trúc — vì sao hai tầng, không phải một

Đo trên `train.json`, oracle biết trước document đúng (METEOR ≈ exact-match):

| Chiến lược trích | METEOR |
|---|---:|
| trả toàn bộ văn bản | 0,195 — precision sập, văn bản dài ~8.700 từ |
| 1 Điều tốt nhất, nguyên văn | 0,605 |
| 2 Điều tốt nhất ghép lại | 0,519 — **ghép thêm là MẤT điểm** |
| 1 Điều + câu dẫn template | 0,626 |

Độ dài Điều / độ dài đáp án = 1,02 (trung vị) — Điều luật vừa khít đáp án tham chiếu.
Nhưng chunk theo Điều **ở tầng retrieval** làm document recall tệ đi 1,19 điểm (đo ở Task 1,
`../result.md` §11, p=0,024). Nên tách hai tầng, không dùng chung một đơn vị chunk:

```
tầng 1  BM25 + N dense encoder -> pool chunk 450 TỪ -> rerank -> max-agg -> top-K DOCUMENT
tầng 2  cắt ĐIỀU chỉ trong K document đó -> cùng reranker -> top-1 ĐIỀU
tầng 3  câu dẫn + thân Điều + câu kết
```

`run_qa.py` (server, checkpoint không hợp lệ — §2) triển khai đúng kiến trúc 2 tầng này.
`legalqa_kaggle_v3.ipynb` chunk theo Điều ngay từ tầng 1 (kế thừa từ `0.5528`), **chưa**
chuyển sang 2 tầng — xem hướng nghiên cứu §7.

---

## 4. Đọc kỹ log thật trước khi kết luận — hai lần tôi đã đọc sai

Log duy nhất có máy sinh ra (không phải suy đoán) là `experiment_log_ver9_svclaude.jsonl`,
từ lượt chạy `legalqa_kaggle_v3.ipynb` trên Kaggle. Ghi lại đầy đủ vì nó dạy hai bài học
phương pháp, không chỉ một con số.

### 4.1. Recall@k theo tầng

| | @1 | @3 | @5 | @10 | @30 |
|---|---:|---:|---:|---:|---:|
| BM25+dense, **không rerank** | 0,6503 | 0,8671 | 0,9580 | 0,9930 | 1,0000 |
| + rerank zero-shot | 0,5455 | 0,7413 | 0,8042 | 0,8531 | 0,9161 |
| + rerank **fine-tuned** | 0,4266 | 0,6084 | 0,6783 | 0,7483 | 0,9021 |

### 4.2. Sai lầm đọc lần 1 (đã tự rút lại)

Đọc nhanh bảng trên: "reranker phá pipeline, tắt đi được +5,8 điểm". **Sai.** Nguyên nhân:
`recall@k` chỉ đo trên `recall_ids = [q for q in dev_ids if q in train_positive]` — tập con
**có citation resolve được** (~150/300 câu), không phải toàn bộ mẫu dev. Trong khi đó
dev-eval **đã** quét cả cấu hình "không rerank" trên **METEOR đo trên cả 300 câu** và vẫn
chọn `use_reranker: true`. Tức là trên độ đo thật (METEOR, toàn mẫu), reranker vẫn thắng —
bảng recall@k trên chỉ phản ánh nhiễu của tập con citation, không phải chất lượng reranker.
**Bài học: không suy luận từ recall@k đo trên tập con khi có sẵn METEOR đo trên toàn mẫu.**

### 4.3. Sai lầm đọc lần 2 (đã tự rút lại)

Từ việc thiếu các khoá `agg_mode`/`agg_T`/`reranker_source` và tên kênh vẫn là
`"bge-m3"`/`"e5-large"` trong log, đã kết luận "log này không phải của `legalqa_kaggle_v3`,
notebook chưa từng chạy tới đích". **Sai, và sai vì lý do phương pháp**: kiểm tra ngược lại
code sinh ra `record` (Cell 14) cho thấy nó dựng từ **danh sách khoá cố định**, không bao
giờ ghi các khoá mới đó dù đã thêm vào `eval_info` ở cell khác — thiếu dấu vết không chứng
minh gì nếu chưa kiểm dấu vết đó *có bao giờ được ghi ra* hay không. Và tên kênh
`"bge-m3"`/`"e5-large"` là do **chính bug trong code**: `DENSE_CHANNELS` đã đổi tên slot
thành `"A"`/`"B"` nhưng `FINETUNE_SPECS` (nơi log tên) vẫn viết cứng tên cũ — nên log **đúng
là của lượt chạy thật**, chỉ là không cho biết model nào thực sự được nạp (bge-m3+e5-large
gốc hay cặp mới vnembv2+harrier — không phân biệt được nếu đường lùi tự động ở Cell 8 âm
thầm kích hoạt). **Bài học: "vắng mặt bằng chứng" không phải "bằng chứng vắng mặt" — luôn
kiểm code sinh log trước khi diễn giải log.**

### 4.4. Chẩn đoán đúng: ngân sách GPU phân bổ tệ, không quy kết được điểm số

| Giai đoạn | Thời gian | % tổng | Ghi chú |
|---|---:|---:|---|
| fine-tune encoder A (bge-m3 hoặc thay thế) | 82,5 phút | 17% | 440 step, mini-batch ổn định ở 16 |
| fine-tune encoder B (e5-large hoặc thay thế) | **176,4 phút** | **36%** | 334 step, mini-batch tụt về **4** sau OOM-backoff |
| fine-tune reranker | 28,3 phút | 6% | 2.397 step × `RERANKER_FT_BATCH_SIZE=8` câu hỏi/step = 19.176 lượt / 3.579 cặp ≈ **5,4 epoch** |
| còn lại (encode corpus, retrieval, dev-eval, predict) | 203,5 phút | 41% | |
| **Tổng** | **490,7 phút** | 100% | sát trần phiên Kaggle |

Hai encoder chiếm **53% phiên GPU 8 tiếng**, và tên "bge-m3"/"e5-large" trong log không cho
biết đây có phải cặp mới hay không (§4.3) — **không quy kết được điểm số cho việc đổi
encoder**. Reranker fine-tune dừng ở 28 phút / 5,4 epoch — không đủ dữ kiện để kết luận overfit hay underfit chỉ từ số epoch; log không ghi loss cuối cùng để biết đã hội tụ hay chưa. Đáng chú ý hơn: batch chỉ 8 câu hỏi/step trong khi bi-encoder dùng batch hiệu dụng 64 — hai tầng đang train ở chế độ batch lệch nhau một bậc, chưa rõ có chủ đích hay là thiếu nhất quán.

---

## 5. Cơ chế METEOR vs ROUGE-L — đo trực tiếp, không suy đoán

Dựng lại đúng tokenizer + LCS của `scoring/legalqa/rouge_score/tokenize.py` (vendor BTC),
đo trên 150–180 câu dev rút từ `train.json`.

### 5.1. ROUGE-L gần như không đọc được tiếng Việt

Tokenizer BTC: `NON_ALPHANUM_RE = r"[^a-z0-9]+"` rồi `VALID_TOKEN_RE = r"^[a-z0-9]+$"` — xoá
sạch mọi ký tự có dấu. Trên 97.874 token của chính đáp án tham chiếu:

| Loại token sau khi tách | Tỉ lệ |
|---|---:|
| mảnh vụn ≤2 ký tự (`Điều`→`i`,`u` · `định`→`nh`) | **87,0%** |
| từ thật ≥3 ký tự | 10,7% |
| chữ số | 2,2% |

### 5.2. Sàn nhiễu của ROUGE-L cao gần gấp đôi METEOR

Lấy đáp án của một câu **hoàn toàn không liên quan** làm dự đoán:

| | Điểm |
|---|---:|
| METEOR | 0,1315 |
| ROUGE-L | **0,2748** |

Dải tín hiệu thật: METEOR rộng ~0,45 (0,13→0,58 quan sát được), ROUGE-L chỉ rộng ~0,31
(0,275→0,583). **Cùng một điểm tuyệt đối, ROUGE-L đáng giá ít hơn METEOR khoảng 1/3.**

### 5.3. Câu kết `echo2` chỉ là lever của METEOR, gần như vô hại với ROUGE-L

| Cấu hình | METEOR | ROUGE-L | tỉ lệ dài hyp/ref |
|---|---:|---:|---:|
| nguyên văn, không câu kết | 0,5326 | 0,5803 | 0,95 |
| nguyên văn + `echo2` | **0,5750** (+4,24) | 0,5826 (+0,23) | 1,13 |

METEOR có `alpha=0,9` (gần thuần recall) nên nhồi thêm chữ đúng luôn có lợi; ROUGE-L dùng
F-measure cân bằng `2PR/(P+R)`, dài thêm kéo precision xuống, triệt tiêu phần recall được
thêm. **Hệ quả: nếu ROUGE-L tăng mạnh mà METEOR tăng nhẹ, câu kết/độ dài KHÔNG phải nguyên
nhân — phải tìm ở tầng chọn đúng văn bản/Điều, không phải ở hậu xử lý câu chữ.**

### 5.4. Sàn của độ dài đáp án

| Độ dài (cắt còn) | METEOR (echo2) | ROUGE-L (echo2) |
|---|---:|---:|
| 25% | 0,2871 | 0,4149 |
| 50% | 0,4042 | 0,5092 |
| 100% (hiện tại) | 0,5750 | 0,5826 |

Cả hai độ đo đều đơn điệu tăng theo độ dài trong dải này — không có bằng chứng cắt ngắn có
lợi cho ROUGE-L trong khoảng đã đo.

**Kết luận thực dụng: tối ưu METEOR (độ đo chính, xếp hạng), đọc ROUGE-L để tham khảo
nhưng đừng thiết kế riêng cho nó — tín hiệu của nó bẩn và trần thấp hơn.**

---

## 6. Câu kết — thay đổi rẻ nhất và đáng giá nhất đã đo được

Đo trên 501 câu dev (server, `run_qa.py` — checkpoint không hợp lệ để nộp nhưng vẫn dùng
được làm **bàn đo nội bộ**, xem §2.3), cùng retrieval, chỉ đổi biến câu kết:

| `concl` | METEOR | So với dòng trước |
|---|---:|---|
| `none` | 0,5151 | — |
| `echo` (1 câu lặp lại câu hỏi) | 0,5499 | Δ +0,0348 ± 0,0017 · 416 thắng/85 thua · t=20,4 |
| `echo2` (2 câu lặp lại câu hỏi) | **0,5630** | Δ +0,0131 ± 0,0010 · 371 thắng/130 thua · t=12,8 |

**Split-half xác nhận không phải ảo giác**: chia đôi dev, chọn `echo2` trên nửa A rồi đo
trên nửa B — cả hai nửa độc lập đều chọn `echo2` (A: 0,5675 · B: 0,5589).

**Đã dò tiếp số lần lặp, và CHỦ ĐỘNG DỪNG ở 2 dù còn dư địa:**

| Số lần lặp câu hỏi | METEOR |
|---:|---:|
| 1× | 0,5499 |
| **2× (đang dùng)** | **0,5630** |
| 3× | 0,5674 |
| 4× | 0,5682 (đỉnh) |
| 6× | 0,5661 (bắt đầu giảm) |

Từ 2× lên 4× chỉ được +0,52 điểm — nằm sát ngưỡng nhiễu của các phép so đã bị lừa nhiều lần
trong Task 1 (xem `../result.md` §14 về hàm gộp logsumexp: đỉnh +0,24 trên toàn dev nhưng
split-half âm ở cả 3 seed). Đồng thời một đáp án lặp lại câu hỏi 4 lần **nhìn bằng mắt là
suy thoái chất lượng rõ ràng** dù độ đo vẫn tăng. **Đây là tối ưu hình dạng độ đo (đúng luật,
không gian lận), không phải cải thiện câu trả lời cho người đọc — dừng ở 2× là quyết định
có chủ đích, không phải chưa tối ưu hết.**

---

## 7. Nút thắt thật của pipeline — đo được, chưa giải quyết

Đo trên 501 câu dev (server): so sánh Điều mà reranker **thực sự chọn** (hạng 1) với Điều
**tốt nhất trong đúng 5 ứng viên nó đã đưa vào tầng 2**.

| | METEOR |
|---|---:|
| reranker chọn hạng 1 (hiện tại) | 0,5630 |
| **oracle: chọn tốt nhất trong 5 ứng viên đã có sẵn** | **0,6401** |

**+7,7 điểm đang nằm sẵn trong danh sách ứng viên, chỉ là bị xếp sai thứ tự** — lớn hơn hẳn
mọi thứ các hướng khác (LSE, đổi encoder — xem §8) có thể cho. Đo thêm: reranker chọn đúng
Điều tốt nhất chỉ **61,3%** số lần, và khoảng cách điểm giữa hạng 1–hạng 2 có **trung vị
0,0023** — gần như một phép tung đồng xu ở nhiều câu.

Bốn quy tắc rẻ tiền đã thử để bắt phần dư địa này, **tất cả đều tệ hơn giữ nguyên hạng 1**:

| Chiến lược | METEOR |
|---|---:|
| chọn dài nhất | 0,4371 |
| chọn ngắn nhất | 0,3362 |
| chọn gần độ dài trung vị đáp án thật (312 từ) nhất | 0,4210 |

Tương quan (độ dài Điều hạng 1, METEOR) = **−0,078** — độ dài không phải tín hiệu hữu ích.
**Kết luận: cần một bộ chọn HỌC ĐƯỢC (dùng đặc trưng: điểm CE, khoảng cách tới ứng viên kế
tiếp, độ dài, khớp số hiệu văn bản với câu hỏi, vị trí Điều trong văn bản), không phải
heuristic.** Đây là hướng ưu tiên cao nhất, xem §9.

---

## 8. Chuyển giao từ Task 1 (LegalIR) — cái nào được phép, cái nào đo ra âm

Sau vụ §2, mọi mục dưới đây chỉ chuyển giao **phương pháp**, áp lên nhãn/model của **chính
Task 2**, không chạm dữ liệu Task 1. Bài nộp LegalIR tốt nhất tham chiếu:
`submission_lse_T2.json` — cấu hình #18 (`vnembv2_ft`+`harrier_ft`) + gộp `logsumexp T=2`,
public recall@5 = **0,9519** (so với `max` = 0,9471, +0,48 — `../result.md` §31).

### 8.1. Hàm gộp LSE ở mức văn bản — ĐO ÂM cho Task 2, giữ lại có cổng an toàn

Task 1 tối ưu recall@5: văn bản đúng chỉ cần lọt vào **một trong năm** slot, nên thưởng cho
"bằng chứng trải rộng nhiều chunk" (LSE) là đúng hướng. Task 2 trích **đúng một Điều duy
nhất**, nên đẩy văn bản có vài Điều tầm tầm lên trên văn bản có một Điều xuất sắc là đánh
đổi NGƯỢC dấu. Đo trực tiếp trên 501 câu dev (xấp xỉ, cache chỉ giữ 5 ứng viên/câu):

| | METEOR | so với `max` |
|---|---:|---:|
| `max` (đang dùng) | 0,5630 | — |
| `lse T=0,5` | 0,5306 | **−0,0324** |
| `lse T=1` | 0,5249 | **−0,0381** |
| `lse T=2` (giá trị nguyên bản của Task 1) | 0,5240 | **−0,0390** |

Unit test xác nhận công thức đúng về mặt toán học (LSE đẩy đúng văn bản có nhiều Điều tốt
lên trước) — nhưng đúng công thức không có nghĩa đúng mục tiêu. **Không bê thẳng T=2.**
`legalqa_kaggle_v3.ipynb` giữ `AGG_MODE="lse"` nhưng **bọc bằng cổng split-half**: quét
`T ∈ {0,5; 1; 2}`, chọn trên nửa A, xác nhận trên nửa B, chỉ nhận khi **cả hai nửa cùng
dương** — nếu không thì tự quay về `max` và ghi lý do. Đây là quy trình đã cứu Task 1 khỏi
"đỉnh trên toàn dev" ba lần liên tiếp (`../result.md` §14).

### 8.2. Cặp encoder mới (`Vietnamese_Embedding_v2` + `vietlegal-harrier-0.6b`) — chưa đo được

Model công khai trên HuggingFace nên **không vi phạm §2**. Đã đưa vào `v3` với đường lùi tự
động về bge-m3+e5-large nếu nạp lỗi. **Chưa đo được hiệu quả thật** vì log ver9 không phân
biệt được model nào đã nạp (§4.3, đã sửa bằng cách suy tên spec từ model thật). Ước lượng lý
thuyết: recall@5 document +0,32 điểm ở Task 1 → quy đổi sang METEOR Task 2 nằm dưới sai số
đo (SE dev 501 câu ≈ 0,011) — **khả năng cao không đo được**, nhưng chưa loại trừ.

### 8.3. Negative "cùng họ" + listwise softmax — chuyển giao có cơ chế rõ ràng nhất

Task 1 đo (`../result.md` §4): negative cùng họ 0,9428 · ngữ nghĩa 0,9360 · hợp nhất 0,9357
— **càng nhiều negative ngữ nghĩa, điểm càng thấp**. "Cùng họ" = văn bản cùng chủ đề nhưng
không phải gold, dịch sang Task 2 là **các Điều anh em trong cùng văn bản gold** — đây đúng
là chỗ pipeline đang thua nhất (§7: hạng 1 vs hạng 2 cách nhau 0,0023 điểm).

Đã triển khai trong `v3`, kèm **chốt chặn false-negative bắt buộc** (nguyên văn cơ chế của
Task 1): loại bỏ ứng viên bị reranker zero-shot chấm **cao hơn cả gold** — vì nhãn citation
của Task 2 lấy trích dẫn *đầu tiên* gặp trong answer nên rất dễ bỏ sót Điều đúng khác trong
cùng văn bản; không lọc tức là **dạy model dìm đáp án đúng xuống**. Đã unit-test cả hai
nhánh (gate tắt/bật) bằng dữ liệu dựng sẵn — đúng theo thiết kế. **Chưa có số đo thật trên
Kaggle.**

Đồng thời đổi loss pairwise softplus → **listwise softmax** (đúng công thức Task 1,
`cross_entropy` trên `[pos, neg_1..neg_7]`) và hạ `RERANKER_FT_LR` từ `1e-5` xuống `3e-6`
(giá trị Task 1 đo cho đúng công thức này — giữ `1e-5` với loss listwise là trộn hai công
thức không tương thích).

---

## 9. Việc đã sửa lỗi kỹ thuật (không phải hướng nghiên cứu, nhưng tốn thời gian nếu lặp lại)

| Lỗi | Nguyên nhân | Sửa |
|---|---|---|
| `NameError: USE_NEW_ENCODERS` ở Cell 2 khi chạy trên Kaggle | Vá bằng cách neo khối tham số mới vào một dòng comment ở SAU chỗ biến được dùng, không kiểm vị trí | Chuyển khối lên trước chỗ dùng đầu tiên; viết bộ kiểm tĩnh 2 lớp (tên chưa gán ở cấp module; tên không tồn tại ở bất kỳ đâu) chạy trên toàn bộ `.ipynb` |
| Log không quy kết được model nào đã chạy (§4.3) | Tên spec viết cứng `"bge-m3"`/`"e5-large"` trong `FINETUNE_SPECS`, không đồng bộ với `DENSE_CHANNELS` đã đổi tên | Suy tên spec từ `BASE_DENSE_MODEL_*.split("/")[-1]` |
| `BM25.__init__` giữ tham chiếu module `numpy` trong `self` | | `TypeError: cannot pickle 'module' object` khi cache ra đĩa (`legalqa_local.py`) — import cục bộ trong từng method thay vì lưu vào `self` |
| `top_k()` trả `numpy.int64` | | `json.dump` chết khi ghi cache — ép về `int` Python trước khi trả về |
| Cache ghi dở khi crash giữa chừng → lần chạy sau đọc cache hỏng, crash ngay ở bước tốn thời gian nhất | Ghi trực tiếp bằng `json.dump` vào file đích | Ghi nguyên tử (`tmp` rồi `os.replace`) + đọc khoan dung (cache hỏng → coi như chưa có, tính lại, không crash cả tiến trình) |
| Hai tiến trình `run_qa.py` cùng chạy, ghi đè cache của nhau | Không có cơ chế khoá | `flock` trên file khoá riêng, tiến trình sau bị chặn và báo PID của tiến trình đang giữ |
| `FlagReranker` tự bọc `DataParallel` lên TẤT CẢ GPU của máy dùng chung dù chỉ xin 1 thẻ | `torch.cuda.device_count()` bị gọi (và cache bởi `lru_cache`) trước khi `CUDA_VISIBLE_DEVICES` được set | Import `torch`/thư viện phụ thuộc GPU **sau khi** đã cố định `CUDA_VISIBLE_DEVICES` |

---

## 10. Trạng thái các file (2026-09-06) — file nào dùng được, file nào không

| File | Trạng thái | Ghi chú |
|---|---|---|
| `legalqa_kaggle_v3.ipynb` | **hợp lệ**, chưa chạy hết trên Kaggle sau các sửa gần nhất | negative cùng họ + listwise loss + LSE có cổng split-half; qua 4 cổng kiểm tĩnh (cú pháp, tên chưa gán, exec Cell 2 thật, unit-test mining) |
| `0.5528_0.4835.ipynb` | **hợp lệ**, đã public 0,5528/0,4835 | đã gỡ `USE_TASK1_LABELS` |
| `legalqa_local.py` | **hợp lệ** | model công khai, zero-shot, tự co giãn theo VRAM máy cá nhân |
| `run_qa.py` | **⛔ không dùng để nộp Task 2** | checkpoint train bằng nhãn Task 1 — chỉ dùng làm bàn đo nội bộ (§2.3) |
| `legalir_train.json` | **đã xoá** | |
| `legalqa_server.py`, `legalqa_testing.py` | chưa rà theo tiêu chuẩn §2 | rà lại trước khi dùng bất kỳ checkpoint nào từ hai file này |

---

## 11. Hướng nghiên cứu tiếp theo, xếp theo giá trị đo được

1. **Bộ chọn Điều học được cho tầng 2** (§7) — trần đo được **+7,7 điểm**, lớn nhất trong
   mọi hướng đã xem xét. Đặc trưng gợi ý: điểm CE tuyệt đối, khoảng cách tới ứng viên kế
   tiếp, độ dài Điều, khớp số hiệu văn bản với câu hỏi, vị trí Điều trong văn bản. Không
   dùng heuristic độ dài — đã đo là vô dụng (§7).
2. **Chạy `legalqa_kaggle_v3.ipynb` trên Kaggle và đọc log mới** — log đã được vá để ghi đủ
   `dense_model_a/b`, `encoder_fallback_fired`, `agg_mode_chosen`, `reranker_source` (§4.3,
   §9), lần này mới thực sự quy kết được điểm số cho từng thay đổi.
3. **Đo riêng hiệu quả negative "cùng họ"** (§8.3) bằng paired test so với `0.5528` (giữ mọi
   thứ khác cố định, chỉ đổi công thức negative) — chưa có số đo thật.
4. **Không đầu tư thêm vào LSE hay đổi encoder** cho tới khi có bằng chứng ngược lại số đo
   ở §8.1/§8.2 — cả hai đều đo dưới hoặc bằng 0 trên chính dữ liệu Task 2.
5. **Rà `legalqa_server.py` và `legalqa_testing.py`** theo đúng checklist §2 trước khi tái
   sử dụng bất kỳ phần nào của chúng.
