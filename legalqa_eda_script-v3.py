import os
import json
import re
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Thiết lập backend không tương tác để tránh lỗi Tcl/Tkinter
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style for plotting
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

# ==========================================
# CONFIGURATION & DATA LOADERS
# ==========================================

# Người dùng yêu cầu xếp dữ liệu cùng thư mục với file code
DATA_DIR = "." 
TRAIN_PATH = "train.json"
WARMUP_PATH = "warmup.json"
CONTEXT_DIR = "." # Các tệp context_*.json nằm trực tiếp cùng thư mục với file code

# Thư mục lưu các kết quả phân tích đầu ra
OUTPUT_DIR = "eda_output"

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)

class DualLogger:
    """Ghi dữ liệu đồng thời ra màn hình console và tệp báo cáo."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout để tự động ghi lại toàn bộ nội dung in ra thành báo cáo
report_path = os.path.join(OUTPUT_DIR, "eda_report.txt")
sys.stdout = DualLogger(report_path)

def load_json(file_path):
    """Đọc file JSON an toàn."""
    if not os.path.exists(file_path):
        print(f"[CẢNH BÁO] Không tìm thấy file: {file_path}")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_corpus(context_dir):
    """Load tất cả các file context_*.json từ thư mục."""
    corpus = []
    pattern = os.path.join(context_dir, "context_*.json")
    files = glob.glob(pattern)
    
    if not files:
        # Thử quét đệ quy nếu cấu trúc thư mục lồng nhau
        pattern = os.path.join(context_dir, "**/context_*.json")
        files = glob.glob(pattern, recursive=True)
        
    print(f"-> Tìm thấy {len(files)} file context JSON.")
    for fpath in files:
        data = load_json(fpath)
        if data:
            if isinstance(data, list):
                corpus.extend(data)
            elif isinstance(data, dict):
                corpus.append(data)
    return corpus

# ==========================================
# MODULE 1: ANALYZE QUERIES (CÂU HỎI)
# ==========================================

def analyze_questions(train_data):
    """
    Phân tích đặc trưng của tập câu hỏi (Queries):
    - Phân phối độ dài (từ, ký tự)
    - Phân loại dạng câu hỏi (Heuristic)
    - Nhận diện từ khóa pháp lý (Số hiệu văn bản, điều khoản, tên luật)
    """
    print("\n=== [MÔ ĐUN 1] PHÂN TÍCH TẬP CÂU HỎI ===")
    
    # Giả định cấu trúc train_data dạng list of dicts. 
    # Nếu train_data dạng dict (key là id), chuyển đổi sang list.
    if isinstance(train_data, dict):
        questions_list = []
        for q_id, q_val in train_data.items():
            item = {"id": q_id}
            item.update(q_val)
            questions_list.append(item)
    else:
        questions_list = train_data

    # Trích xuất trường 'question' / 'text' tuỳ theo schema
    q_key = 'question' if 'question' in questions_list[0] else 'text' if 'text' in questions_list[0] else None
    if not q_key:
        print("[LỖI] Không tìm thấy trường câu hỏi trong train/warmup. Các key hiện tại:", questions_list[0].keys())
        return pd.DataFrame()

    df_q = pd.DataFrame(questions_list)
    df_q['num_chars'] = df_q[q_key].apply(len)
    df_q['num_words'] = df_q[q_key].apply(lambda x: len(x.split()))

    # 1. Thống kê độ dài câu hỏi
    print("\n--- 1. Thống kê độ dài câu hỏi (Tính bằng số Từ/Words) ---")
    print(df_q['num_words'].describe())

    # Vẽ phân phối độ dài câu hỏi
    plt.figure()
    sns.histplot(df_q['num_words'], kde=True, bins=30, color='royalblue')
    plt.title("Phân phối độ dài câu hỏi (Số từ)")
    plt.xlabel("Số lượng từ")
    plt.ylabel("Tần suất")
    
    # Lưu vào thư mục eda_output
    q_plot_path = os.path.join(OUTPUT_DIR, "question_length_distribution.png")
    plt.savefig(q_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Đã lưu biểu đồ phân phối độ dài câu hỏi tại '{q_plot_path}'")

    # 2. Phân loại dạng câu hỏi (Heuristic)
    def classify_question_type(text):
        text_lower = text.lower()
        if any(w in text_lower for w in ["có được", "có được phép", "có được không", "có phải là", "không?", "hay không"]):
            return "True/False (Xác nhận)"
        elif any(w in text_lower for w in ["là gì", "như thế nào là", "thế nào là", "khái niệm", "định nghĩa"]):
            return "Định nghĩa / Khái niệm"
        elif any(w in text_lower for w in ["khi nào", "trường hợp nào", "thủ tục", "bao nhiêu lâu", "mức phạt"]):
            return "Tình huống / Thực thi / Mức phạt"
        else:
            return "Yêu cầu thông tin khác"

    df_q['q_type'] = df_q[q_key].apply(classify_question_type)
    type_counts = df_q['q_type'].value_counts()
    print("\n--- 2. Tỷ lệ các dạng câu hỏi (Dự đoán bằng từ khoá) ---")
    for t, count in type_counts.items():
        print(f" - {t}: {count} câu ({count/len(df_q)*100:.2f}%)")

    # 3. Nhận diện từ khóa pháp lý cốt lõi ngay trong câu hỏi
    regex_so_hieu = r'\b\d+/\d+/[A-ZĐ-]+|\b\d+/[A-ZĐ-]+'
    regex_dieu_luat = r'\b[Đđ]iều\s+\d+'
    
    df_q['has_doc_id'] = df_q[q_key].apply(lambda x: 1 if re.search(regex_so_hieu, x) else 0)
    df_q['has_article_ref'] = df_q[q_key].apply(lambda x: 1 if re.search(regex_dieu_luat, x) else 0)

    print("\n--- 3. Mật độ chứa từ khóa trích dẫn trực tiếp trong câu hỏi ---")
    print(f" - Tỷ lệ câu hỏi chứa Số hiệu văn bản cụ thể (Ví dụ: 12/2020/NĐ-CP): {df_q['has_doc_id'].mean()*100:.2f}%")
    print(f" - Tỷ lệ câu hỏi đề cập trực tiếp từ khóa 'Điều [số]': {df_q['has_article_ref'].mean()*100:.2f}%")

    return df_q

# ==========================================
# MODULE 2: ANALYZE CORPUS/ARTICLES (VĂN BẢN)
# ==========================================

def analyze_corpus(corpus_data):
    """
    Phân tích đặc trưng kho văn bản (Corpus):
    - Phân phối độ dài của các Điều luật / Chunks
    - Mật độ tham chiếu chéo giữa các văn bản pháp luật
    """
    print("\n=== [MÔ ĐUN 2] PHÂN TÍCH TẬP VĂN BẢN PHÁP LUẬT ===")
    if not corpus_data:
        print("[LỖI] Tập văn bản rỗng.")
        return pd.DataFrame()

    df_c = pd.DataFrame(corpus_data)
    df_c['num_chars'] = df_c['passage'].apply(len)
    df_c['num_words'] = df_c['passage'].apply(lambda x: len(x.split()))

    # 1. Phân phối độ dài điều luật
    print("\n--- 1. Thống kê độ dài các Điều luật (Số Từ/Words) ---")
    print(df_c['num_words'].describe())

    # Gợi ý thiết lập chunk_size dựa trên phân vị (percentiles)
    p95 = int(np.percentile(df_c['num_words'], 95))
    p99 = int(np.percentile(df_c['num_words'], 99))
    print(f" * Phân vị 95% độ dài văn bản là: {p95} từ -> Gợi ý chunk_size tối thiểu là {p95} từ để không làm đứt mạch văn bản.")
    print(f" * Phân vị 99% độ dài văn bản là: {p99} từ.")

    plt.figure()
    sns.histplot(df_c['num_words'], kde=True, bins=40, color='darkorange')
    plt.title("Phân phối độ dài của các Điều luật (Số từ)")
    plt.xlabel("Số lượng từ")
    plt.ylabel("Tần suất")
    
    # Lưu vào thư mục eda_output
    c_plot_path = os.path.join(OUTPUT_DIR, "article_length_distribution.png")
    plt.savefig(c_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Đã lưu biểu đồ phân phối độ dài văn bản tại '{c_plot_path}'")

    # 2. Phân tích mật độ tham chiếu chéo (Cross-reference density)
    regex_cross_ref = r'([đĐ]iều|[kK]hoản|[đĐ]iểm)\s+\d+.*?([đĐ]iều|[lL]uật|[nN]ghị\s+định|[qQ]uyết\s+định)\s+'
    df_c['has_cross_ref'] = df_c['passage'].apply(lambda x: 1 if re.search(regex_cross_ref, x) else 0)

    cross_ref_rate = df_c['has_cross_ref'].mean() * 100
    print("\n--- 2. Phân tích tham chiếu chéo (Cross-reference) ---")
    print(f" - Tỉ lệ Điều luật chứa tham chiếu đến điều khoản khác: {cross_ref_rate:.2f}%")
    print("   * Lưu ý: Tỷ lệ này cao biểu thị mô hình Retriever cần hỗ trợ ngữ cảnh rộng hơn hoặc có cơ chế liên kết đồ thị (graph-based / metadata expansion).")

    return df_c

# ==========================================
# MODULE 3: RETRIEVAL & RELATION ANALYSIS
# ==========================================

def analyze_retrieval_relations(train_data, corpus_data):
    """
    Phân tích quan hệ Câu hỏi - Văn bản đáp án:
    - Số lượng document mục tiêu trung bình trên mỗi câu hỏi
    - Độ trùng lặp từ vựng trực tiếp (Lexical Overlap) giữa câu hỏi và văn bản vàng
    """
    print("\n=== [MÔ ĐUN 3] PHÂN TÍCH QUAN HỆ CÂU HỎI - ĐÁP ÁN ===")
    if not train_data or not corpus_data:
        print("[LỖI] Dữ liệu đầu vào không đủ.")
        return

    corpus_map = {str(item['id']): item['passage'] for item in corpus_data}

    if isinstance(train_data, dict):
        questions_list = []
        for q_id, q_val in train_data.items():
            item = {"id": q_id}
            item.update(q_val)
            questions_list.append(item)
    else:
        questions_list = train_data

    target_docs_counts = []
    overlaps = []

    def simple_tokenize(text):
        tokens = re.findall(r'\w+', text.lower())
        return set(tokens)

    missing_context_count = 0

    for item in questions_list:
        q_text = item.get('question', item.get('text', ''))
        gold_ids = item.get('answer', item.get('relevant_articles', item.get('document_ids', [])))
        if isinstance(gold_ids, str):
            gold_ids = [gold_ids]
        
        target_docs_counts.append(len(gold_ids))
        
        q_tokens = simple_tokenize(q_text)
        
        gold_passages = []
        for g_id in gold_ids:
            g_id_str = str(g_id)
            if g_id_str in corpus_map:
                gold_passages.append(corpus_map[g_id_str])
            else:
                missing_context_count += 1
                
        if gold_passages:
            combined_gold_text = " ".join(gold_passages)
            gold_tokens = simple_tokenize(combined_gold_text)
            
            intersection = q_tokens.intersection(gold_tokens)
            union = q_tokens.union(gold_tokens)
            
            jaccard = len(intersection) / len(union) if union else 0
            overlap_coeff = len(intersection) / min(len(q_tokens), len(gold_tokens)) if min(len(q_tokens), len(gold_tokens)) else 0
            
            overlaps.append({
                'jaccard': jaccard,
                'overlap_coeff': overlap_coeff
            })

    # 1. Phân tích số lượng tài liệu vàng cần truy xuất
    print("\n--- 1. Thống kê số lượng văn bản vàng cần thiết cho mỗi câu hỏi ---")
    counts_series = pd.Series(target_docs_counts)
    print(counts_series.describe())
    
    val_counts = counts_series.value_counts().sort_index()
    print("\n Tỷ lệ số lượng văn bản đúng:")
    for k, v in val_counts.items():
        print(f" - {k} văn bản: {v} câu hỏi ({v/len(counts_series)*100:.2f}%)")

    # 2. Thống kê độ trùng lặp từ vựng (Lexical Overlap)
    if overlaps:
        df_ov = pd.DataFrame(overlaps)
        print("\n--- 2. Đánh giá trùng lặp từ vựng trực tiếp (Lexical Overlap) ---")
        print(f" - Chỉ số Jaccard trung bình: {df_ov['jaccard'].mean():.4f}")
        print(f" - Overlap Coefficient trung bình: {df_ov['overlap_coeff'].mean():.4f}")
        
        low_overlap_pct = (df_ov['overlap_coeff'] < 0.2).mean() * 100
        print(f" - Tỷ lệ câu hỏi có độ trùng khớp từ vựng rất thấp (< 20%): {low_overlap_pct:.2f}%")
        print("   * Nhận xét: Nếu tỷ lệ này > 15%, thuật toán BM25 truyền thống sẽ bỏ lỡ rất nhiều câu hỏi do hiện tượng lệch pha từ vựng (Vocabulary Mismatch). Cần bắt buộc dùng Dense Retriever (Bi-Encoder) để bổ trợ.")
    
    if missing_context_count > 0:
        print(f"\n[CẢNH BÁO] Có {missing_context_count} ID văn bản đúng được trích dẫn trong train nhưng không tìm thấy nội dung tương ứng trong kho Corpus.")

# ==========================================
# MODULE 4: ANSWER STYLE & TEMPLATES
# ==========================================

def analyze_answer_styles(train_data):
    """
    Phân tích phong cách viết và cấu trúc của Câu trả lời thực tế (Ground-truth answers):
    - Phân phối độ dài câu trả lời
    - Nhận diện các template phổ biến được chuyên gia pháp lý sử dụng
    """
    print("\n=== [MÔ ĐUN 4] PHÂN TÍCH TẬP CÂU TRẢ LỜI THAM CHIẾU ===")
    
    if isinstance(train_data, dict):
        questions_list = []
        for q_id, q_val in train_data.items():
            item = {"id": q_id}
            item.update(q_val)
            questions_list.append(item)
    else:
        questions_list = train_data

    # Trích xuất trường đáp án bằng chữ (thường là 'answer_text', 'response' hay 'answer')
    ans_key = None
    for key in ['answer_text', 'response', 'target', 'gold_answer', 'answer']:
        if len(questions_list) > 0 and key in questions_list[0]:
            if isinstance(questions_list[0][key], str):
                ans_key = key
                break
                
    if not ans_key:
        print("[LỖI] Không tìm thấy trường văn bản câu trả lời (Task 2) trong tập dữ liệu. Các key có sẵn:", questions_list[0].keys())
        return

    df_ans = pd.DataFrame(questions_list)
    df_ans['num_chars'] = df_ans[ans_key].apply(len)
    df_ans['num_words'] = df_ans[ans_key].apply(lambda x: len(x.split()))

    # 1. Thống kê độ dài câu trả lời
    print(f"\n--- 1. Thống kê độ dài câu trả lời vàng (Trường '{ans_key}') ---")
    print(df_ans['num_words'].describe())

    # 2. Phát hiện các khuôn mẫu mở đầu (Template Detection)
    templates = [
        ("Căn cứ", r"^Căn cứ"),
        ("Theo Điều", r"^Theo Điều"),
        ("Theo quy định", r"^Theo quy định"),
        ("Như vậy / Theo đó", r"^(Như vậy|Theo đó)"),
        ("Trực tiếp trích văn bản", r"^[A-ZĐ]")
    ]

    print("\n--- 2. Phân tích khuôn mẫu mở đầu câu trả lời (Mẹo thiết kế Template sinh) ---")
    matched_any = 0
    for label, regex in templates:
        matches = df_ans[ans_key].apply(lambda x: 1 if re.match(regex, x.strip()) else 0).sum()
        pct = (matches / len(df_ans)) * 100
        print(f" - Mẫu bắt đầu bằng '{label}': {matches} câu ({pct:.2f}%)")
        matched_any += matches

    print(f" - Các dạng mở đầu khác: {len(df_ans) - matched_any} câu ({(len(df_ans) - matched_any)/len(df_ans)*100:.2f}%)")
    print("   * Gợi ý: Nếu mẫu 'Căn cứ...' hoặc 'Theo Điều...' chiếm ưu thế tuyệt đối, việc bạn thiết lập template sinh tương ứng sẽ kéo điểm METEOR/ROUGE-L lên cực nhanh (như thực tế thử nghiệm đã tăng +4.8 METEOR).")

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================

def main():
    print("=====================================================================")
    print("       HỆ THỐNG PHÂN TÍCH DỮ LIỆU ĐẶC TRƯNG LEGALQA (EDA SCRIPT)      ")
    print("=====================================================================")
    
    # 1. Đọc dữ liệu
    print("\n[BƯỚC 1] Đang tải các tập tin dữ liệu (ở cùng thư mục với file code)...")
    train_data = load_json(TRAIN_PATH)
    if train_data is None:
        print(f" -> Đang thử tải tệp dự phòng Vòng khởi động: {WARMUP_PATH}")
        train_data = load_json(WARMUP_PATH)

    corpus_data = load_corpus(CONTEXT_DIR)

    if not train_data:
        print("\n[LỖI] Không thể tiếp tục vì không có dữ liệu câu hỏi (train.json hoặc warmup.json).")
        print("-> Vui lòng đặt các tập tin dữ liệu pháp luật vào cùng thư mục với file script.")
        return

    # 2. Thực thi từng Mô-đun phân tích
    # Module 1: Phân tích Câu hỏi
    df_questions = analyze_questions(train_data)

    # Module 2: Phân tích Corpus
    df_corpus = analyze_corpus(corpus_data)

    # Module 3: Phân tích Mối liên kết (Retrieval)
    if not df_corpus.empty:
        analyze_retrieval_relations(train_data, corpus_data)

    # Module 4: Phân tích Câu trả lời
    analyze_answer_styles(train_data)

    print("\n=====================================================================")
    print("   QUÁ TRÌNH EDA HOÀN TẤT. VUI LÒNG KIỂM TRA CÁC PHÂN TÍCH TRONG THƯ MỤC:")
    print(f"   => Thư mục đầu ra: '{OUTPUT_DIR}/'")
    print(f"   - Biểu đồ câu hỏi: '{OUTPUT_DIR}/question_length_distribution.png'")
    print(f"   - Biểu đồ điều luật: '{OUTPUT_DIR}/article_length_distribution.png'")
    print(f"   - Tệp báo cáo chi tiết: '{OUTPUT_DIR}/eda_report.txt'")
    print("=====================================================================")

if __name__ == "__main__":
    main()
