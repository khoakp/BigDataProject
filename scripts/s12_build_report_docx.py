"""Build the course report from the supplied IUH DOCX template.

The template's two cover pages, logos, margins, section settings and footer are
retained. Its instructional/sample body is replaced with project evidence.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT.parent / "MauBaocaoBigData.docx"
OUTPUT = ROOT / "docs" / "Bao-cao-MovieGraph-IUH.docx"


def replace_text(paragraph, value: str) -> None:
    # Keep the paragraph's formatting, including cover alignment and spacing.
    if paragraph.runs:
        paragraph.runs[0].text = value
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(value)


def set_cell_shading(cell, fill: str = "EDEDED") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "Cập nhật mục lục trong Microsoft Word (F9)."
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for node in (begin, instr, separate, text, end):
        run._r.append(node)


def heading(doc, title: str, level: int = 1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.add_run(title)
    p.keep_with_next = True
    return p


def para(doc, text: str, *, bold: bool = False):
    p = doc.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(0.25)
    r = p.add_run(text)
    r.bold = bold
    return p


def new_page(doc, title: str, *, break_before: bool = True):
    if break_before:
        doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(16)
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(16)
    return p


def table(doc, caption: str, headers: list[str], rows: list[list[str]]):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(caption).bold = True
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        t.rows[0].cells[i].text = h
        set_cell_shading(t.rows[0].cells[i])
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = value
    for row in t.rows:
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(10)
                    r.font.color.rgb = RGBColor(0, 0, 0)
    doc.add_paragraph()
    return t


def build() -> None:
    doc = Document(TEMPLATE)
    # The first 51 paragraphs are the two cover pages. The rest of the supplied
    # file is instructional sample content, including broken bookmark fields.
    cover = doc.paragraphs[:51]
    for p in doc.paragraphs[51:]:
        p._element.getparent().remove(p._element)
    for t in list(doc.tables):
        t._element.getparent().remove(t._element)

    cover_edits = {
        0: "TRƯỜNG ĐẠI HỌC CÔNG NGHIỆP TP HỒ CHÍ MINH",
        7: "MOVIEGRAPH: GỢI Ý PHIM VỚI LIGHTGCN",
        17: "Người thực hiện: [HỌ TÊN – MSSV]",
        18: "[THÀNH VIÊN KHÁC – MSSV, NẾU CÓ]",
        19: "Lớp: [CẦN ĐIỀN]",
        20: "Khóa: [CẦN ĐIỀN]",
        21: "Giảng viên hướng dẫn: [CẦN ĐIỀN]",
        25: "TRƯỜNG ĐẠI HỌC CÔNG NGHIỆP TP HỒ CHÍ MINH",
        29: "",
        32: "MOVIEGRAPH: GỢI Ý PHIM VỚI LIGHTGCN",
        39: "Người thực hiện: [HỌ TÊN – MSSV]",
        40: "[THÀNH VIÊN KHÁC – MSSV, NẾU CÓ]",
        41: "Lớp: [CẦN ĐIỀN]",
        42: "Khóa: [CẦN ĐIỀN]",
        43: "Giảng viên hướng dẫn: [CẦN ĐIỀN]",
    }
    for index, value in cover_edits.items():
        replace_text(cover[index], value)
    # The cover is positioned with blank paragraphs. Shorten only those spacers
    # needed to keep each title and metadata block on its original single page.
    for index in (12, 13, 14, 15, 16, 35, 36, 37, 38, 46, 47, 48, 49):
        cover[index]._element.getparent().remove(cover[index]._element)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_after = Pt(6)
    for name, size in (("Heading 1", 14), ("Heading 2", 12)):
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True

    new_page(doc, "LỜI CẢM ƠN", break_before=False)
    para(doc, "[Người thực hiện tự viết lời cảm ơn và xác nhận trước khi nộp.]", bold=True)
    new_page(doc, "CAM ĐOAN")
    para(doc, "[Người thực hiện kiểm tra nội dung, bổ sung lời cam đoan, ngày tháng và ký tên trước khi nộp.]", bold=True)
    new_page(doc, "PHẦN ĐÁNH GIÁ CỦA GIẢNG VIÊN")
    para(doc, "........................................................................................................................")
    para(doc, "........................................................................................................................")
    para(doc, "........................................................................................................................")
    new_page(doc, "TÓM TẮT")
    para(doc, "Đề tài xây dựng pipeline MovieLens 32M, đồ thị user–movie cập nhật tăng dần, mô hình LightGCN và giao diện gợi ý phim. Trên canonical prefix 1M, giao thức REC-01 chia thời gian thành train/validation/test, chọn checkpoint bằng validation NDCG@20 và đánh giá test trên 403 warm users. LightGCN đạt Recall@20 = 0,128000 và NDCG@20 = 0,085161; Popularity đạt 0,128099 và 0,086897. Trong run này, Popularity nhỉnh hơn. Full Rebuild và Incremental Update có registry, edges và degrees tương đương chính xác trong 48 cặp benchmark. Với history tối đa 1M, Full nhanh hơn; tại điểm 2M và delta 1k, Incremental thắng 5/5 cặp, tỷ số median Full/Incremental bằng 1,156. Thử nghiệm S11 xác nhận batch 20 ratings làm graph version tăng 81→82, checkpoint giữ v81 và replay không tạo thay đổi. Kết quả mô tả đúng phạm vi dữ liệu và phần cứng đã đo, chưa đại diện cho huấn luyện trên toàn bộ 32M ratings.")
    new_page(doc, "MỤC LỤC")
    p = doc.add_paragraph()
    field(p, 'TOC \\o "1-2" \\h \\z \\u')
    new_page(doc, "DANH MỤC KÝ HIỆU VÀ CHỮ VIẾT TẮT")
    for term, meaning in [
        ("BPR", "Bayesian Personalized Ranking"),
        ("GCN", "Graph Convolutional Network"),
        ("LightGCN", "mô hình graph collaborative filtering giản lược"),
        ("NDCG@20", "Normalized Discounted Cumulative Gain ở Top-20"),
        ("Recall@20", "tỷ lệ positive items xuất hiện trong Top-20"),
        ("H, Δ", "số event history và số event trong batch mới"),
    ]:
        para(doc, f"{term}: {meaning}")
    new_page(doc, "DANH MỤC CÁC HÌNH VẼ VÀ BẢNG")
    para(doc, "Hình 1.1. Luồng dữ liệu và phiên bản của MovieGraph; Hình 1.2. Tỷ số paired Full/Incremental trong EXP-01.")
    para(doc, "Bảng 1.1. Chia thời gian REC-01; Bảng 1.2. Kết quả đánh giá; Bảng 1.3. Benchmark Full/Incremental.")

    new_page(doc, "HỆ THỐNG GỢI Ý PHIM MOVIEGRAPH")
    heading(doc, "1.1 Giới thiệu về bài toán")
    para(doc, "Hệ gợi ý phim phải học sở thích từ lịch sử ratings, đồng thời tiếp nhận rating mới mà không phá vỡ khả năng truy vết kết quả. MovieLens là bộ dữ liệu nghiên cứu do GroupLens công bố; bản 32M có 32.000.204 ratings [1], lịch sử phát triển và giới hạn sử dụng được mô tả trong [2]. Bài toán của đồ án là tổ chức dữ liệu theo thời gian, dựng đồ thị hai phía user–movie, huấn luyện mô hình xếp hạng và cung cấp Top-20 cho người dùng. Giao diện minh họa graph refresh sau batch mới và trình bày rõ model còn sử dụng checkpoint cũ.")
    para(doc, "Ba câu hỏi nghiên cứu: (RQ1) Incremental Update có tương đương Full Rebuild ở cùng prefix không; (RQ2) chi phí thay đổi theo history H, delta Δ và lịch materialize graph như thế nào; (RQ3) LightGCN được huấn luyện và đánh giá theo thời gian cho kết quả gì so với Popularity trên cùng cohort? Đồ án đánh giá mô hình trên prefix 1M, benchmark Engineering tới 2M; không công bố kết quả model toàn bộ 32M.")

    heading(doc, "1.2 Phân tích yêu cầu của bài toán")
    heading(doc, "1.2.1 Yêu cầu dữ liệu và tính đúng", 2)
    para(doc, "Canonical events phải sắp xác định theo timestamp, userId, movieId, source_row_id; node ID ổn định, mỗi positive user–movie pair sinh tối đa một cạnh khi rating ≥4,0. Registry, binary edges và degrees của Incremental phải khớp Full Rebuild. Batch ledger phải bảo đảm replay cùng ID/hash trả NO_OP, payload khác cùng ID bị từ chối. Normalized snapshot, checkpoint và giao diện phải mang version/hash để phát hiện dữ liệu hoặc mô hình cũ.")
    heading(doc, "1.2.2 Các phương pháp liên quan", 2)
    para(doc, "Lọc cộng tác item-based là một hướng kinh điển [3]. NGCF sử dụng high-order graph connectivity cho recommendation [4]; LightGCN giản lược phép lan truyền trên đồ thị bằng cách bỏ feature transformation và nonlinear activation [5]. BPR tối ưu thứ tự positive/negative implicit feedback [6]. Trong phạm vi đồ án, Popularity dựa trên số train users positive mỗi phim là baseline đơn giản, được chạy cùng candidate universe, mask train-seen và warm cohort. Không khẳng định Popularity là baseline mạnh nhất có thể.")
    heading(doc, "1.2.3 Hướng giải quyết", 2)
    para(doc, "Hệ thống chia canonical graph state và model snapshot thành hai vòng đời. Incremental phục vụ cập nhật trạng thái nhanh và kiểm correctness bằng Full Rebuild độc lập. LightGCN được train trên normalized train graph đóng băng; validation chọn checkpoint, test chỉ chạy sau lựa chọn. Feedback app được ghi vào graph branch riêng, không trộn vào tập test. Cách tách này cho phép đo đúng chi phí và độ cũ của từng thành phần.")

    heading(doc, "1.3 Phương pháp giải quyết bài toán")
    heading(doc, "1.3.1 Mô hình tổng quát", 2)
    para(doc, "Hình 1.1 biểu diễn luồng chính. Các khối và mũi tên bên dưới là đối tượng văn bản/bảng gốc của Word để có thể chỉnh sửa trong tài liệu.")
    flow = doc.add_table(rows=1, cols=7)
    flow.style = "Table Grid"
    flow.alignment = WD_TABLE_ALIGNMENT.CENTER
    for c, v in zip(flow.rows[0].cells, ["MovieLens 32M", "→", "Canonical Parquet", "→", "Graph state + versions", "→", "LightGCN / App"]):
        c.text = v
        for p in c.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.size = Pt(9)
    p = doc.add_paragraph("Hình 1.1. Luồng dữ liệu và phiên bản của MovieGraph (tự vẽ bằng thành phần Word).")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, "Canonical Parquet là nguồn đầu vào thống nhất. NodeRegistry gán ID, EdgeState giữ cặp positive duy nhất, DegreeState hỗ trợ chuẩn hóa; applied_batches giữ định danh/hash/watermark trong cùng DuckDB transaction. Full Rebuild từ cùng prefix làm đối chứng cho Incremental. Normalized sparse adjacency và node map được seal bằng hash; model checkpoint chứa version và config. Giao diện Streamlit đọc checkpoint, rồi nhận rating batch trên graph branch riêng.")
    heading(doc, "1.3.2 LightGCN, BPR và phiên bản đồ thị", 2)
    para(doc, "Với cạnh (u,i), trọng số đối xứng là Ã_ui = 1 / √(d_u d_i) (1.1). LightGCN lan truyền embedding qua ba tầng sparse graph và kết hợp các tầng [5]. Implementation dùng embedding 64 chiều, PyTorch sparse COO, không thêm feature transformation hoặc nonlinear activation. Mini-batch negative sampling tối ưu softplus[−(s_ui − s_uj)] (1.2), là biến thể xếp hạng dựa trên BPR [6]; code hiện tại không thêm hạng L2 của objective gốc.")
    para(doc, "canonical_graph_version chỉ tăng sau batch đã apply; model_graph_version gắn với snapshot tạo checkpoint. Nếu canonical version lớn hơn model version thì recommendation vẫn có thể dùng checkpoint cũ. Graph refresh không đồng nghĩa đã retrain. Đây là ranh giới quan trọng khi giải thích kết quả ứng dụng.")

    heading(doc, "1.4 Thực nghiệm")
    heading(doc, "1.4.1 Dữ liệu", 2)
    para(doc, "Nguồn MovieLens 32M [1] gồm ratings.csv với userId, movieId, rating, timestamp và movies.csv cho title/genres. D02 kiểm MD5 ratings.csv = cf12b74f9ad4b94a011f079e26d4270a, movies.csv = 0df90835c19151f9d819d0822e190797, profile đủ 32M và tạo prefix Parquet. movies.csv chỉ bổ sung thông tin sau ranking, không làm feature của LightGCN.")
    table(doc, "Bảng 1.1. Chia thời gian trên canonical prefix 1M", ["Tập", "Số events", "Cutoff / ghi chú"], [["Train", "802.988", "timestamp ≤ 840100781"], ["Validation", "102.046", "đến timestamp 841267862"], ["Test", "94.966", "sau cutoff validation"]])
    heading(doc, "1.4.2 Xử lý dữ liệu", 2)
    para(doc, "Canonical prefix 1M có SHA-256 f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae. Cutoff lấy theo 80% và 90% distinct timestamps, bảo đảm cùng timestamp không lọt qua ranh giới. Train graph version 81 có 14.388 nodes, 392.626 positive edges và 773 train candidate movies. Cold user/movie ngoài train catalog được loại có đếm; train-seen items bị mask khi tạo Top-20.")
    heading(doc, "1.4.3 Công nghệ sử dụng", 2)
    para(doc, "Python xử lý dữ liệu và ứng dụng; DuckDB đọc/ghi Parquet [7], giữ graph ledger; PyTorch huấn luyện sparse LightGCN; Streamlit cung cấp giao diện. Training REC-01 trên Kaggle Tesla T4; benchmark Engineering trên CPU Windows, DuckDB một thread.")
    heading(doc, "1.4.4 Cách đánh giá", 2)
    para(doc, "REC-01 chọn checkpoint 100/300/500 bằng validation NDCG@20, rồi đánh giá test độc lập trên warm users. Recall@20(u) = |Top20(u) ∩ Rel(u)| / |Rel(u)| (1.3). DCG@20(u) = Σ từ r=1 đến 20 rel_r/log₂(r+1); NDCG@20(u) = DCG@20/IDCG@20 (1.4). Mỗi metric lấy trung bình theo user. EXP-01 so wall time paired Full/Incremental, năm repetitions/case và chỉ dùng cặp có exact registry/edges/degrees. EXP-02 so R=1 với R=5 về chi phí normalization/export và staleness theo batch; không đo retrain.")

    heading(doc, "1.5 Kết quả đạt được")
    para(doc, "Checkpoint step 300 có validation NDCG@20 cao nhất trong ba checkpoint LightGCN. Test 403 warm users và 773 train candidates được so bằng cùng protocol. Bảng 1.2 cho thấy Popularity nhỉnh hơn 0,000099 Recall và 0,001736 NDCG trong run này; không có nhiều seeds hoặc kiểm định thống kê để kết luận tổng quát.")
    table(doc, "Bảng 1.2. REC-01 Top-20 trên warm cohort", ["Split", "Phương pháp", "Step", "Users", "Recall@20", "NDCG@20"], [["Validation", "LightGCN", "100", "482", "0,119490", "0,082000"], ["Validation", "LightGCN", "300", "482", "0,120954", "0,083543"], ["Validation", "LightGCN", "500", "482", "0,119489", "0,082913"], ["Validation", "Popularity", "—", "482", "0,120696", "0,085337"], ["Test", "LightGCN", "300", "403", "0,128000", "0,085161"], ["Test", "Popularity", "—", "403", "0,128099", "0,086897"]])
    para(doc, "Test có 44.865 future positive pairs, nhưng chỉ 2.069 eligible warm pairs. Có 42.176 pairs bị loại vì cold user và 620 vì cold movie. Vì vậy metric mô tả cohort này, không đại diện toàn bộ users hay catalog của 32M.")
    table(doc, "Bảng 1.3. EXP-01 median wall time (giây)", ["History", "Delta", "Full", "Incremental", "Full/Inc", "Inc thắng"], [["100k", "1k", "0,0911", "0,1624", "0,559", "0/5"], ["500k", "1k", "0,1748", "0,2219", "0,779", "0/5"], ["1M", "1k", "0,2865", "0,2970", "0,965", "0/5"], ["2M riêng", "1k", "0,4908", "0,4321", "1,156", "5/5"], ["2M riêng", "10k", "0,4951", "0,4941", "0,999", "2/5"]])
    plot_path = ROOT / "docs" / "report-assets" / "exp01_speedup.png"
    if plot_path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(plot_path), width=Inches(5.3))
        p = doc.add_paragraph("Hình 1.2. Tỷ số paired Full/Incremental theo history và delta (số đo EXP-01).")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, "Tất cả 48 cặp benchmark đã audit có registry, edges và degrees khớp. Core H≤1M Full nhanh hơn. Ở run 2M riêng, Incremental thắng 5/5 cặp khi Δ=1k, còn Δ=10k gần hòa. Đây là crossover quan sát giữa các điểm đo, không xác định ngưỡng trên toàn bộ miền H/Δ. Cache hệ điều hành không được reset; prebuild history của Incremental nằm ngoài timer và được báo riêng.")
    para(doc, "EXP-02 trên prefix 100k, mười batch 10k: R=1 có cumulative median apply+materialize+export 5,283 s, R=5 là 1,648 s. Tại v5/v10 normalized snapshot giống nhau (max_abs_diff=0), nhưng R=5 chỉ có hai snapshot và model graph cũ tối đa bốn batch/40k events giữa hai lần refresh. Đây là trade-off chi phí–độ cũ graph, không phải chất lượng recommendation.")
    para(doc, "S11 trên 802.988 train events + 20 feedback events xác nhận Full và Incremental khớp từng dòng: 14.388 registry nodes, 392.636 edges, 14.388 degree rows. Mười ratings 5,0 thêm mười cạnh; mười ratings 2,5 không thêm cạnh. Replay trả NO_OP, graph version 81→82 còn model graph version giữ 81. Top-20 không đổi vì checkpoint chưa refresh. 60 local tests PASS gồm temporal leakage, UI, crash recovery và tamper paths.")

    heading(doc, "1.6 Kết luận")
    para(doc, "RQ1 được hỗ trợ bằng exact equivalence trên toy, 100k, train graph REC-01 và probe feedback 1M+20. RQ2 cho thấy Incremental không luôn nhanh hơn Full: lợi thế phụ thuộc H, Δ và output contract; deferred normalization giảm số snapshots nhưng làm model graph cũ theo batch. RQ3 xác nhận pipeline LightGCN đã train thật và đánh giá Top-20 tái lập được; Popularity nhỉnh hơn trong một run REC-01.")
    para(doc, "Giới hạn: chỉ một seed, ba candidate checkpoint steps và baseline Popularity; model chỉ train trên prefix 1M, warm catalog 773 movies; benchmark tối đa H=2M trên một máy, cache không kiểm soát. Feedback graph không cập nhật checkpoint. Hướng tiếp theo là REC-02 so stale, propagation-only và trained refresh trên cùng graph version; thêm ItemKNN, nhiều seeds và đo inference ở scale lớn hơn. Các hướng này chưa được tính là kết quả đã hoàn thành.")

    new_page(doc, "LÀM VIỆC NHÓM")
    para(doc, "[Cần bổ sung tên thành viên, phân công, số buổi và tổng giờ họp thực tế trước khi nộp. Không sử dụng thông tin ví dụ của mẫu.]", bold=True)
    new_page(doc, "TÀI LIỆU THAM KHẢO")
    references = [
        "[1] GroupLens Research. MovieLens 32M Dataset, README. https://files.grouplens.org/datasets/movielens/ml-32m-README.html",
        "[2] Harper, F. M. và Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5(4). https://doi.org/10.1145/2827872",
        "[3] Sarwar, B. et al. (2001). Item-Based Collaborative Filtering Recommendation Algorithms. WWW 2001, 285–295. https://doi.org/10.1145/371920.372071",
        "[4] Wang, X. et al. (2019). Neural Graph Collaborative Filtering. SIGIR 2019, 165–174. https://doi.org/10.1145/3331184.3331267",
        "[5] He, X. et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020. https://arxiv.org/abs/2002.02126",
        "[6] Rendle, S. et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009. https://proceedings.mlr.press/r7/rendle09a.html",
        "[7] DuckDB. Reading and Writing Parquet Files. https://duckdb.org/docs/data/parquet",
    ]
    for ref in references:
        para(doc, ref)
    new_page(doc, "PHỤ LỤC: TÁI LẬP")
    para(doc, "Mã nguồn, protocol, manifest hash và lệnh chạy nằm trong README.md, docs/s01-evaluation-spec.md, docs/s03-kaggle-workflow.md, docs/s04-rec01-results.md, docs/s06-exp01-audit.md và docs/s11-integration-scaling.md. REC-01 checkpoint seal và benchmark raw CSV được đóng gói riêng theo manifest S14; raw MovieLens 32M chịu điều kiện sử dụng của GroupLens. Các giá trị trong tài liệu này là số đo từ các artifact đã seal, không phải phép đo mới của bản DOCX.")
    para(doc, "Các ô bìa, cam đoan và làm việc nhóm cần người thực hiện xác nhận. Không ký hoặc ghi thông tin thay chủ dự án.")

    settings = doc.settings.element
    update = OxmlElement("w:updateFields")
    update.set(qn("w:val"), "true")
    settings.append(update)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
