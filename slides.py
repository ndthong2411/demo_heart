"""Các trang chiếu. Mỗi trang là một hàm vẽ nội dung + ghi chú để nói."""
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from common import (CODES, EVAL_CODES, GROUP_COLOR, NEURAL, OUR, PUBLISHED_SEQ, card, context, curves, deltas, fig_layout,
                    flow, fmt, forecast, group_legend, html, kpi, macro_table, metrics, name, ranking_bar, ul, vn, why, wrap)

CFG = {"displayModeBar": False}


def head(kicker, title, lead=None):
    html(f'<div class="kicker">{kicker}</div><div class="slide-title">{title}</div>')
    if lead:
        html(f'<div class="lead">{lead}</div>')


def ci(task, ev, b, metric="AUPRC"):
    r = deltas(task, ev, metric).set_index("b").loc[b]
    return r["value"], r["lo"], r["hi"]


def pts(x):
    """AP chênh lệch đổi sang điểm trên thang 100."""
    return f"{x * 100:+.2f}".replace(".", ",")


# =============================================================== 1. Mở đầu
def s_intro():
    c = context()["corpus"]
    head("SeqCVD · báo cáo tiến độ · 02/11/2026",
         "Đọc lịch sử xét nghiệm để dự báo bệnh tim mạch ở lần khám sau",
         "Mô hình học sâu SeqLab đọc toàn bộ diễn biến xét nghiệm, sinh hiệu và tiền sử chẩn đoán của bệnh nhân, "
         "rồi ước lượng khả năng lượt khám tới sẽ ghi thêm một chẩn đoán tim mạch mới.")
    k = st.columns(4)
    with k[0]:
        kpi(vn(c["patients"]), "bệnh nhân")
    with k[1]:
        kpi(vn(c["visits"]), "lượt khám")
    with k[2]:
        kpi(f"{c['events'] / 1e3:.0f} nghìn", "sự kiện theo thời gian")
    with k[3]:
        kpi("2020 – 2025", "6 năm dữ liệu")
    st.write("")
    a, b = st.columns(2)
    with a:
        card("Mục 1 · Dự báo nguy cơ tim mạch từ lịch sử khám",
             "<p>Bài <b>T2</b>: đứng ở cuối lượt khám hiện tại, đoán xem lượt sau sẽ có thêm chẩn đoán tim mạch nào "
             "chưa từng ghi.</p>", "accent", "Đề cương")
    with b:
        card("Mục 2 · Học biểu diễn chuỗi xét nghiệm",
             "<p><b>SeqLab</b>: encoder Transformer đọc từng lần đo cùng thời điểm đo, được học trước trên dữ liệu "
             "không cần nhãn.</p>", "accent", "Đề cương")
    st.write("")
    flow([("1", "Dữ liệu", "có gì, dùng gì, xử lý ra sao"), ("2", "Bài toán T2", "đoán gì, khi nào"),
          ("3", "SeqLab", "kiến trúc và lý do"), ("4", "Đánh giá", "chia dữ liệu, chỉ số"),
          ("5", "Kết quả", "so với HGB và SOTA"), ("6", "Tiếp theo", "T1, T3, hạn chế")])


N_INTRO = """Hôm nay em báo cáo phần SeqCVD, tức là phần học sâu trên dữ liệu chuỗi trong luận án.

Đề cương có hai mục học sâu: một là dự báo nguy cơ tim mạch từ lịch sử khám, hai là học biểu diễn cho chuỗi xét nghiệm. Em gộp cả hai vào một bài toán cụ thể gọi là T2: nhìn hồ sơ tới hết lượt khám này, đoán lượt khám sau bệnh nhân có được ghi thêm chẩn đoán tim mạch mới hay không.

Mô hình em xây là SeqLab. Nó vừa là câu trả lời cho mục 2 (cách biểu diễn chuỗi xét nghiệm), vừa là công cụ để làm mục 1.

Em sẽ đi theo thứ tự: dữ liệu, bài toán, mô hình, cách đánh giá, rồi mới tới kết quả. Cuối buổi em nhắc nhanh hai bài phụ T1, T3."""


# =============================================================== 2. Dữ liệu gốc
def s_raw():
    ctx = context()
    raw = ctx["raw"]
    B, C = raw["sources"]
    head("Dữ liệu · 1/4", "Dữ liệu gốc: hồ sơ xét nghiệm của một bệnh viện, 2020–2025",
         "Hai bộ file xét nghiệm có cùng cấu trúc cột. Ngoài ra có bảng mã chẩn đoán theo lượt và bảng chẩn đoán hình ảnh.")
    a, b, c = st.columns([1, 1, 1.25])
    with a:
        card("Bảng B · file tổng hợp",
             f"<p><b>{vn(B['rows'])}</b> dòng<br>{vn(B['patients'])} bệnh nhân · {vn(B['visits'])} lượt<br>1 file</p>",
             tag="Nguồn 1")
    with b:
        card("Bảng C · file theo từng bệnh nhân",
             f"<p><b>{vn(C['rows'])}</b> dòng<br>{vn(C['patients'])} bệnh nhân · {vn(C['visits'])} lượt<br>"
             f"{vn(C['files'])} file</p>", tag="Nguồn 2")
    with c:
        card("Hai nguồn chồng lên nhau",
             f"<p>Gộp lại có <b>{vn(raw['rows_before_dedup'])}</b> dòng, trong đó <b>{vn(raw['duplicate_rows_removed'])}</b> "
             f"dòng trùng. Sau khi bỏ trùng còn {vn(raw['rows_after_dedup'])} kết quả của {vn(raw['union_patients'])} "
             "bệnh nhân.</p>", tag="Sau khi gộp")
    st.write("")
    l, r = st.columns([1.15, 1])
    with l:
        html("""<div class="card"><h4>Mỗi dòng xét nghiệm chứa gì</h4>
<table class="simple">
<tr><th>Nhóm</th><th>Cột</th></tr>
<tr><td>Định danh, thời gian</td><td><code>mabn</code>, <code>mavaovien</code>, <code>maql</code>, <code>id_*</code>, <code>ddmmyyyy</code></td></tr>
<tr><td>Xét nghiệm</td><td><code>maxn</code>, <code>tenxn</code>, <code>idxetnghiem</code>, <code>ketqua</code></td></tr>
<tr><td>Sinh hiệu</td><td><code>mach</code>, <code>huyetap</code>, <code>nhietdo</code>, <code>cannang</code>, <code>chieucao</code></td></tr>
<tr><td>Người bệnh</td><td><code>tuoi</code>, <code>phai</code>, <code>loaibenhan</code></td></tr>
<tr><td>Chẩn đoán</td><td><code>maicd</code> (mã ICD), <code>chandoan</code> (văn bản)</td></tr>
</table></div>""")
    with r:
        y = ctx["corpus"]["lab_years"]
        fig = go.Figure(go.Bar(x=list(y), y=list(y.values()), marker=dict(color="#2a78d6", cornerradius=4),
                               text=[f"{v / 1e3:.0f}k" for v in y.values()], textposition="outside", cliponaxis=False,
                               hovertemplate="Năm %{x}: %{y:,} kết quả<extra></extra>"))
        fig.update_yaxes(title="Số kết quả xét nghiệm", range=[0, max(y.values()) * 1.18])
        html('<div style="font-weight:700;margin-bottom:-.4rem">Số kết quả xét nghiệm theo năm</div>')
        st.plotly_chart(fig_layout(fig, 290), config=CFG, width="stretch")
    why("<p>Dữ liệu trải đều qua 6 năm, nên có thể làm cả hai kiểu kiểm tra: chia theo bệnh nhân, và học trên năm cũ "
        "rồi thử trên năm mới. Kiểu thứ hai mới giống cách mô hình được dùng thật.</p>")


N_RAW = """Dữ liệu gốc gồm hai bộ file xét nghiệm. Bảng B là một file lớn, bảng C là hàng nghìn file nhỏ, mỗi file một bệnh nhân. Hai bộ có cùng cột nhưng chồng lên nhau khá nhiều: gần 290 nghìn dòng bị trùng.

Mỗi dòng là một kết quả xét nghiệm, kèm sinh hiệu đo cùng lúc, tuổi, giới, loại bệnh án, mã ICD và một ô chẩn đoán dạng chữ. Bên cạnh đó có bảng mã chẩn đoán theo lượt và bảng chẩn đoán hình ảnh.

Số kết quả mỗi năm khá đều, khoảng 110 đến 150 nghìn. Điều này quan trọng vì em muốn kiểm tra mô hình theo thời gian: học trên các năm trước, thử trên năm sau."""


# =============================================================== 3. Chọn gì, bỏ gì
def s_select():
    k = context()["corpus"]["by_kind"]
    head("Dữ liệu · 2/4", "Chỉ giữ thứ bác sĩ có trong tay lúc dự báo, bỏ mọi thứ chứa sẵn đáp án",
         "Nhiều cột nếu đưa vào sẽ làm điểm số đẹp lên, nhưng là đẹp giả vì chúng tiết lộ chẩn đoán.")
    a, b = st.columns(2)
    with a:
        card("Giữ lại", ul([
            f"<b>Kết quả xét nghiệm</b>: {k[0]['keys']} loại, {vn(k[0]['events'])} kết quả. Đây là nguồn chính.",
            f"<b>Sinh hiệu</b>: {k[1]['keys']} loại (mạch, huyết áp, nhiệt độ, cân nặng, chiều cao...).",
            f"<b>Mã ICD của các lượt trước</b>: {k[2]['keys']} mã khác nhau. Tiền sử bệnh là tín hiệu mạnh.",
            f"<b>Lượt khám</b> và loại đợt điều trị ({k[3]['keys']} loại), lấy theo đợt tại thời điểm dự báo.",
            "<b>Tuổi lúc khám</b> và <b>giới</b>.",
        ]), "keep")
    with b:
        card("Bỏ đi", ul([
            "<b>Ô chẩn đoán dạng chữ</b> (<code>chandoan</code>) và <b>kết luận chẩn đoán hình ảnh</b>: "
            "viết thẳng ra chẩn đoán.",
            "<b>Mã ICD của lượt cần đoán</b>: đây chính là đáp án.",
            "<b>Loại bệnh án ghi theo lượt</b>: lộ việc bệnh nhân về sau có nhập viện hay không.",
            "<b>Bảng xét nghiệm đã lọc theo ICD</b> (bảng A): dòng nào được giữ lại phụ thuộc vào chẩn đoán, "
            "tức là phụ thuộc nhãn.",
        ]), "drop")
    why(ul([
        "Bài toán là dự báo, nên đầu vào chỉ được chứa thông tin có trước thời điểm dự báo.",
        "Một lần chạy đầu có lỗi đúng kiểu này: token loại lượt của lượt hiện tại lộ việc sẽ nhập nội trú. "
        "Em đã bỏ token đó và chạy lại; kết quả gần như không đổi, nhưng từ đó mọi cột đều được rà theo tiêu chí trên.",
    ]))


N_SELECT = """Nguyên tắc chọn dữ liệu của em rất đơn giản: thứ gì bác sĩ chưa có trong tay lúc dự báo thì không được đưa vào.

Giữ lại: xét nghiệm, sinh hiệu, mã chẩn đoán của các lượt trước, thông tin lượt khám, tuổi và giới.

Bỏ đi: ô chẩn đoán dạng chữ và kết luận chẩn đoán hình ảnh, vì nó viết thẳng chẩn đoán. Loại bệnh án ghi theo lượt cũng bỏ, vì nhìn vào là biết bệnh nhân sau đó có nhập viện. Còn bảng A thì đã bị lọc theo ICD từ trước, nên bản thân việc một dòng có mặt hay không đã chứa thông tin về nhãn.

Em có một lần bị dính lỗi kiểu này ở bản đầu, đã sửa và chạy lại, kết quả gần như không đổi. Từ đó em rà mọi cột theo đúng tiêu chí này."""


# =============================================================== 4. Xử lý
def s_process():
    ctx = context()
    raw, c = ctx["raw"], ctx["corpus"]
    head("Dữ liệu · 3/4", "Từ hơn 1 triệu dòng thô thành một dòng thời gian cho mỗi bệnh nhân",
         "Mỗi bệnh nhân được biểu diễn bằng một chuỗi sự kiện sắp theo thời gian. Mỗi sự kiện là một lần đo, "
         "một mã chẩn đoán hoặc một lượt khám.")
    flow([
        ("Bước 1", "Gộp B + C", f"{vn(raw['rows_before_dedup'])} dòng"),
        ("Bước 2", "Bỏ trùng", f"theo mã phiếu + loại xét nghiệm, bỏ {vn(raw['duplicate_rows_removed'])} dòng"),
        ("Bước 3", "Gắn thời gian", "12 chữ số đầu của ID là năm-tháng-ngày-giờ-phút-giây"),
        ("Bước 4", "Tuổi lúc khám", "file ghi tuổi lúc trích xuất (19/03/2025), phải trừ ngược về ngày khám"),
        ("Bước 5", "Ghép sự kiện", "xét nghiệm + sinh hiệu + mã ICD + lượt khám"),
        ("Kết quả", f"{vn(c['events'])} sự kiện", f"{vn(c['patients'])} bệnh nhân · {vn(c['visits'])} lượt"),
    ], highlight=(5,))
    st.write("")
    l, r = st.columns([1, 1.1])
    with l:
        kinds = c["by_kind"]
        lab = [k["name"] for k in kinds][::-1]
        val = [k["events"] for k in kinds][::-1]
        fig = go.Figure(go.Bar(x=val, y=lab, orientation="h", marker=dict(color="#2a78d6", cornerradius=4),
                               text=[vn(v) for v in val], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:,}<extra></extra>"))
        fig.update_xaxes(type="log", title="Số sự kiện (thang log)", range=[4, 6.3])
        html('<div style="font-weight:700;margin-bottom:-.4rem">Thành phần của kho sự kiện</div>')
        st.plotly_chart(fig_layout(fig, 250), config=CFG, width="stretch")
    with r:
        html("""<div class="card"><h4>Ví dụ một đoạn chuỗi (bệnh nhân giả định)</h4>
<table class="simple">
<tr><th>Thời điểm</th><th>Loại</th><th>Mã</th><th style="text-align:right">Giá trị</th></tr>
<tr><td>03/2023 08:12</td><td>Lượt khám</td><td>ngoại trú</td><td class="num">–</td></tr>
<tr><td>03/2023 08:40</td><td>Sinh hiệu</td><td>huyết áp tâm thu</td><td class="num">152</td></tr>
<tr><td>03/2023 09:05</td><td>Xét nghiệm</td><td>Creatinin</td><td class="num">1,4</td></tr>
<tr><td>03/2023 09:05</td><td>Xét nghiệm</td><td>NT-proBNP</td><td class="num">820</td></tr>
<tr><td>03/2023</td><td>Mã ICD</td><td>I10</td><td class="num">–</td></tr>
<tr><td>05/2023 07:55</td><td>Lượt khám</td><td>nội trú</td><td class="num">–</td></tr>
<tr><td colspan="4" style="color:#7a7f8a">... tiếp tục tới thời điểm dự báo</td></tr>
</table></div>""")
    why(ul([
        "Không ép dữ liệu vào lưới thời gian cố định và không điền giá trị thiếu: lần đo nào có thật thì mới có sự kiện.",
        "Giá trị được chuẩn hoá theo phân bố của riêng phần dữ liệu học, từng loại xét nghiệm một, để không lấy "
        "thông tin từ phần kiểm tra.",
    ]))


N_PROCESS = """Phần xử lý có mấy bước chính.

Gộp hai nguồn rồi bỏ trùng theo mã phiếu và loại xét nghiệm, còn khoảng 800 nghìn kết quả.

Thời gian: em lấy từ 12 chữ số đầu của ID bệnh viện, vì nó mã hoá năm-tháng-ngày-giờ-phút-giây. Nhờ vậy biết được giờ lấy mẫu chứ không chỉ ngày.

Tuổi: cột tuổi trong file là tuổi tại lúc trích xuất dữ liệu năm 2025, nên phải trừ ngược về ngày khám, nếu không bệnh nhân năm 2020 sẽ bị già thêm 5 tuổi.

Cuối cùng ghép xét nghiệm, sinh hiệu, mã ICD và lượt khám thành một chuỗi duy nhất cho mỗi bệnh nhân, tổng cộng khoảng 920 nghìn sự kiện. Xét nghiệm chiếm phần lớn.

Điểm quan trọng: em không chia lưới thời gian, không điền giá trị thiếu. Đo lúc nào thì có sự kiện lúc đó."""


# =============================================================== 5. Mã ICD mục tiêu
ICD_GROUPS = [
    ("Tăng huyết áp", [("I10", "Tăng huyết áp vô căn (nguyên phát)")]),
    ("Bệnh mạch vành", [("I20", "Cơn đau thắt ngực"), ("I21", "Nhồi máu cơ tim cấp"),
                        ("I25", "Bệnh tim thiếu máu cục bộ mạn")]),
    ("Suy tim", [("I50", "Suy tim")]),
    ("Nhịp tim", [("I48", "Rung nhĩ và cuồng nhĩ"), ("I46", "Ngừng tim")]),
]


def s_icd():
    t2 = context()["tasks"]["T2"]
    new, n = t2["new_by_code"], t2["samples"]
    head("Dữ liệu · 4/4", "Mục tiêu dự báo: 7 mã ICD-10 tim mạch",
         "Mã ICD được gộp ở mức 3 ký tự, ví dụ I21.0 đến I21.9 đều tính là I21. Mỗi mã là một câu hỏi có/không riêng: "
         "lượt sau có ghi mã này lần đầu hay không.")
    cols = st.columns([1, 2.4, 1, 1.7])
    for col, (grp, codes) in zip(cols, ICD_GROUPS):
        tiles = ""
        for code, nm in codes:
            main = code in EVAL_CODES
            badge = ('<span class="pill b">chỉ số chính</span>' if main
                     else '<span class="pill">quá ít ca, không tính</span>')
            tiles += (f'<div class="card {"accent" if main else ""}" style="flex:1 1 120px;padding:.8rem 1rem">'
                      f'<div style="font-size:1.6rem;font-weight:800;color:{"#1d5fae" if main else "#7a7f8a"}">{code}</div>'
                      f'<div style="font-weight:600;color:#16181d;margin:.1rem 0 .35rem 0">{nm}</div>'
                      f'<div style="color:#4a4f5a;font-size:.92rem">{vn(new[code])} ca mới · '
                      f'{fmt(new[code] / n * 100, 1)}%</div><div style="margin-top:.35rem">{badge}</div></div>')
        with col:
            html(f'<div class="kicker" style="color:#4a4f5a">{grp}</div>'
                 f'<div style="display:flex;gap:.6rem;flex-wrap:wrap;margin-top:.5rem">{tiles}</div>')
    html(f'<div class="note">Số ca mới đếm trên {vn(n)} cặp lượt của T2. "Mới" nghĩa là mã chưa từng xuất hiện ở mọi lượt '
         'trước và lượt hiện tại của bệnh nhân.</div>')
    st.write("")
    a, b = st.columns(2)
    with a:
        card("Vì sao là 7 mã này", "<p>Dùng chung bộ mã với bài RAD để các bài so sánh được với nhau. Đây là 7 mã chương IX "
             "(bệnh hệ tuần hoàn) có nhiều lượt nhập viện nhất trong cohort RAD, từ 114 lượt (I46) tới 2.072 lượt (I10), "
             "phủ bốn nhóm bệnh tim mạch chính.</p>")
    with b:
        card("Vì sao chỉ số chính chỉ tính 5 mã", "<p>Quy tắc đặt từ trước: một mã cần ít nhất 50 ca mới tổng cộng và ít "
             "nhất 10 ca mới trong mỗi kỳ test theo năm. I48 (55 ca) và I46 (43 ca) không đạt. Với vài ca mỗi kỳ, AP của "
             "chúng dao động quá mạnh, kéo lệch điểm trung bình. Mô hình vẫn dự báo cả 7 mã.</p>")
    why("<p>Gộp ở mức 3 ký tự để mỗi mã đủ ca mà vẫn có ý nghĩa lâm sàng rõ. Chia nhỏ hơn, ví dụ nhồi máu thành vách "
        "trước, vách dưới, thì mỗi nhóm chỉ còn vài chục ca, không đủ để học và đánh giá.</p>")


N_ICD = """Trước khi vào bài toán, em nói rõ mô hình đoán những gì.

Có 7 mã ICD-10 tim mạch, chia bốn nhóm: tăng huyết áp; bệnh mạch vành gồm đau thắt ngực, nhồi máu cơ tim cấp và bệnh tim thiếu máu mạn; suy tim; và rối loạn nhịp gồm rung nhĩ và ngừng tim.

Mã được gộp ở mức 3 ký tự, ví dụ mọi loại nhồi máu cơ tim cấp đều tính là I21. Chia nhỏ hơn thì mỗi nhóm quá ít ca.

Em dùng chung bộ mã với bài RAD, để các bài so sánh được. Đây là 7 mã tim mạch có nhiều lượt nhập viện nhất trong cohort.

Mô hình dự báo cả 7 mã, nhưng điểm chính chỉ tính 5. Rung nhĩ và ngừng tim chỉ có 55 và 43 ca mới, chia ra 4 năm test thì mỗi năm vài ca, AP sẽ nhảy lung tung. Quy tắc này em đặt từ trước khi xem kết quả."""


# =============================================================== 6. Bài toán T2
def timeline_fig():
    fig = go.Figure()
    visits = [(0, 3, "Lượt 1", "I10"), (104, 110, "Lượt 2 (hiện tại)", "I25"), (156, 160, "Lượt 3 (+40 ngày)", "I20")]
    for x0, x1, lab, code in visits:
        future = x0 > 120
        fig.add_shape(type="rect", x0=x0, x1=x1 + 6, y0=0.25, y1=0.75, line=dict(width=0),
                      fillcolor="#fde3d7" if future else "#dbe9fb")
        fig.add_annotation(x=(x0 + x1 + 6) / 2, y=1.05, text=f"<b>{lab}</b>", showarrow=False, font=dict(size=13))
        fig.add_annotation(x=(x0 + x1 + 6) / 2, y=0.5, text=f"ghi <b>{code}</b>", showarrow=False,
                           font=dict(size=13, color="#c2410c" if future else "#1d5fae"))
    import random
    random.seed(3)
    xs = [random.uniform(0, 3) for _ in range(6)] + [random.uniform(104, 116) for _ in range(9)]
    fig.add_trace(go.Scatter(x=xs, y=[random.uniform(-0.25, 0.05) for _ in xs], mode="markers",
                             marker=dict(size=8, color="#2a78d6", line=dict(width=2, color="white")),
                             hoverinfo="skip", showlegend=False))
    fig.add_annotation(x=58, y=-0.12, text="xét nghiệm, sinh hiệu (chấm xanh)", showarrow=False,
                       font=dict(size=12, color="#4a4f5a"))
    fig.add_vline(x=124, line=dict(color="#16181d", width=2, dash="dash"))
    fig.add_annotation(x=124, y=1.38, text="<b>Thời điểm dự báo</b>", showarrow=False, bgcolor="white",
                       font=dict(size=13))
    fig.add_annotation(x=60, y=1.38, text="← mô hình được xem", showarrow=False, font=dict(color="#1d5fae"))
    fig.add_annotation(x=152, y=1.38, text="phải đoán →", showarrow=False, font=dict(color="#c2410c"))
    fig.update_xaxes(range=[-5, 172], title="ngày", showgrid=False)
    fig.update_yaxes(visible=False, range=[-0.35, 1.5])
    return fig_layout(fig, 250, showlegend=False)


def s_task():
    t2 = context()["tasks"]["T2"]
    head("Bài toán", "T2: lượt khám sau sẽ ghi thêm chẩn đoán tim mạch nào chưa từng có?",
         "Mỗi mẫu là một cặp lượt khám liên tiếp của cùng một bệnh nhân. Lượt hiện tại và mọi thứ trước đó là đầu vào; "
         "lượt sau cho đáp án.")
    st.plotly_chart(timeline_fig(), config=CFG, width="stretch")
    l, r = st.columns([1, 1.15])
    with l:
        html("""<div class="card"><h4>Nhãn của ví dụ trên</h4>
<table class="simple"><tr><th>Mã</th><th>Đã có trước?</th><th>Lượt sau ghi?</th><th>Nhãn</th></tr>
<tr><td>I10</td><td>có</td><td>có</td><td><b>0</b> (không còn mới)</td></tr>
<tr><td>I25</td><td>có</td><td>không</td><td><b>0</b></td></tr>
<tr><td>I20</td><td>chưa</td><td>có</td><td><b style="color:#c2410c">1</b> (mã mới)</td></tr>
<tr><td>I50, I21, ...</td><td>chưa</td><td>không</td><td><b>0</b></td></tr></table>
<p style="margin-top:.6rem">Một lượt có thể có nhiều mã mới cùng lúc, nên mỗi mã là một câu hỏi có/không riêng.</p></div>""")
    with r:
        codes = list(t2["new_by_code"])
        n = [t2["new_by_code"][c] for c in codes]
        cols = ["#2a78d6" if c in EVAL_CODES else "#c9ccd2" for c in codes]
        fig = go.Figure(go.Bar(x=codes, y=n, customdata=[CODES[c] for c in codes],
                               marker=dict(color=cols, cornerradius=4), cliponaxis=False,
                               text=[f"{v} ({fmt(v / t2['samples'] * 100, 1)}%)" for v in n], textposition="outside",
                               hovertemplate="%{x} · %{customdata}<br>%{y} ca mới<extra></extra>"))
        fig.update_yaxes(title="Số ca có mã mới", range=[0, max(n) * 1.22])
        html(f'<div style="font-weight:700;margin-bottom:-.4rem">Số ca dương trên {vn(t2["samples"])} cặp lượt '
             f'({vn(t2["patients"])} bệnh nhân)</div>')
        st.plotly_chart(fig_layout(fig, 270), config=CFG, width="stretch")
    why(ul([
        "Đây là câu hỏi thực tế khi bệnh nhân tái khám, và trả lời trực tiếp mục 1 của đề cương.",
        "Nhãn rõ ràng: mã nào được ghi ở lượt sau mà trước đó chưa từng có. Hai lượt cách nhau ít nhất 1 ngày "
        "(trung vị khoảng 37 ngày).",
        "I48 và I46 quá ít ca (55 và 43) nên vẫn được dự báo nhưng không tính vào điểm trung bình chính; "
        "chỉ số chính lấy trung bình trên 5 mã màu xanh.",
    ]))


N_TASK = """Đây là bài toán chính. Nhìn ví dụ: bệnh nhân này lượt 1 được ghi tăng huyết áp, lượt 2 ghi thêm bệnh tim thiếu máu mạn. Em đứng ở cuối lượt 2, mô hình được xem toàn bộ những gì xảy ra trước đường đứt.

Câu hỏi là: lượt sau, cụ thể ở đây là 40 ngày sau, sẽ có thêm mã tim mạch nào chưa từng có? Ở đây là I20, đau thắt ngực. Còn I10 và I25 đã có rồi nên không tính là mới.

Em có 12.514 cặp lượt như vậy từ gần 2.500 bệnh nhân. Ca dương rất hiếm: mỗi mã chỉ khoảng 2 đến 7%. Hai mã I48 và I46 chỉ có vài chục ca nên em không đưa vào điểm trung bình, để con số chính không bị nhiễu.

Một điểm cần nói rõ: nhãn ở đây là mã được ghi mới, chứ không phải ngày khởi phát bệnh. Và chỉ đánh giá được bệnh nhân có quay lại."""


# =============================================================== 6. Vì sao đọc chuỗi
def s_why_seq():
    head("Ý tưởng", "Vì sao đọc chuỗi, thay vì tóm tắt lịch sử thành một bảng?",
         "Cách quen thuộc là tính min, max, trung bình, giá trị cuối cho mỗi xét nghiệm rồi đưa vào mô hình cây (HGB). "
         "Cách này nhanh và mạnh, nhưng làm mất thứ tự và nhịp thời gian.")
    l, r = st.columns([1.2, 1])
    with l:
        x = ["Lần 1", "Lần 2", "Lần 3", "Lần 4"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[1.0, 1.2, 1.6, 1.4], name="Bệnh nhân A: tăng rồi đang giảm",
                                 mode="lines+markers", line=dict(color="#2a78d6", width=2.5), marker=dict(size=9)))
        fig.add_trace(go.Scatter(x=x, y=[1.6, 1.2, 1.0, 1.4], name="Bệnh nhân B: giảm rồi đang tăng",
                                 mode="lines+markers", line=dict(color="#eb6834", width=2.5), marker=dict(size=9)))
        fig.update_yaxes(title="Creatinin (mg/dL)", range=[0.8, 1.8])
        html('<div style="font-weight:700;margin-bottom:-.4rem">Hai diễn biến creatinin (ví dụ giả định)</div>')
        st.plotly_chart(fig_layout(fig, 300), config=CFG, width="stretch")
        html('<div class="note">Cả hai có cùng <b>min 1,0 · max 1,6 · trung bình 1,3 · giá trị cuối 1,4 · 4 lần đo</b>. '
             'Bảng tóm tắt thấy hai người này giống hệt nhau, dù xu hướng ngược chiều.</div>')
    with r:
        card("Ba khó khăn của dữ liệu này", ul([
            "<b>Đo không đều</b>: mỗi bệnh nhân, mỗi xét nghiệm một nhịp riêng.",
            "<b>Rất thưa</b>: có 353 loại xét nghiệm, mỗi lượt chỉ làm vài chục loại.",
            "<b>Ca dương hiếm</b>: 2–7% mỗi mã, tức là chỉ vài trăm ca để học.",
        ]))
        st.write("")
        card("Hai cách đọc lịch sử", "<p><b>HGB</b>: bảng 451 cột tóm tắt (60 xét nghiệm hay gặp × thống kê, giá trị "
             "gần nhất, mã cũ, sinh hiệu, tuổi, giới).<br><br><b>SeqLab</b>: đọc thẳng chuỗi sự kiện, mỗi giá trị đi kèm "
             "thời điểm đo.</p>")
    why("<p>Chuỗi giữ được xu hướng và thời gian mà bảng tóm tắt bỏ mất. Còn chuyện nó có thật sự dự báo tốt hơn "
        "hay không thì phải kiểm tra, nên HGB luôn là mốc so sánh chính.</p>")


N_WHY_SEQ = """Đây là lý do em đi theo hướng chuỗi.

Cách làm phổ biến là tóm tắt mỗi xét nghiệm thành vài con số: min, max, trung bình, giá trị cuối, rồi đưa vào mô hình cây như HGB. Cách này rất mạnh, và em giữ nó làm đối thủ chính.

Nhưng nhìn hai bệnh nhân này: cùng 4 lần đo creatinin, cùng min, max, trung bình và giá trị cuối. Với bảng tóm tắt, hai người giống hệt nhau. Thực tế một người đang giảm, một người đang tăng.

Dữ liệu bệnh viện còn có ba cái khó: đo không đều, rất thưa, và ca dương hiếm. Mô hình chuỗi xử lý tự nhiên được hai cái đầu. Cái thứ ba thì em xử lý bằng học trước, sẽ nói ở trang sau.

Em không giả định chuỗi chắc chắn thắng. Đó là câu hỏi cần kiểm tra."""


# =============================================================== 7. Toàn cảnh kiến trúc
ARCH = Path(__file__).resolve().parent / "kien_truc"


def s_overview():
    head("Mô hình · toàn cảnh", "Kiến trúc SeqCVD: từ lịch sử bệnh nhân tới 7 xác suất",
         "Đọc từ trái sang phải: sự kiện → nhúng → encoder SeqLab → học trước (①) và tinh chỉnh cho T2 (②). "
         "Hàng dưới là hệ cuối: tổ hợp 21 mạng SeqLab ghép với HGB.")
    st.image(str(ARCH / "seqcvd_architecture_vi.png"), width="stretch")
    a, b, c, _ = st.columns([1, 1, 1, 3])
    for col, (lab, f, mime) in zip((a, b, c), [("Tải PNG (tiếng Việt)", "seqcvd_architecture_vi.png", "image/png"),
                                               ("Tải PNG (English)", "seqcvd_architecture_en.png", "image/png"),
                                               ("Tải PDF (English)", "seqcvd_architecture_en.pdf", "application/pdf")]):
        col.download_button(lab, (ARCH / f).read_bytes(), file_name=f, mime=mime, width="stretch")


N_OVERVIEW = """Đây là toàn cảnh, em đi theo hình từ trái sang phải.

Ô đầu tiên là đầu vào: toàn bộ lịch sử của bệnh nhân tới thời điểm dự báo τ, gồm xét nghiệm, sinh hiệu, mã ICD, lượt khám, cộng tuổi và giới. Mỗi sự kiện là một bộ ba: đo cái gì, giá trị bao nhiêu, lúc nào.

Ô màu vàng: mỗi sự kiện được nhúng bằng năm thành phần rồi cộng lại thành một vector 96 chiều.

Ô màu xanh dương là encoder SeqLab: ba lớp Transformer, rồi gộp bằng attention ra một vector z đại diện cho cả bệnh nhân.

Bên phải có hai giai đoạn. Giai đoạn một, màu xanh lá, là học trước không cần nhãn: che giá trị rồi đoán lại, và đoán lần lấy máu kế tiếp. Giai đoạn hai, màu cam, là tinh chỉnh cho T2 với 7 đầu ra.

Hàng dưới là hệ cuối: lấy trung bình 21 mạng SeqLab, ghép với HGB đọc bảng tóm tắt, trọng số ghép chọn trên validation. Đầu ra là 7 xác suất; điểm chính tính trên 5 mã tô đậm.

Các trang sau em đi vào từng phần và giải thích vì sao chọn như vậy."""


# =============================================================== 8. SeqLab
def s_arch():
    head("Mô hình · SeqLab", "SeqLab: mỗi sự kiện thành một token, Transformer đọc cả chuỗi, ra 7 xác suất",
         "Kế thừa ý tưởng của STraTS (mỗi lần đo là bộ ba loại–giá trị–thời gian), mở rộng để đọc cả mã chẩn đoán và "
         "lượt khám trong cùng một chuỗi.")
    flow([
        ("Đầu vào", "≤ 256 sự kiện gần nhất", "+ tuổi, giới. Khi phải cắt, ưu tiên giữ tới 128 mã/lượt gần nhất"),
        ("Token", "5 thành phần", "loại đo · giá trị · thời gian · loại sự kiện · thuộc lượt hiện tại?"),
        ("Nhúng", "vector 96 chiều", "mỗi thành phần một embedding, cộng lại"),
        ("Transformer", "3 lớp × 4 head", "mỗi sự kiện nhìn được mọi sự kiện khác trong lịch sử"),
        ("Gộp", "attention pooling", "học trọng số cho từng sự kiện → vector hồ sơ z"),
        ("Đầu ra", "7 sigmoid", "mỗi mã một xác suất, không cần cộng bằng 1"),
    ], highlight=(1, 4))
    st.write("")
    html("""<div class="card"><h4>Mỗi lựa chọn và lý do</h4><table class="simple">
<tr><th style="width:28%">Lựa chọn</th><th>Vì sao</th></tr>
<tr><td>Token loại–giá trị–thời gian</td><td>Không phải chia lưới thời gian, không phải điền giá trị thiếu. Đo không đều không còn là vấn đề.</td></tr>
<tr><td>Đưa mã ICD và lượt khám vào chuỗi</td><td>Tiền sử bệnh là tín hiệu mạnh nhất cho mã mới; để chung một chuỗi thì mô hình tự học quan hệ giữa xét nghiệm và chẩn đoán.</td></tr>
<tr><td>Cắt chuỗi ưu tiên mã/lượt</td><td>Bệnh nhân lâu năm có hàng nghìn kết quả; nếu chỉ lấy 256 kết quả mới nhất sẽ mất hết tiền sử bệnh.</td></tr>
<tr><td>Transformer</td><td>Xét nghiệm cách nhau hàng tháng vẫn liên hệ trực tiếp được với nhau, không phải đi qua từng bước như RNN.</td></tr>
<tr><td>Attention pooling</td><td>Mô hình tự chọn sự kiện quan trọng (ví dụ troponin cao) thay vì lấy trung bình mọi thứ.</td></tr>
<tr><td>Mô hình nhỏ (96 chiều, 3 lớp)</td><td>Chỉ có vài nghìn bệnh nhân nên giữ mô hình gọn để đỡ học thuộc. Cửa sổ dài hơn (512 sự kiện) đã thử, không tốt hơn.</td></tr>
</table></div>""")


N_ARCH = """Đây là kiến trúc SeqLab, đi từ trái sang phải.

Mỗi sự kiện được biến thành một token gồm năm phần: đây là xét nghiệm gì, giá trị bao nhiêu, đo lúc nào, là loại sự kiện gì, và có thuộc lượt hiện tại hay không. Mỗi phần có embedding riêng, cộng lại thành vector 96 chiều.

Transformer 3 lớp cho phép mọi sự kiện nhìn nhau. Sau đó một lớp gộp có trọng số tạo ra một vector duy nhất đại diện cho cả hồ sơ, rồi 7 đầu sigmoid cho 7 xác suất.

Phần này em kế thừa từ STraTS, em nói rõ để không nhận là của mình. Phần em thêm vào là: đưa cả mã chẩn đoán và lượt khám vào chung chuỗi, cách cắt chuỗi ưu tiên giữ tiền sử bệnh, bài học trước dự báo lần lấy máu tiếp theo, và cách ghép hệ cuối.

Mô hình cố tình nhỏ, vì dữ liệu chỉ vài nghìn bệnh nhân."""


# =============================================================== 8. Học trước
def s_pretrain():
    f = forecast()
    better = int((f.mse_model < f.mse_locf).sum())
    gain = 1 - f.mse_model.sum() / f.mse_locf.sum()
    head("Mô hình · học trước", "Học trước trên dữ liệu không nhãn, rồi mới học bài T2",
         "Nhãn T2 chỉ có vài trăm ca dương mỗi mã, nhưng kho có hàng chục nghìn đoạn lịch sử không cần nhãn. "
         "SeqLab học \"đọc\" xét nghiệm từ đó trước.")
    a, b, c = st.columns(3)
    with a:
        card("Bài 1 · Che rồi đoán lại", "<p>Che ngẫu nhiên một phần giá trị trong chuỗi, bắt mô hình đoán lại từ "
             "các sự kiện xung quanh. Giống cách BERT học ngôn ngữ.</p>", "accent", "Tự giám sát")
    with b:
        card("Bài 2 · Đoán lần lấy máu kế tiếp", "<p>Từ lịch sử, đoán kết quả của lần xét nghiệm / sinh hiệu đầu tiên "
             "sau đó (trong 30 ngày). Bài này buộc mô hình hiểu xu hướng.</p>", "accent", "Tự giám sát")
    with c:
        card("Sau đó · Tinh chỉnh cho T2", "<p>Giữ encoder đã học, gắn đầu 7 mã và học tiếp bằng nhãn T2. "
             "Một nhóm encoder còn học thêm bài đoán mã ICD bị che.</p>", "accent", "Có giám sát")
    st.write("")
    l, r = st.columns([1.35, 1])
    with l:
        top = f.sort_values("n", ascending=False).head(14).iloc[::-1]
        g = top.mse_gain_vs_locf * 100
        fig = go.Figure(go.Bar(x=g, y=top["name"].str.replace(r"\s*\(.*\)", "", regex=True), orientation="h", cliponaxis=False,
                               marker=dict(color=["#2a78d6" if v > 0 else "#e34948" for v in g], cornerradius=4),
                               text=[f"{v:.0f}%" for v in g], textposition="outside",
                               hovertemplate="%{y}<br>giảm sai số %{x:.1f}%<extra></extra>"))
        fig.update_xaxes(title="Sai số giảm so với \"lấy lại giá trị lần trước\" (%)", range=[0, max(g) * 1.2])
        html('<div style="font-weight:700;margin-bottom:-.4rem">Dự báo lần lấy máu kế tiếp, 14 xét nghiệm hay làm nhất</div>')
        st.plotly_chart(fig_layout(fig, 400), config=CFG, width="stretch")
    with r:
        kpi(f"{gain * 100:.0f}%", "sai số (MSE) thấp hơn cách lấy lại giá trị lần trước", "blue")
        st.write("")
        kpi(f"{better}/{len(f)}", "xét nghiệm dự báo tốt hơn", "green")
        st.write("")
        html('<div class="note">Mốc so sánh là "giá trị sẽ giống lần trước", một mốc khó vượt với xét nghiệm. '
             'Creatinin gần như ngang (sai số tuyệt đối 0,346 so với 0,345 mg/dL).</div>')
    why("<p>Đây là bằng chứng cho mục 2: biểu diễn SeqLab học được đã chứa thông tin về diễn biến xét nghiệm, "
        "trước khi được cho xem bất kỳ nhãn bệnh nào. Ở bài T1, gần như toàn bộ lợi ích của SeqLab cũng đến từ bước "
        "học trước này.</p>")


N_PRETRAIN = """Vấn đề của T2 là nhãn ít: mỗi mã chỉ vài trăm ca dương. Nhưng dữ liệu không nhãn thì nhiều: mỗi lần chia có khoảng 45 nghìn đoạn lịch sử dùng được để học trước.

Nên em cho SeqLab làm hai bài tự giám sát trước. Bài một là che giá trị rồi đoán lại, giống BERT. Bài hai là đoán kết quả của lần lấy máu tiếp theo. Bài hai buộc mô hình phải hiểu xu hướng, chứ không chỉ nhớ giá trị.

Kết quả bài hai cũng chính là bằng chứng cho mục 2 của đề cương: so với cách đơn giản là lấy lại giá trị lần trước, sai số giảm khoảng 29%, tốt hơn ở 72 trên 76 xét nghiệm. Creatinin thì gần như ngang, vì creatinin vốn ổn định.

Sau đó em mới gắn đầu dự báo 7 mã và tinh chỉnh bằng nhãn T2."""


# =============================================================== 9. Hệ cuối
def s_system():
    head("Mô hình · hệ cuối", "Hệ cuối: trung bình 21 mạng SeqLab, rồi ghép với HGB",
         "Một mạng đơn lẻ dao động khá nhiều theo seed. Lấy trung bình nhiều mạng cho kết quả ổn định hơn, và ghép với "
         "cây giúp bền khi dữ liệu học ít.")
    flow([
        ("Nhánh SeqLab", "21 mạng", "9 mạng học riêng T2 · 3 mạng từ encoder có che mã ICD · 9 mạng học chung T2 + bài phụ 1 năm"),
        ("", "Trung bình", "lấy trung bình điểm của 21 mạng"),
        ("Nhánh cây", "HGB", "bảng 451 cột tóm tắt, trung bình 5 bộ HGB cho mỗi mã"),
        ("Ghép", "w·SeqLab + (1−w)·HGB", "ghép trên thang logit; w chọn trên validation cho từng lần chia"),
        ("Kết quả", "7 xác suất", "một cho mỗi mã"),
    ], highlight=(3,))
    st.write("")
    a, b = st.columns(2)
    with a:
        card("Vì sao nhiều mạng", ul([
            "Cùng cấu hình, đổi seed là AP đổi đáng kể. Trung bình nhiều mạng làm giảm phần dao động này.",
            "Lợi ích bão hoà khá sớm: ở bài T1, tăng từ 12 lên 15–18 mạng không hơn.",
            "Trộn các cách học khác nhau (riêng, có che mã, học chung) cho đa dạng hơn là chỉ đổi seed.",
        ]))
    with b:
        card("Vì sao ghép với HGB", ul([
            "Cây rất mạnh trên dữ liệu bảng và bền khi ít dữ liệu; mạng thì cần nhiều dữ liệu hơn.",
            "Hai nhánh đọc dữ liệu theo hai cách khác nhau nên sai ở những chỗ khác nhau, ghép lại bù cho nhau.",
            "Trọng số chỉ chọn theo validation, không nhìn test. Có kỳ hệ chọn 100% SeqLab, không phải lúc nào cũng 50–50.",
        ]))
    why("<p>Ở các thử nghiệm, SeqLab một mình mới ngang HGB. Ghép lại thì vượt HGB rõ ràng ở cả hai cách đánh giá "
        "(trang kết quả). Như vậy hai cách đọc lịch sử thật sự bổ sung cho nhau.</p>")


N_SYSTEM = """Hệ cuối có hai nhánh.

Nhánh SeqLab là trung bình của 21 mạng. Lý do là một mạng đơn rất nhạy với seed, đổi seed là AP lên xuống. Em thử tăng dần số mạng thì thấy lợi ích bão hoà ở khoảng 12 đến 21. 21 mạng này cũng không giống hệt nhau: có nhóm học riêng T2, nhóm dùng encoder học thêm bài che mã ICD, nhóm học chung T2 với bài phụ một năm.

Nhánh thứ hai là HGB trên bảng tóm tắt 451 cột.

Hai nhánh ghép trên thang logit với một trọng số chọn bằng validation. Em không nhìn test khi chọn. Có kỳ hệ tự chọn 100% SeqLab.

Lý do ghép: SeqLab một mình chỉ ngang HGB, nhưng hai bên sai ở những chỗ khác nhau, nên ghép lại thì hơn cả hai."""


# =============================================================== 10. Đánh giá
def rolling_fig():
    ctx = context()
    rows = [s for s in ctx["multitask_fixed_splits"] if s["task"] == "T2"]
    labels = {"roll2021": "2021", "roll2022": "2022", "roll2023": "2023", "roll2024-25": "2024–25"}
    end = {"roll2021": 2021, "roll2022": 2022, "roll2023": 2023, "roll2024-25": 2024}
    fig = go.Figure()
    for s in rows:
        y = f"Kỳ test {labels[s['split']]}"
        e = end[s["split"]]
        w = 2 if s["split"] == "roll2024-25" else 1
        fig.add_trace(go.Bar(y=[y], x=[e - 2020], base=[2020], orientation="h", marker=dict(color="#8db7ea", cornerradius=4),
                             name="Học + chọn (quá khứ)", showlegend=s["split"] == "roll2021",
                             hovertemplate=f"Học: {vn(s['train']['samples'])} mẫu · chọn: {vn(s['validation']['samples'])} mẫu<extra></extra>"))
        fig.add_trace(go.Bar(y=[y], x=[w], base=[e], orientation="h", marker=dict(color="#eb6834", cornerradius=4),
                             name="Test", showlegend=s["split"] == "roll2021",
                             text=[f"{vn(s['test']['samples'])} mẫu"], textposition="inside",
                             hovertemplate=f"Test: {vn(s['test']['samples'])} mẫu<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(range=[2020, 2026], dtick=1, title="năm")
    fig.update_yaxes(autorange="reversed")
    return fig_layout(fig, 260)


def s_eval():
    head("Đánh giá", "Hai cách kiểm tra, một chỉ số chính",
         "Cả hai cách đều tách hẳn dữ liệu kiểm tra khỏi dữ liệu học. Mọi lựa chọn mô hình chỉ dựa trên validation.")
    l, r = st.columns(2)
    with l:
        card("Cách 1 · CV 5 phần theo bệnh nhân", "<p>Chia bệnh nhân thành 5 nhóm. Mỗi lần 3 nhóm để học, 1 nhóm để "
             "chọn, 1 nhóm để test, rồi xoay vòng. Một bệnh nhân không bao giờ vừa ở phần học vừa ở phần test.</p>",
             "accent")
        z = go.Figure()
        roles = ["học", "học", "học", "chọn", "test"]
        col = {"học": "#8db7ea", "chọn": "#c9ccd2", "test": "#eb6834"}
        for i in range(5):
            rr = roles[-i:] + roles[:-i] if i else roles
            for j, role in enumerate(rr):
                z.add_trace(go.Bar(y=[f"Lần {i + 1}"], x=[1], base=[j], orientation="h", showlegend=(i == 0 and j in (0, 3, 4)),
                                   name=role, marker=dict(color=col[role], line=dict(color="white", width=2)),
                                   hovertemplate=f"Nhóm {j + 1}: {role}<extra></extra>"))
        z.update_layout(barmode="overlay")
        z.update_xaxes(visible=False)
        z.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_layout(z, 220), config=CFG, width="stretch")
    with r:
        card("Cách 2 · Học quá khứ, test tương lai", "<p>Mỗi kỳ chỉ học trên các năm trước rồi test trên năm sau. "
             "Gộp 4 kỳ test lại để tính chỉ số. Cách này gần với triển khai thật nhất, và khó hơn CV.</p>", "accent")
        st.plotly_chart(rolling_fig(), config=CFG, width="stretch")
    a, b = st.columns([1.1, 1])
    with a:
        card("Chỉ số chính: macro AP (AUPRC)", "<p>Tính AP cho từng mã trong 5 mã chính, lấy trung bình. "
             "AP đo xem các ca có mã mới có được xếp lên đầu danh sách hay không. Đoán bừa thì AP bằng tỉ lệ ca dương, "
             "khoảng 0,05.</p>")
    with b:
        card("Chỉ số phụ (trang Kết quả)", "<p>AUROC, Accuracy, Balanced acc., Recall, Specificity, Precision, NPV, "
             "F1, MCC, Brier. Với các chỉ số cần ngưỡng: mọi phương pháp báo động đúng bằng số ca bằng tỉ lệ dương "
             "trên validation. Khoảng tin cậy 95% bằng bootstrap theo bệnh nhân.</p>")
    why("<p>Ca dương chỉ khoảng 5%, nên một mô hình luôn đoán \"không\" vẫn có Accuracy khoảng 95%. "
        "AP không bị đánh lừa kiểu đó, nên em dùng AP làm thước đo chính và mọi quyết định giữ / bỏ đều theo AP "
        "trên validation.</p>")


N_EVAL = """Em kiểm tra theo hai cách.

Cách một là CV 5 phần chia theo bệnh nhân: 3 phần học, 1 phần chọn mô hình, 1 phần test, xoay vòng. Chia theo bệnh nhân để một người không xuất hiện ở cả học và test.

Cách hai khó hơn và gần thực tế hơn: chỉ học trên các năm trước, test trên năm sau. Có 4 kỳ: 2021, 2022, 2023 và 2024–25. Kỳ 2021 chỉ có 1 năm dữ liệu để học nên rất khó.

Chỉ số chính là macro AP: AP của từng mã rồi lấy trung bình 5 mã. Lý do: ca dương chỉ 5%, nên Accuracy gần như vô nghĩa. Mô hình đoán "không" hết vẫn được 95%.

Các chỉ số khác em vẫn báo đủ ở tab Kết quả. Để so công bằng, mọi phương pháp dùng chung một quy tắc ngưỡng: số ca bị báo động bằng tỉ lệ ca dương trên validation."""


# =============================================================== 11. Kết quả chính
def s_result():
    head("Kết quả", "Hệ SeqLab + HGB cho AP cao nhất ở cả hai cách đánh giá")
    task = "T2"
    rows = []
    for ev in ("cv", "future"):
        t = macro_table(task, ev)["AUPRC"]
        best_seq = max((k for k in t.index if k in PUBLISHED_SEQ), key=lambda k: t[k])
        rows.append((ev, t, best_seq))
    l, r = st.columns([1.35, 1])
    with l:
        fig = go.Figure()
        order = [("hgb", "HGB"), ("best", "Mô hình chuỗi tốt nhất đã công bố"), (NEURAL[task], "SeqLab"),
                 (OUR[task], "SeqLab + HGB")]
        colors = {"hgb": GROUP_COLOR["Mô hình cây"], "best": GROUP_COLOR["Mô hình chuỗi EHR đã công bố"],
                  NEURAL[task]: GROUP_COLOR["SeqLab"], OUR[task]: GROUP_COLOR["Hệ cuối"]}
        xs = ["CV theo bệnh nhân", "Các năm sau"]
        for key, lab in order:
            ys, hov = [], []
            for ev, t, best in rows:
                k = best if key == "best" else key
                ys.append(t[k])
                hov.append(name(k))
            fig.add_trace(go.Bar(x=xs, y=ys, name=lab, marker=dict(color=colors[key], cornerradius=4),
                                 text=[fmt(v) for v in ys], textposition="outside", customdata=hov, cliponaxis=False,
                                 hovertemplate="%{customdata}: %{y:.4f}<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
        fig.update_yaxes(title="macro AP", range=[0.10, 0.168])
        html('<div style="font-weight:700;margin-bottom:-.4rem">macro AP trên 5 mã (trục y bắt đầu từ 0,10)</div>')
        st.plotly_chart(fig_layout(fig, 380), config=CFG, width="stretch")
    with r:
        v, lo, hi = ci(task, "cv", "hgb")
        kpi(f"{pts(v)} điểm AP", f"hơn HGB trên CV · CI 95% [{pts(lo)}; {pts(hi)}]", "blue")
        st.write("")
        v, lo, hi = ci(task, "future", "hgb")
        kpi(f"{pts(v)} điểm AP", f"hơn HGB trên các năm sau · CI 95% [{pts(lo)}; {pts(hi)}]", "blue")
        st.write("")
        cv_best, fu_best = rows[0][2], rows[1][2]
        v1 = ci(task, "cv", cv_best)[0]
        v2 = ci(task, "future", fu_best)[0]
        kpi(f"{pts(v1)} / {pts(v2)}", f"hơn mô hình chuỗi đã công bố tốt nhất ({name(cv_best)} / {name(fu_best)}), "
            "CV / năm sau", "green")
    why(ul([
        "Khoảng tin cậy của mọi chênh lệch trên đều nằm trên 0.",
        "SeqLab một mình hơn HGB khoảng +0,5 điểm nhưng khoảng tin cậy còn chạm 0, nên em chỉ nói là ngang HGB. "
        "Lợi ích rõ ràng đến khi ghép hai nhánh.",
        "Trên các năm sau, mọi mô hình đều tụt AP so với CV; thứ tự thì giữ nguyên.",
    ]), "Đọc kết quả này thế nào")


N_RESULT = """Đây là kết quả chính của T2. Trục là macro AP.

Trên CV, hệ SeqLab + HGB đạt khoảng 0,159, hơn HGB 1,2 điểm, khoảng tin cậy từ 0,8 đến 1,7, không chạm 0. Trên các năm sau, mức hơn nhỏ hơn, khoảng 0,6 điểm, nhưng vẫn có ý nghĩa.

So với mô hình chuỗi tốt nhất trong các bài đã công bố, hệ hơn khoảng 2 điểm trên CV và hơn 1 điểm ở các năm sau.

Em muốn nói trung thực một điểm: SeqLab một mình chỉ ngang HGB, chưa hơn rõ. Lợi ích có ý nghĩa xuất hiện khi ghép hai nhánh. Em đọc điều này là: chuỗi chứa thông tin mà bảng tóm tắt không có, nhưng cây vẫn khai thác bảng tóm tắt tốt hơn."""


# =============================================================== 12. SOTA
def s_sota():
    head("So sánh", "Xếp hạng 13 phương pháp trên cùng dữ liệu, cùng cách chia",
         "Các mô hình chuỗi EHR đã công bố (RETAIN, Doctor AI, GRU-D, BEHRT, STraTS) được cài lại trên dữ liệu của mình, "
         "cùng đầu vào sự kiện, cùng tập test. Bảng đầy đủ mọi chỉ số ở tab Kết quả.")
    ev = st.radio("Cách đánh giá", ["cv", "future"], format_func=lambda e: {"cv": "CV theo bệnh nhân",
                  "future": "Các năm sau"}[e], horizontal=True, key="sota_ev", label_visibility="collapsed")
    l, r = st.columns([1.5, 1])
    with l:
        st.plotly_chart(ranking_bar("T2", ev, "AUPRC", height=470), config=CFG, width="stretch")
        group_legend("T2", macro_table("T2", ev).index)
    with r:
        card("Điều rút ra", ul([
            "SeqLab đứng trên mọi mô hình chuỗi EHR đã công bố, ở cả hai cách đánh giá.",
            "Các mô hình cây (HGB, LightGBM, XGBoost) cũng vượt hầu hết mô hình chuỗi đã công bố. "
            "Trên dữ liệu cỡ này, cây là đối thủ khó nhất, nên em giữ HGB trong hệ cuối.",
            "Đoán theo tần suất chỉ đạt AP khoảng 0,05: đây là mức sàn.",
        ]))
        st.write("")
        html('<div class="note">Lưu ý: các mô hình đã công bố là bản cài lại cho bài toán này, không phải mã gốc '
             'của tác giả. BEHRT ở đây chỉ đọc mã chẩn đoán và chưa học trước như bản gốc; RETAIN, Doctor AI, GRU-D '
             'đọc dữ liệu đã gộp theo lượt.</div>')


N_SOTA = """Đây là bảng xếp hạng 13 phương pháp, theo AP.

Nhóm màu xanh lá là các mô hình chuỗi EHR đã công bố: RETAIN, Doctor AI, GRU-D, BEHRT, STraTS. Em cài lại chúng trên đúng dữ liệu và cách chia của mình để so công bằng. SeqLab đứng trên tất cả, ở cả CV lẫn các năm sau.

Một điểm đáng chú ý: các mô hình cây như HGB, LightGBM, XGBoost cũng vượt gần hết mô hình chuỗi đã công bố. Điều này khớp với kết quả của bài Labrador năm 2024: trên dữ liệu xét nghiệm, cây là baseline rất khó vượt. Đó là lý do em giữ HGB trong hệ.

Em cũng nói rõ giới hạn: đây là bản cài lại, không phải mã gốc của tác giả. Em không lấy số từ các bài báo để so, vì khác dữ liệu, khác nhãn."""


# =============================================================== 13. Đọc con số
def s_read():
    head("Đọc con số cho đúng", "AP 0,16 nghe thấp, nhưng phải đặt cạnh tỉ lệ ca dương khoảng 5%")
    c = curves()["T2"]
    l, r = st.columns([1.25, 1])
    with l:
        ev = "cv"
        per = {x["cls"]: x for x in c[ev]["perclass"] if x["method"] == OUR["T2"]}
        prev = [c[ev]["curves"][k][OUR["T2"]]["prevalence"] for k in EVAL_CODES]
        ap = [per[k]["AUPRC"] for k in EVAL_CODES]
        xs = [f"<b>{k}</b><br>{wrap(CODES[k])}" for k in EVAL_CODES]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=xs, y=prev, name="Tỉ lệ ca dương (= AP khi đoán bừa)", marker=dict(color="#c9ccd2", cornerradius=4),
                             text=[fmt(v) for v in prev], textposition="outside", cliponaxis=False))
        fig.add_trace(go.Bar(x=xs, y=ap, name="AP của hệ cuối", marker=dict(color="#2a78d6", cornerradius=4),
                             text=[fmt(v) for v in ap], textposition="outside", cliponaxis=False))
        fig.update_layout(barmode="group", bargap=0.3)
        fig.update_yaxes(title="AP", range=[0, max(ap) * 1.2])
        fig.update_xaxes(tickangle=0, tickfont=dict(size=11))
        html('<div style="font-weight:700;margin-bottom:-.4rem">AP từng mã so với mức đoán bừa (CV)</div>')
        st.plotly_chart(fig_layout(fig, 360), config=CFG, width="stretch")
        ratio = sum(ap) / sum(prev)
        html(f'<div class="note">Trung bình AP gấp khoảng <b>{fmt(ratio, 1)} lần</b> mức đoán bừa. I20 (đau thắt ngực) '
             'dễ nhất; I25 khó nhất, cũng là mã ít ca nhất trong 5 mã.</div>')
    with r:
        fu = {x["cls"]: x for x in c["future"]["perclass"] if x["method"] == OUR["T2"]}["I20"]
        alerts = fu["TP"] + fu["FP"]
        card("Ví dụ: I20 trên các năm sau", f"""<table class="simple">
<tr><td>Số lượt thật sự ghi thêm I20</td><td class="num"><b>{fu['positive']}</b></td></tr>
<tr><td>Số lượt hệ báo động</td><td class="num">{alerts}</td></tr>
<tr><td>Báo động đúng</td><td class="num">{fu['TP']}</td></tr>
<tr><td>Bỏ sót</td><td class="num">{fu['FN']}</td></tr>
<tr><td>Accuracy</td><td class="num">{fmt(fu['Accuracy'])}</td></tr>
<tr><td>Recall</td><td class="num">{fmt(fu['Recall'])}</td></tr>
<tr><td>Precision</td><td class="num">{fmt(fu['Precision'])}</td></tr>
</table><p style="margin-top:.6rem">Accuracy cao vì phần lớn lượt không có mã mới. Thực tế hệ mới bắt được khoảng
1/8 số ca khi chỉ được báo động ít lượt.</p>""")
    why("<p>Em không trình bày hệ này như công cụ sẵn sàng dùng lâm sàng. Điều kết quả cho thấy là: trong cùng điều "
        "kiện, đọc chuỗi + ghép cây xếp hạng bệnh nhân nguy cơ tốt hơn mọi cách đã thử. Bài toán bản thân nó rất khó: "
        "chỉ biết được lượt sau có được ghi mã mới hay không, không biết bệnh khởi phát khi nào.</p>", "Nói thẳng")


N_READ = """Trang này để tránh hiểu nhầm con số.

AP 0,16 nghe thấp, nhưng mức đoán bừa chỉ khoảng 0,05. Tính trung bình, hệ cao gấp khoảng 3 lần mức đó. I20 dễ nhất, I25 khó nhất.

Bên phải là I20 ở các năm sau, khi hệ chỉ được báo động số lượt bằng tỉ lệ dương. Accuracy 0,91 nghe rất đẹp, nhưng thực ra trong 845 ca thật hệ mới bắt được 99. Đó là lý do em không dùng Accuracy để so.

Em nói thẳng: đây chưa phải công cụ lâm sàng. Kết luận của em hẹp hơn: trong cùng dữ liệu và cùng cách chấm, cách đọc chuỗi ghép với cây xếp hạng nguy cơ tốt hơn mọi phương pháp em đã thử."""


# =============================================================== 14. T1, T3
def s_others():
    head("Nhắc thêm", "Hai bài phụ: T1 và T3",
         "Cùng SeqLab, cùng dữ liệu, nhưng hiện em chưa thấy giá trị đủ lớn để đưa thành kết quả chính.")
    a, b = st.columns(2)
    d1 = []
    for ev in ("cv", "future"):
        x = deltas("T1", ev, "AUPRC").assign(tau=lambda z: metrics("T1", ev).loc[z.index, "tau"])
        x = x[x.b == "RAD system (published)"].sort_values("tau")
        d1.append([fmt(v * 100, 1, True) for v in x["value"]])
    t3 = {ev: macro_table("T3", ev)["AUPRC"] for ev in ("cv", "future")}
    v, lo, hi = ci("T3", "future", "hgb")
    with a:
        card("T1 · Trong cùng lượt nhập viện", f"""<p>Tại mốc 2 / 6 / 12 giờ sau nhập viện, đoán mã tim mạch sẽ được ghi
thêm trong lượt này. Đưa SeqLab vào hệ RAD (bài đã nộp) thì AP tăng ở cả 6 ô, khoảng tin cậy đều trên 0.</p>
<table class="simple" style="margin-top:.5rem"><tr><th>Chênh AP (điểm)</th><th>2 giờ</th><th>6 giờ</th><th>12 giờ</th></tr>
<tr><td>CV</td><td>{d1[0][0]}</td><td>{d1[0][1]}</td><td>{d1[0][2]}</td></tr>
<tr><td>Các năm sau</td><td>{d1[1][0]}</td><td>{d1[1][1]}</td><td>{d1[1][2]}</td></tr></table>
<p style="margin-top:.6rem"><b>Vướng:</b> giờ có kết quả xét nghiệm đang suy ra từ mã phiếu, chưa xác minh được. Chưa
thể gọi đây là "cảnh báo sớm sau 2 giờ" trong thực tế.</p>""", tag="Bài RAD")
    with b:
        card("T3 · Mã mới trong vòng 1 năm", f"""<p>Giống T2 nhưng nhìn xa hơn: mã mới trong 1–365 ngày tới.</p>
<table class="simple" style="margin-top:.5rem"><tr><th>macro AP</th><th>CV</th><th>Các năm sau</th></tr>
<tr><td>HGB</td><td>{fmt(t3['cv']['hgb'])}</td><td>{fmt(t3['future']['hgb'])}</td></tr>
<tr><td>SeqLab</td><td>{fmt(t3['cv']['seqlab_t3mix12'])}</td><td>{fmt(t3['future']['seqlab_t3mix12'])}</td></tr>
<tr class="us"><td>SeqLab + HGB</td><td>{fmt(t3['cv'][OUR['T3']])}</td><td>{fmt(t3['future'][OUR['T3']])}</td></tr></table>
<p style="margin-top:.6rem"><b>Vướng:</b> ở các năm sau, SeqLab một mình kém HGB rõ; hệ ghép chỉ hơn HGB {pts(v)} điểm,
khoảng tin cậy [{pts(lo)}; {pts(hi)}] còn chứa 0. Các kỳ đầu có quá ít dữ liệu học.</p>""", tag="Một năm")
    why("<p>Em giữ hai bài này trong phần phương pháp nhưng không đặt làm kết luận chính. T1 cần xác minh thời điểm "
        "dữ liệu sẵn có; T3 cần thêm dữ liệu hoặc một cách học tốt hơn cho kỳ ít mẫu.</p>")


N_OTHERS = """Hai bài phụ em chỉ nhắc nhanh.

T1 là bài trong cùng lượt nhập viện, gắn với bài RAD. Khi đưa SeqLab vào, AP tăng ở cả 6 ô, có ý nghĩa thống kê. Nhưng em chưa xác minh được giờ kết quả xét nghiệm thật sự có trong tay bác sĩ, nên chưa dám gọi là cảnh báo sớm.

T3 là mã mới trong một năm. Trên CV có lợi, nhưng trên các năm sau SeqLab một mình thua HGB rõ, còn hệ ghép chỉ ngang HGB. Em nghĩ nguyên nhân chính là các kỳ đầu có quá ít dữ liệu học, nhưng đó mới là giả thuyết.

Nên hiện tại em coi cả hai là kết quả phụ."""


# =============================================================== 15. Kết luận
def s_end():
    head("Tổng kết", "Đã làm được gì, còn gì chưa chắc, và bước tiếp theo")
    a, b, c = st.columns(3)
    with a:
        card("Đã làm được", ul([
            "Kho 920 nghìn sự kiện theo thời gian, đã rà rò rỉ nhãn.",
            "SeqLab + học trước: dự báo lần lấy máu kế tiếp tốt hơn 29% so với lấy giá trị cũ (mục 2).",
            "T2: hệ SeqLab + HGB có AP cao nhất, hơn HGB có ý nghĩa ở cả CV và các năm sau (mục 1).",
            "SeqLab vượt mọi mô hình chuỗi EHR đã công bố được cài lại.",
        ]), "keep")
    with b:
        card("Còn chưa chắc", ul([
            "Một bệnh viện, chưa kiểm định ngoài.",
            "Nhãn là mã được ghi mới, không phải ngày khởi phát; chỉ có bệnh nhân quay lại.",
            "SeqLab một mình mới ngang HGB.",
            "Chưa tách được bao nhiêu lợi ích đến từ học trước, từ chuỗi, từ số lượng mạng.",
            "Qua nhiều vòng phát triển đã nhìn kết quả test; cần một lần đánh giá cuối độc lập.",
        ]), "drop")
    with c:
        card("Bước tiếp theo", ul([
            "Thí nghiệm tách phần: học trước vs khởi tạo ngẫu nhiên, 1 mạng vs 21 mạng, ngay trên T2.",
            "Báo thêm kết quả trên nhóm chưa từng có mã (nguy cơ lần đầu).",
            "Chốt quy trình rồi chạy một lần đánh giá cuối, sau đó kiểm định ngoài.",
            "Viết bài: gộp vào bản thảo RAD hay viết riêng cho mục 1 + 2.",
        ]), "accent")


N_END = """Tóm lại.

Đã làm được: kho dữ liệu chuỗi sạch rò rỉ; SeqLab có học trước, dự báo xét nghiệm tốt hơn cách đơn giản khoảng 29%, đáp ứng mục 2; và trên T2, hệ ghép SeqLab với HGB tốt nhất, hơn HGB có ý nghĩa ở cả hai cách đánh giá, đáp ứng mục 1.

Còn chưa chắc: một bệnh viện; nhãn là mã được ghi, không phải ngày khởi phát; SeqLab một mình mới ngang HGB; và em chưa tách được lợi ích đến từ đâu. Ngoài ra, qua nhiều vòng phát triển em có nhìn kết quả test, nên cần một lần đánh giá cuối độc lập.

Bước tiếp theo là các thí nghiệm tách phần, một lần đánh giá cuối, rồi viết bài."""


SLIDES = [  # (phần, tên trang, hàm vẽ, ghi chú để nói)
    ("Mở đầu", "Mở đầu", s_intro, N_INTRO),
    ("Dữ liệu", "Dữ liệu gốc", s_raw, N_RAW),
    ("Dữ liệu", "Chọn gì, bỏ gì", s_select, N_SELECT),
    ("Dữ liệu", "Xử lý dữ liệu", s_process, N_PROCESS),
    ("Dữ liệu", "Mã ICD mục tiêu", s_icd, N_ICD),
    ("Bài toán", "Bài toán T2", s_task, N_TASK),
    ("Bài toán", "Vì sao đọc chuỗi", s_why_seq, N_WHY_SEQ),
    ("Mô hình", "Toàn cảnh kiến trúc", s_overview, N_OVERVIEW),
    ("Mô hình", "Kiến trúc SeqLab", s_arch, N_ARCH),
    ("Mô hình", "Học trước", s_pretrain, N_PRETRAIN),
    ("Mô hình", "Hệ cuối", s_system, N_SYSTEM),
    ("Kết quả", "Cách đánh giá", s_eval, N_EVAL),
    ("Kết quả", "Kết quả T2", s_result, N_RESULT),
    ("Kết quả", "So với SOTA", s_sota, N_SOTA),
    ("Kết quả", "Đọc con số cho đúng", s_read, N_READ),
    ("Kết thúc", "T1 và T3", s_others, N_OTHERS),
    ("Kết thúc", "Tổng kết", s_end, N_END),
]
