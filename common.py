"""Dữ liệu, tên gọi, màu và các khối giao diện dùng chung."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------- tên và nhóm
OUR = {"T1": "final system", "T2": "hybrid_seqlab_t2allfix", "T3": "hybrid_seqlab_t3mix12"}
NEURAL = {"T1": "SeqLab network (cond)", "T2": "seqlab_t2allfix", "T3": "seqlab_t3mix12"}

NAMES = {
    "hybrid_seqlab_t2allfix": "SeqLab + HGB (hệ cuối)", "seqlab_t2allfix": "SeqLab (21 mạng)",
    "hybrid_seqlab_t3mix12": "SeqLab + HGB (hệ cuối)", "seqlab_t3mix12": "SeqLab (12 mạng)",
    "hgb": "HGB", "lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost", "lr": "Hồi quy logistic",
    "retain_e60": "RETAIN", "gru_e60": "Doctor AI (GRU)", "grud_e60": "GRU-D",
    "seqlab_codes_e60": "BEHRT", "seqlab_pt_e60": "STraTS", "seqlab_pt_strats": "STraTS",
    "prior": "Đoán theo tần suất",
    "final system": "HGB RAD-C + mạng có SeqLab (hệ cuối)", "RAD system (published)": "Hệ RAD đã công bố",
    "SeqLab network (cond)": "Mạng RAD có SeqLab", "direct ensemble": "Ghép trực tiếp",
    "FuseMoE direct": "FuseMoE",
}
PUBLISHED_SEQ = {"retain_e60", "gru_e60", "grud_e60", "seqlab_codes_e60", "seqlab_pt_e60", "seqlab_pt_strats"}
TREES = {"hgb", "lgbm", "xgb", "cat"}
SIMPLE = {"lr", "prior"}

GROUP_COLOR = {
    "Hệ cuối": "#2a78d6",
    "SeqLab": "#8db7ea",
    "Mô hình cây": "#eb6834",
    "Mô hình chuỗi EHR đã công bố": "#1baf7a",
    "Mô hình tabular khác": "#6250d6",
    "Mốc đơn giản": "#a3a29c",
}


def wrap(text, width=11):
    """Xuống dòng nhãn dài cho trục biểu đồ."""
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return "<br>".join(lines + [cur])


def name(key):
    if key in NAMES:
        return NAMES[key]
    return key.replace(" direct", " (trực tiếp)").replace(" RAD-C", " (RAD-C)")


def group(key, task):
    if key == OUR[task]:
        return "Hệ cuối"
    if key == NEURAL[task]:
        return "SeqLab"
    if key in TREES or key.startswith(("HGB", "LightGBM", "XGBoost", "CatBoost")):
        return "Mô hình cây"
    if key in PUBLISHED_SEQ:
        return "Mô hình chuỗi EHR đã công bố"
    if key in SIMPLE:
        return "Mốc đơn giản"
    if key == "RAD system (published)":
        return "Mô hình cây"
    return "Mô hình tabular khác"


CODES = {
    "I10": "Tăng huyết áp", "I50": "Suy tim", "I21": "Nhồi máu cơ tim cấp", "I25": "Bệnh tim thiếu máu mạn",
    "I20": "Đau thắt ngực", "I48": "Rung nhĩ", "I46": "Ngừng tim",
}
EVAL_CODES = ["I10", "I50", "I21", "I25", "I20"]

METRICS = ["AUPRC", "AUROC", "Accuracy", "Balanced acc.", "Recall", "Specificity", "Precision", "NPV", "F1", "MCC",
           "Brier", "Sens@Spec90"]
LOWER_BETTER = {"Brier"}
METRIC_INFO = {
    "AUPRC": ("AP / AUPRC — chỉ số chính",
              "Xếp hồ sơ theo điểm từ cao xuống thấp, các ca có mã mới có nằm ở đầu danh sách không. Hợp với bài có ca "
              "dương hiếm. Mốc đoán bừa bằng đúng tỉ lệ ca dương (khoảng 0,05)."),
    "AUROC": ("AUROC", "Bốc ngẫu nhiên một ca có mã mới và một ca không có: mô hình chấm ca có mã mới cao hơn bao nhiêu "
              "phần trăm số lần. 0,5 là đoán bừa."),
    "Accuracy": ("Accuracy", "Tỉ lệ đoán đúng trên tất cả hồ sơ. Vì 95% hồ sơ không có mã mới, đoán 'không' hết cũng "
                 "được khoảng 0,92–0,95, nên chỉ số này gần như không phân biệt được các mô hình."),
    "Balanced acc.": ("Balanced accuracy", "Trung bình của Recall và Specificity, coi hai nhóm có/không quan trọng ngang nhau."),
    "Recall": ("Recall (độ nhạy)", "Trong các ca thật sự có mã mới, mô hình bắt được bao nhiêu."),
    "Specificity": ("Specificity (độ đặc hiệu)", "Trong các ca không có mã mới, mô hình để yên được bao nhiêu."),
    "Precision": ("Precision (PPV)", "Trong các ca bị cảnh báo, bao nhiêu ca đúng là có mã mới."),
    "NPV": ("NPV", "Trong các ca không bị cảnh báo, bao nhiêu ca đúng là không có mã mới."),
    "F1": ("F1", "Trung bình điều hoà của Precision và Recall."),
    "MCC": ("MCC", "Tương quan giữa dự đoán và thực tế, dùng cả bốn ô của bảng nhầm lẫn. Từ −1 đến 1, 0 là đoán bừa."),
    "Brier": ("Brier", "Sai số bình phương giữa xác suất dự báo và kết quả 0/1. Càng thấp càng tốt."),
    "Sens@Spec90": ("Recall khi Specificity = 90%", "Đặt ngưỡng sao cho 90% ca âm không bị báo động, rồi xem bắt được bao nhiêu ca dương."),
}
EVALS = {"cv": "CV 5 phần theo bệnh nhân", "future": "Các năm sau (2021 → 2024–25)"}


# ---------------------------------------------------------------- dữ liệu
@st.cache_data
def metrics(task, ev):
    f = {"T1": {"cv": "t1_cv", "future": "t1_future"}}.get(task, {}).get(ev, f"{task.lower()}_{ev}")
    d = pd.read_csv(DATA / f"{f}.csv")
    d["metric"] = d["metric"].replace({"Sensitivity": "Recall", "PPV": "Precision"})
    return d


@st.cache_data
def curves():
    return json.loads((DATA / "curves.json").read_text(encoding="utf-8"))


@st.cache_data
def context():
    return json.loads((DATA / "research_context.json").read_text(encoding="utf-8"))


@st.cache_data
def forecast():
    return pd.read_csv(DATA / "forecast.csv")


def macro_table(task, ev, tau=None):
    d = metrics(task, ev)
    v = d[d.kind == "value"]
    if tau is not None:
        v = v[v.tau == tau]
    t = v.pivot_table(index="a", columns="metric", values="value")
    return t[[m for m in METRICS if m in t.columns]]


def deltas(task, ev, metric, tau=None):
    d = metrics(task, ev)
    x = d[(d.kind == "delta") & (d.a == OUR[task]) & (d.metric == metric)]
    if tau is not None:
        x = x[x.tau == tau]
    return x[["b", "value", "lo", "hi"]]


# ---------------------------------------------------------------- giao diện
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
:root{--blue:#2a78d6;--blue-soft:#eaf2fc;--ink:#16181d;--ink2:#4a4f5a;--muted:#7a7f8a;--line:#e3e5ea;
--card:#ffffff;--bg:#f6f7f9;--amber:#b86e00;--amber-soft:#fff5e3;--green:#13804f;--green-soft:#e8f6ef;
--red:#c23b3a;--red-soft:#fdeeee;}
html, body, [class*="css"], .stMarkdown, .stButton button, .stRadio, .stSelectbox{font-family:'Be Vietnam Pro',sans-serif !important;}
.stApp{background:var(--bg);}
.block-container{padding-top:3.4rem;max-width:1280px;}
h1,h2,h3{color:var(--ink);letter-spacing:-0.01em;}
.kicker{font-size:.8rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--blue);margin-bottom:.2rem;}
.slide-title{font-size:2.05rem;font-weight:800;line-height:1.2;color:var(--ink);margin:0 0 .45rem 0;}
.lead{font-size:1.12rem;color:var(--ink2);line-height:1.55;margin-bottom:1.1rem;max-width:62rem;}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.05rem 1.2rem;height:100%;
box-shadow:0 1px 2px rgba(16,24,40,.04);}
.card h4{margin:0 0 .45rem 0;font-size:1.02rem;font-weight:700;color:var(--ink);}
.card p,.card li{color:var(--ink2);font-size:.96rem;line-height:1.55;margin:0;}
.card ul{margin:.2rem 0 0 1.1rem;padding:0;}
.card li{margin-bottom:.25rem;}
.card .tag{display:inline-block;font-size:.72rem;font-weight:700;padding:.12rem .5rem;border-radius:99px;
background:var(--blue-soft);color:var(--blue);margin-bottom:.45rem;}
.card.keep{border-top:4px solid var(--green);} .card.drop{border-top:4px solid var(--red);}
.card.accent{border-top:4px solid var(--blue);}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:.85rem 1rem;}
.kpi .v{font-size:1.75rem;font-weight:800;color:var(--ink);line-height:1.1;}
.kpi .l{font-size:.85rem;color:var(--muted);margin-top:.2rem;}
.kpi.blue .v{color:var(--blue);} .kpi.green .v{color:var(--green);}
.why{background:var(--amber-soft);border-left:4px solid #e6a23c;border-radius:10px;padding:.85rem 1.1rem;margin-top:1rem;}
.why b.h{color:var(--amber);font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;display:block;margin-bottom:.3rem;}
.why p,.why li{color:#4a3a1c;font-size:.97rem;line-height:1.55;margin:0;}
.why ul{margin:.1rem 0 0 1.1rem;padding:0;}
.note{background:#f0f1f4;border-radius:10px;padding:.7rem 1rem;color:var(--ink2);font-size:.92rem;line-height:1.5;margin-top:.6rem;}
.flow{display:flex;align-items:stretch;gap:.4rem;flex-wrap:wrap;margin:.4rem 0 .2rem 0;}
.flow .step{flex:1 1 0;min-width:130px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:.7rem .75rem;}
.flow .step.hl{border:2px solid var(--blue);background:var(--blue-soft);}
.flow .step .n{font-size:.72rem;font-weight:700;color:var(--blue);letter-spacing:.05em;text-transform:uppercase;}
.flow .step .t{font-weight:700;color:var(--ink);font-size:.95rem;margin:.15rem 0;}
.flow .step .d{color:var(--ink2);font-size:.84rem;line-height:1.45;}
.flow .arrow{align-self:center;color:var(--muted);font-size:1.3rem;font-weight:700;}
.pill{display:inline-block;padding:.18rem .6rem;border-radius:99px;font-size:.82rem;font-weight:600;margin:.12rem .2rem .12rem 0;
border:1px solid var(--line);background:#fff;color:var(--ink2);}
.pill.b{background:var(--blue-soft);border-color:#c5dbf6;color:#1d5fae;}
.pill.g{background:var(--green-soft);border-color:#bfe6d2;color:var(--green);}
.pill.r{background:var(--red-soft);border-color:#f3c9c9;color:var(--red);}
.big-claim{font-size:1.25rem;font-weight:700;color:var(--ink);line-height:1.45;}
table.simple{border-collapse:collapse;width:100%;font-size:.92rem;}
table.simple th{text-align:left;color:var(--muted);font-weight:600;border-bottom:1px solid var(--line);padding:.4rem .5rem;}
table.simple td{border-bottom:1px solid var(--line);padding:.42rem .5rem;color:var(--ink2);vertical-align:top;}
table.simple td.num{text-align:right;font-variant-numeric:tabular-nums;}
table.simple tr.us td{background:var(--blue-soft);color:var(--ink);font-weight:600;}
.progress-wrap{height:5px;background:#e3e5ea;border-radius:99px;margin:.3rem 0 1.2rem 0;overflow:hidden;}
.progress-bar{height:100%;background:var(--blue);border-radius:99px;}
div[data-testid="stExpander"] details{background:#fff;border-radius:12px;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def kpi(value, label, tone=""):
    html(f'<div class="kpi {tone}"><div class="v">{value}</div><div class="l">{label}</div></div>')


def card(title, body, cls="", tag=None):
    t = f'<span class="tag">{tag}</span>' if tag else ""
    html(f'<div class="card {cls}">{t}<h4>{title}</h4>{body}</div>')


def why(body, head="Vì sao làm vậy"):
    html(f'<div class="why"><b class="h">{head}</b>{body}</div>')


def flow(steps, highlight=()):
    parts = []
    for i, (n, t, d) in enumerate(steps):
        cls = "step hl" if i in highlight else "step"
        parts.append(f'<div class="{cls}"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>')
    html('<div class="flow">' + '<div class="arrow">›</div>'.join(parts) + "</div>")


def ul(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def fig_layout(fig, height=360, **kw):
    fig.update_layout(
        height=height, margin=dict(l=10, r=20, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Be Vietnam Pro, sans-serif", size=13, color="#2b2f36"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(font_family="Be Vietnam Pro, sans-serif"), separators=",.", **kw)
    fig.update_xaxes(gridcolor="#eef0f3", zeroline=False, linecolor="#d5d8de")
    fig.update_yaxes(gridcolor="#eef0f3", zeroline=False, linecolor="#d5d8de")
    return fig


def fmt(x, nd=3, sign=False):
    """Số thập phân kiểu Việt Nam: 0,159."""
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace(".", ",")


def vn(n):
    """Số nguyên kiểu Việt Nam: 12.514."""
    return f"{n:,}".replace(",", ".")


def ranking_bar(task, ev, metric="AUPRC", tau=None, height=430):
    t = macro_table(task, ev, tau)[metric].dropna()
    t = t.sort_values(ascending=metric in LOWER_BETTER)
    keys = list(t.index)[::-1]
    vals = [t[k] for k in keys]
    cols = [GROUP_COLOR[group(k, task)] for k in keys]
    fig = go.Figure(go.Bar(
        x=vals, y=[name(k) for k in keys], orientation="h", marker=dict(color=cols, cornerradius=4),
        text=[fmt(v) for v in vals], textposition="outside", cliponaxis=False,
        customdata=[group(k, task) for k in keys],
        hovertemplate="<b>%{y}</b><br>%{customdata}<br>" + metric + ": %{x:.4f}<extra></extra>"))
    lo = min(vals)
    fig.update_xaxes(range=[max(0, lo * 0.8) if metric not in ("MCC",) else min(0, lo * 1.2), max(vals) * 1.12],
                     title=metric)
    return fig_layout(fig, height=height, showlegend=False)


def group_legend(task, keys):
    seen = []
    for k in keys:
        g = group(k, task)
        if g not in seen:
            seen.append(g)
    html("".join(f'<span class="pill"><span style="display:inline-block;width:10px;height:10px;border-radius:3px;'
                 f'background:{GROUP_COLOR[g]};margin-right:6px"></span>{g}</span>' for g in seen))
