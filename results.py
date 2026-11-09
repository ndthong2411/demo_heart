"""Trang kết quả: bảng đủ chỉ số, xếp hạng, chênh lệch có CI, từng mã."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (CODES, EVAL_CODES, EVALS, LOWER_BETTER, METRIC_INFO, METRICS, NEURAL, OUR, PUBLISHED_SEQ, curves,
                    deltas, fig_layout, fmt, group, group_legend, html, kpi, macro_table, name, ranking_bar)

CFG = {"displayModeBar": False}
TASKS = {"T2": "T2 · lượt khám sau (bài chính)", "T3": "T3 · trong 1 năm (khảo sát thêm)",
         "T1": "T1 · trong lượt nhập viện, bài RAD (khảo sát thêm)"}


def full_table(task, ev, tau):
    t = macro_table(task, ev, tau)
    t = t.sort_values("AUPRC", ascending=False)
    out = t.copy()
    out.insert(0, "Nhóm", [group(k, task) for k in t.index])
    out.index = [name(k) for k in t.index]
    out.index.name = "Phương pháp"
    return out


def style_table(df, task):
    ours = name(OUR[task])

    def hl(col):
        if col.name == "Nhóm":
            return [""] * len(col)
        best = col.min() if col.name in LOWER_BETTER else col.max()
        return ["font-weight:700;color:#13804f;background:#e8f6ef" if v == best else "" for v in col]

    def row(r):
        return ["background:#eaf2fc" if r.name == ours else ""] * len(r)

    return (df.style.apply(row, axis=1).apply(hl, axis=0)
            .format({c: "{:.3f}" for c in df.columns if c != "Nhóm"}, decimal=","))


def forest(task, ev, metric, tau):
    d = deltas(task, ev, metric, tau).copy()
    d = d.sort_values("value")
    if metric in LOWER_BETTER:
        d[["value", "lo", "hi"]] = -d[["value", "hi", "lo"]].values
    sig = (d.lo > 0) | (d.hi < 0)
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color="#7a7f8a", width=1.5))
    fig.add_trace(go.Scatter(
        x=d.value, y=[name(b) for b in d.b], mode="markers",
        error_x=dict(type="data", symmetric=False, array=d.hi - d.value, arrayminus=d.value - d.lo, color="#8a8f99",
                     thickness=2, width=0),
        marker=dict(size=11, color=["#2a78d6" if s else "#c9ccd2" for s in sig], line=dict(width=2, color="white")),
        customdata=d[["lo", "hi"]].values,
        hovertemplate="so với %{y}<br>chênh %{x:+.4f}<br>CI 95% [%{customdata[0]:+.4f}; %{customdata[1]:+.4f}]<extra></extra>"))
    fig.update_xaxes(title=f"Hệ cuối trừ phương pháp khác ({metric}{', đã đảo dấu' if metric in LOWER_BETTER else ''})")
    return fig_layout(fig, max(300, 34 * len(d) + 80), showlegend=False)


def per_code(task, ev, models):
    c = curves()[task][ev]
    pc = pd.DataFrame(c["perclass"])
    pc = pc[pc.method.isin(models)]
    return pc, c["curves"]


def page():
    html('<div class="kicker">Kết quả</div><div class="slide-title">Kết quả và so sánh với SOTA</div>')
    html('<div class="lead">Tất cả số trên trang này đọc từ kết quả đã lưu của V4–V5, không chạy lại mô hình. '
         'Chỉ số chính là <b>macro AP (AUPRC)</b>, trung bình trên 5 mã I10, I50, I21, I25, I20.</div>')

    f1, f2, f3, f4 = st.columns([1.5, 1.2, 1, 0.8])
    task = f1.selectbox("Bài toán", list(TASKS), format_func=TASKS.get)
    ev = f2.selectbox("Cách đánh giá", ["cv", "future"], format_func=EVALS.get)
    metric = f3.selectbox("Chỉ số để xếp hạng", METRICS, format_func=lambda m: METRIC_INFO[m][0])
    tau = None
    if task == "T1":
        tau = f4.selectbox("Mốc giờ", [2.0, 6.0, 12.0], index=1, format_func=lambda t: f"{t:.0f} giờ")
    if task != "T2":
        st.info("Đây là bài khảo sát thêm. Bài chính của báo cáo là T2.", icon="ℹ️")

    t = macro_table(task, ev, tau)
    ours, neural = OUR[task], NEURAL[task]
    base = "RAD system (published)" if task == "T1" else "hgb"
    seq = [k for k in t.index if k in PUBLISHED_SEQ]
    best_seq = max(seq, key=lambda k: t.loc[k, "AUPRC"]) if seq else None
    dl = deltas(task, ev, "AUPRC", tau).set_index("b")

    k = st.columns(4)
    with k[0]:
        kpi(fmt(t.loc[ours, "AUPRC"]), f"macro AP của hệ cuối · AUROC {fmt(t.loc[ours, 'AUROC'])}", "blue")
    with k[1]:
        r = dl.loc[base]
        kpi(fmt(r.value, sign=True), f"so với {name(base)} · CI [{fmt(r.lo, sign=True)}; {fmt(r.hi, sign=True)}]", "green" if r.lo > 0 else "")
    with k[2]:
        r = dl.loc[neural]
        kpi(fmt(r.value, sign=True), f"so với {name(neural)} · CI [{fmt(r.lo, sign=True)}; {fmt(r.hi, sign=True)}]", "green" if r.lo > 0 else "")
    with k[3]:
        if best_seq:
            r = dl.loc[best_seq]
            kpi(fmt(r.value, sign=True), f"so với mô hình chuỗi đã công bố tốt nhất ({name(best_seq)}) · CI [{fmt(r.lo, sign=True)}; {fmt(r.hi, sign=True)}]",
                "green" if r.lo > 0 else "")
        else:
            other = t["AUPRC"].drop([ours, neural, base]).idxmax()
            r = dl.loc[other]
            kpi(fmt(r.value, sign=True), f"so với đối thủ tốt nhất còn lại ({name(other)}) · CI [{fmt(r.lo, sign=True)}; {fmt(r.hi, sign=True)}]", "green" if r.lo > 0 else "")

    st.write("")
    tab1, tab2, tab3, tab4 = st.tabs(["Bảng đầy đủ mọi chỉ số", "Xếp hạng & chênh lệch", "Từng mã bệnh",
                                      "Giải thích chỉ số"])

    with tab1:
        df = full_table(task, ev, tau)
        st.dataframe(style_table(df, task), width="stretch", height=35 * (len(df) + 1) + 3)
        html('<div class="note">Ô xanh lá là tốt nhất trong cột (Brier: thấp nhất). '
             'Các chỉ số cần ngưỡng (Accuracy tới MCC) dùng chung quy tắc: số ca bị báo động bằng tỉ lệ ca dương trên '
             'validation. Accuracy của mọi phương pháp đều sát nhau vì ca dương chỉ khoảng 5%.</div>')
        st.download_button("Tải bảng (CSV)", df.to_csv().encode("utf-8-sig"),
                           file_name=f"{task}_{ev}{'' if tau is None else f'_{tau:.0f}h'}_metrics.csv")

    with tab2:
        l, r = st.columns([1, 1])
        with l:
            html(f'<div style="font-weight:700">Xếp hạng theo {METRIC_INFO[metric][0]}</div>')
            st.plotly_chart(ranking_bar(task, ev, metric, tau, height=max(380, 30 * len(t) + 60)), config=CFG,
                            width="stretch")
            group_legend(task, t.index)
        with r:
            html(f'<div style="font-weight:700">Hệ cuối hơn / kém bao nhiêu ({metric}), kèm CI 95%</div>')
            st.plotly_chart(forest(task, ev, metric, tau), config=CFG, width="stretch")
            html('<div class="note">Chấm xanh: khoảng tin cậy không chứa 0 (khác biệt có ý nghĩa). Chấm xám: chưa đủ '
                 'chắc. Giá trị dương là hệ cuối tốt hơn. CI tính bằng bootstrap theo bệnh nhân.</div>')

    with tab3:
        if task == "T1":
            st.info("Phần từng mã chỉ có cho T2 và T3 trong app này.")
        else:
            default = [ours, neural, "hgb"] + ([best_seq] if best_seq else [])
            models = st.multiselect("Phương pháp", list(t.index), default=default, format_func=name)
            if models:
                pc, cv = per_code(task, ev, models)
                ap = pc.pivot(index="method", columns="cls", values="AUPRC")[EVAL_CODES]
                ap.index = [name(m) for m in ap.index]
                ap.columns = [f"{c} · {CODES[c]}" for c in EVAL_CODES]
                html('<div style="font-weight:700">AP từng mã</div>')
                st.dataframe(ap.style.format("{:.3f}", decimal=",").highlight_max(axis=0, props="font-weight:700;color:#13804f;"
                                                                     "background:#e8f6ef"), width="stretch")
                code = st.radio("Mã", EVAL_CODES, horizontal=True, format_func=lambda c: f"{c} · {CODES[c]}")
                l, r = st.columns(2)
                pal = ["#2a78d6", "#8db7ea", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#e87ba4", "#008300"]
                with l:
                    fig = go.Figure()
                    for i, m in enumerate(models):
                        p = cv[code][m]["pr"]
                        fig.add_trace(go.Scatter(x=p["x"], y=p["y"], name=name(m), mode="lines",
                                                 line=dict(width=2.5 if m == ours else 2, color=pal[i % len(pal)])))
                    prev = cv[code][models[0]]["prevalence"]
                    fig.add_hline(y=prev, line=dict(color="#a3a29c", dash="dot"),
                                  annotation_text=f"đoán bừa = {fmt(prev)}", annotation_position="top right")
                    fig.update_xaxes(title="Recall", range=[0, 1])
                    fig.update_yaxes(title="Precision", range=[0, 1])
                    html(f'<div style="font-weight:700">Đường Precision–Recall · {code}</div>')
                    st.plotly_chart(fig_layout(fig, 400), config=CFG, width="stretch")
                with r:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#c9ccd2", dash="dot"),
                                             showlegend=False, hoverinfo="skip"))
                    mx = 0
                    for i, m in enumerate(models):
                        cal = pd.DataFrame(cv[code][m]["calibration"])
                        mx = max(mx, cal.predicted.max(), cal.observed.max())
                        fig.add_trace(go.Scatter(x=cal.predicted, y=cal.observed, name=name(m), mode="lines+markers",
                                                 line=dict(width=2, color=pal[i % len(pal)]), marker=dict(size=8),
                                                 customdata=cal.n,
                                                 hovertemplate="dự báo %{x:.3f} · thực tế %{y:.3f} · n=%{customdata}<extra></extra>"))
                    fig.update_xaxes(title="Xác suất dự báo trung bình", range=[0, min(1, mx * 1.1)])
                    fig.update_yaxes(title="Tỉ lệ thực tế", range=[0, min(1, mx * 1.1)])
                    html(f'<div style="font-weight:700">Hiệu chuẩn · {code} (càng sát đường chéo càng tốt)</div>')
                    st.plotly_chart(fig_layout(fig, 400), config=CFG, width="stretch")

    with tab4:
        rows = "".join(f"<tr><td><b>{METRIC_INFO[m][0]}</b></td><td>{METRIC_INFO[m][1]}</td></tr>" for m in METRICS)
        html(f'<div class="card"><table class="simple"><tr><th style="width:24%">Chỉ số</th><th>Ý nghĩa</th></tr>{rows}'
             '</table></div>')
        html('<div class="note"><b>Các phương pháp so sánh.</b> Mô hình cây: HGB, LightGBM, XGBoost, CatBoost trên bảng '
             '451 cột tóm tắt. Mô hình chuỗi EHR đã công bố, cài lại trên dữ liệu này: RETAIN (2016), Doctor AI (2016), '
             'GRU-D (2018), BEHRT (2020), STraTS (2022). Mốc đơn giản: hồi quy logistic và đoán theo tần suất mã. '
             'Riêng T1 so với hệ RAD đã công bố và các mô hình tabular mới (TabICL, TabM, RealMLP, FuseMoE).</div>')
