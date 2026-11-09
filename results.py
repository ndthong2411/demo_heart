"""Trang kết quả: bảng đủ chỉ số, xếp hạng, chênh lệch có CI, từng mã."""
import base64

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (EVAL_CODES, LOWER_BETTER, METRICS, NEURAL, OUR, PUBLISHED_SEQ, code_name, curves, deltas, evname,
                    fig_layout, fmt, gname, group, group_legend, html, kpi, lang_toggle, macro_table, minfo, name,
                    ranking_bar, tr)

CFG = {"displayModeBar": False}
TASKS = {"T2": ("T2 · lượt khám sau (bài chính)", "T2 · next visit (main task)"),
         "T3": ("T3 · trong 1 năm (khảo sát thêm)", "T3 · within 1 year (exploratory)"),
         "T1": ("T1 · trong lượt nhập viện, bài RAD (khảo sát thêm)", "T1 · within the admission, RAD study (exploratory)")}


def group_col():
    return tr("Nhóm", "Group")


def full_table(task, ev, tau):
    t = macro_table(task, ev, tau)
    t = t.sort_values("AUPRC", ascending=False)
    out = t.copy()
    out.insert(0, group_col(), [gname(group(k, task)) for k in t.index])
    out.index = [name(k) for k in t.index]
    out.index.name = tr("Phương pháp", "Method")
    return out


def style_table(df, task):
    ours = name(OUR[task])

    def hl(col):
        if col.name == group_col():
            return [""] * len(col)
        best = col.min() if col.name in LOWER_BETTER else col.max()
        return ["font-weight:700;color:#13804f;background:#e8f6ef" if v == best else "" for v in col]

    def row(r):
        return ["background:#eaf2fc" if r.name == ours else ""] * len(r)

    return (df.style.apply(row, axis=1).apply(hl, axis=0)
            .format({c: "{:.3f}" for c in df.columns if c != group_col()}, decimal=tr(",", ".")))


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
        hovertemplate=tr("so với %{y}<br>chênh %{x:+.4f}", "vs %{y}<br>difference %{x:+.4f}")
        + "<br>CI 95% [%{customdata[0]:+.4f}; %{customdata[1]:+.4f}]<extra></extra>"))
    flip = tr(", đã đảo dấu", ", sign flipped") if metric in LOWER_BETTER else ""
    fig.update_xaxes(title=tr("Hệ cuối trừ phương pháp khác", "Final system minus other method") + f" ({metric}{flip})")
    return fig_layout(fig, max(300, 34 * len(d) + 80), showlegend=False)


def per_code(task, ev, models):
    c = curves()[task][ev]
    pc = pd.DataFrame(c["perclass"])
    pc = pc[pc.method.isin(models)]
    return pc, c["curves"]


def page():
    head_l, head_r = st.columns([5, 1.2])
    head_l.markdown(f'<div class="kicker">{tr("Kết quả", "Results")}</div><div class="slide-title">'
                    f'{tr("Kết quả và so sánh với SOTA", "Results and comparison with SOTA")}</div>',
                    unsafe_allow_html=True)
    lang_toggle(head_r)
    html('<div class="lead">' + tr(
        'Tất cả số trên trang này đọc từ kết quả đã lưu của V4–V5, không chạy lại mô hình. '
        'Chỉ số chính là <b>macro AP (AUPRC)</b>, trung bình trên 5 mã I10, I50, I21, I25, I20.',
        'Every number on this page is read from the saved V4–V5 results; no model is rerun. '
        'The primary metric is <b>macro AP (AUPRC)</b>, averaged over the 5 codes I10, I50, I21, I25, I20.') + '</div>')

    f1, f2, f3, f4 = st.columns([1.5, 1.2, 1, 0.8])
    task = f1.selectbox(tr("Bài toán", "Task"), list(TASKS), format_func=lambda k: tr(*TASKS[k]))
    ev = f2.selectbox(tr("Cách đánh giá", "Evaluation"), ["cv", "future"], format_func=evname)
    metric = f3.selectbox(tr("Chỉ số để xếp hạng", "Ranking metric"), METRICS, format_func=lambda m: minfo(m)[0])
    tau = None
    if task == "T1":
        tau = f4.selectbox(tr("Mốc giờ", "Time point"), [2.0, 6.0, 12.0], index=1,
                           format_func=lambda t: f"{t:.0f} {tr('giờ', 'h')}")
    if task != "T2":
        st.info(tr("Đây là bài khảo sát thêm. Bài chính của báo cáo là T2.",
                   "This is an exploratory task. The main task of the report is T2."), icon="ℹ️")

    t = macro_table(task, ev, tau)
    ours, neural = OUR[task], NEURAL[task]
    base = "RAD system (published)" if task == "T1" else "hgb"
    seq = [k for k in t.index if k in PUBLISHED_SEQ]
    best_seq = max(seq, key=lambda k: t.loc[k, "AUPRC"]) if seq else None
    dl = deltas(task, ev, "AUPRC", tau).set_index("b")

    k = st.columns(4)
    with k[0]:
        kpi(fmt(t.loc[ours, "AUPRC"]), tr("macro AP của hệ cuối", "macro AP of the final system")
            + f" · AUROC {fmt(t.loc[ours, 'AUROC'])}", "blue")
    with k[1]:
        r = dl.loc[base]
        kpi(fmt(r.value, sign=True), f"{tr('so với', 'vs')} {name(base)} · CI [{fmt(r.lo, sign=True)}; {fmt(r.hi, sign=True)}]", "green" if r.lo > 0 else "")
    with k[2]:
        r = dl.loc[neural]
        kpi(fmt(r.value, sign=True), f"{tr('so với', 'vs')} {name(neural)} · CI [{fmt(r.lo, sign=True)}; {fmt(r.hi, sign=True)}]", "green" if r.lo > 0 else "")
    with k[3]:
        if best_seq:
            r = dl.loc[best_seq]
            kpi(fmt(r.value, sign=True), tr("so với mô hình chuỗi đã công bố tốt nhất", "vs the best published sequence model")
                + f" ({name(best_seq)}) · CI [{fmt(r.lo, sign=True)}; {fmt(r.hi, sign=True)}]",
                "green" if r.lo > 0 else "")
        else:
            other = t["AUPRC"].drop([ours, neural, base]).idxmax()
            r = dl.loc[other]
            kpi(fmt(r.value, sign=True), tr("so với đối thủ tốt nhất còn lại", "vs the best remaining competitor")
                + f" ({name(other)}) · CI [{fmt(r.lo, sign=True)}; {fmt(r.hi, sign=True)}]", "green" if r.lo > 0 else "")

    st.write("")
    tab1, tab2, tab3, tab4 = st.tabs([tr("Bảng đầy đủ mọi chỉ số", "Full table, all metrics"),
                                      tr("Xếp hạng & chênh lệch", "Ranking & differences"),
                                      tr("Từng mã bệnh", "Per code"), tr("Giải thích chỉ số", "Metric guide")])

    with tab1:
        df = full_table(task, ev, tau)
        st.dataframe(style_table(df, task), width="stretch", height=35 * (len(df) + 1) + 3)
        html('<div class="note">' + tr(
            'Ô xanh lá là tốt nhất trong cột (Brier: thấp nhất). '
            'Các chỉ số cần ngưỡng (Accuracy tới MCC) dùng chung quy tắc: số ca bị báo động bằng tỉ lệ ca dương trên '
            'validation. Accuracy của mọi phương pháp đều sát nhau vì ca dương chỉ khoảng 5%.',
            'Green cells are the best in their column (Brier: lowest). '
            'Threshold-based metrics (Accuracy to MCC) share one rule: the number of flagged cases equals the positive '
            'rate on validation. Accuracy is close for every method because positives are only about 5%.') + '</div>')
        # Link data: thay cho st.download_button (tệp tạm /media/... có lúc hỏng trên Streamlit Cloud).
        fname = f"{task}_{ev}{'' if tau is None else f'_{tau:.0f}h'}_metrics.csv"
        b64 = base64.b64encode(df.to_csv().encode("utf-8-sig")).decode()
        html(f'<div class="dl-row"><a class="dl-btn" href="data:text/csv;base64,{b64}" download="{fname}">'
             f'{tr("Tải bảng (CSV)", "Download table (CSV)")}</a></div>')

    with tab2:
        l, r = st.columns([1, 1])
        with l:
            html(f'<div style="font-weight:700">{tr("Xếp hạng theo", "Ranked by")} {minfo(metric)[0]}</div>')
            st.plotly_chart(ranking_bar(task, ev, metric, tau, height=max(380, 30 * len(t) + 60)), config=CFG,
                            width="stretch")
            group_legend(task, t.index)
        with r:
            html(f'<div style="font-weight:700">{tr("Hệ cuối hơn / kém bao nhiêu", "How much better / worse the final system is")}'
                 f' ({metric}), {tr("kèm CI 95%", "with 95% CI")}</div>')
            st.plotly_chart(forest(task, ev, metric, tau), config=CFG, width="stretch")
            html('<div class="note">' + tr(
                'Chấm xanh: khoảng tin cậy không chứa 0 (khác biệt có ý nghĩa). Chấm xám: chưa đủ '
                'chắc. Giá trị dương là hệ cuối tốt hơn. CI tính bằng bootstrap theo bệnh nhân.',
                'Blue dot: the confidence interval excludes 0 (significant difference). Grey dot: not certain enough. '
                'Positive values mean the final system is better. CIs from patient-level bootstrap.') + '</div>')

    with tab3:
        if task == "T1":
            st.info(tr("Phần từng mã chỉ có cho T2 và T3 trong app này.",
                       "Per-code results are available only for T2 and T3 in this app."))
        else:
            default = [ours, neural, "hgb"] + ([best_seq] if best_seq else [])
            models = st.multiselect(tr("Phương pháp", "Methods"), list(t.index), default=default, format_func=name)
            if models:
                pc, cv = per_code(task, ev, models)
                ap = pc.pivot(index="method", columns="cls", values="AUPRC")[EVAL_CODES]
                ap.index = [name(m) for m in ap.index]
                ap.columns = [f"{c} · {code_name(c)}" for c in EVAL_CODES]
                html(f'<div style="font-weight:700">{tr("AP từng mã", "AP per code")}</div>')
                st.dataframe(ap.style.format("{:.3f}", decimal=tr(",", ".")).highlight_max(axis=0, props="font-weight:700;color:#13804f;"
                                                                     "background:#e8f6ef"), width="stretch")
                code = st.radio(tr("Mã", "Code"), EVAL_CODES, horizontal=True,
                                format_func=lambda c: f"{c} · {code_name(c)}")
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
                                  annotation_text=f"{tr('đoán bừa', 'random guess')} = {fmt(prev)}",
                                  annotation_position="top right")
                    fig.update_xaxes(title="Recall", range=[0, 1])
                    fig.update_yaxes(title="Precision", range=[0, 1])
                    html(f'<div style="font-weight:700">{tr("Đường Precision–Recall", "Precision–Recall curve")} · {code}</div>')
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
                                                 hovertemplate=tr("dự báo", "predicted") + " %{x:.3f} · "
                                                 + tr("thực tế", "observed") + " %{y:.3f} · n=%{customdata}<extra></extra>"))
                    fig.update_xaxes(title=tr("Xác suất dự báo trung bình", "Mean predicted probability"),
                                     range=[0, min(1, mx * 1.1)])
                    fig.update_yaxes(title=tr("Tỉ lệ thực tế", "Observed rate"), range=[0, min(1, mx * 1.1)])
                    html(f'<div style="font-weight:700">{tr("Hiệu chuẩn", "Calibration")} · {code} '
                         f'{tr("(càng sát đường chéo càng tốt)", "(closer to the diagonal is better)")}</div>')
                    st.plotly_chart(fig_layout(fig, 400), config=CFG, width="stretch")

    with tab4:
        rows = "".join(f"<tr><td><b>{minfo(m)[0]}</b></td><td>{minfo(m)[1]}</td></tr>" for m in METRICS)
        html(f'<div class="card"><table class="simple"><tr><th style="width:24%">{tr("Chỉ số", "Metric")}</th>'
             f'<th>{tr("Ý nghĩa", "Meaning")}</th></tr>{rows}</table></div>')
        html('<div class="note">' + tr(
            '<b>Các phương pháp so sánh.</b> Mô hình cây: HGB, LightGBM, XGBoost, CatBoost trên bảng '
            '451 cột tóm tắt. Mô hình chuỗi EHR đã công bố, cài lại trên dữ liệu này: RETAIN (2016), Doctor AI (2016), '
            'GRU-D (2018), BEHRT (2020), STraTS (2022). Mốc đơn giản: hồi quy logistic và đoán theo tần suất mã. '
            'Riêng T1 so với hệ RAD đã công bố và các mô hình tabular mới (TabICL, TabM, RealMLP, FuseMoE).',
            '<b>Compared methods.</b> Tree models: HGB, LightGBM, XGBoost, CatBoost on the 451-column summary table. '
            'Published EHR sequence models, reimplemented on this data: RETAIN (2016), Doctor AI (2016), GRU-D (2018), '
            'BEHRT (2020), STraTS (2022). Simple baselines: logistic regression and code-frequency prior. '
            'T1 alone is compared with the published RAD system and recent tabular models (TabICL, TabM, RealMLP, '
            'FuseMoE).') + '</div>')
