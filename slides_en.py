"""English slides. Same order and content as slides.py; reuses its computations (ci, head, CFG, STATIC)."""
import base64
import random

import plotly.graph_objects as go
import streamlit as st

from common import (CODES_EN, EVAL_CODES, GROUP_COLOR, NEURAL, OUR, PUBLISHED_SEQ, card, context, curves, deltas,
                    fig_layout, flow, fmt, forecast, group_legend, html, kpi, macro_table, metrics, name, ranking_bar,
                    ul, vn, why, wrap)
from slides import CFG, STATIC, ci, head

KINDS_EN = {"Kết quả xét nghiệm": "Lab results", "Sinh hiệu": "Vital signs", "Mã chẩn đoán": "Diagnosis codes",
            "Lượt khám": "Visits"}
LABS_EN = {"Đường máu mao mạch": "Capillary glucose", "Creatinin": "Creatinine", "RBC(Hồng cầu)": "RBC",
           "PLT(Tiểu cầu)": "PLT", "HCT(Hematocrite)": "HCT", "eGFR (độ thanh lọc cầu thận)": "eGFR"}


def pts(x):
    """AP difference expressed in points on a 0–100 scale."""
    return f"{x * 100:+.2f}"


# =============================================================== 1. Opening
def s_intro():
    c = context()["corpus"]
    head("SeqCVD · progress report · 2 Nov 2026",
         "Reading lab histories to predict cardiovascular diagnoses at the next visit",
         "The deep learning model SeqLab reads a patient's full course of lab tests, vital signs and diagnosis history, "
         "then estimates how likely the next visit is to record a new cardiovascular diagnosis.")
    k = st.columns(4)
    with k[0]:
        kpi(vn(c["patients"]), "patients")
    with k[1]:
        kpi(vn(c["visits"]), "visits")
    with k[2]:
        kpi(f"{c['events'] / 1e3:.0f}k", "time-stamped events")
    with k[3]:
        kpi("2020 – 2025", "6 years of data")
    st.write("")
    a, b = st.columns(2)
    with a:
        card("Aim 1 · Predicting cardiovascular risk from visit history",
             "<p>Task <b>T2</b>: at the end of the current visit, predict which cardiovascular diagnoses never "
             "recorded before will appear at the next visit.</p>", "accent", "Proposal")
    with b:
        card("Aim 2 · Learning representations of lab sequences",
             "<p><b>SeqLab</b>: a Transformer encoder that reads every measurement together with its time, "
             "pretrained on unlabelled data.</p>", "accent", "Proposal")
    st.write("")
    flow([("1", "Data", "what we have, what we use, how it is processed"), ("2", "Task T2", "what to predict, when"),
          ("3", "SeqLab", "architecture and rationale"), ("4", "Evaluation", "data splits, metrics"),
          ("5", "Results", "vs HGB and SOTA"), ("6", "Next", "T1, T3, limitations")])


N_INTRO = """Today I am presenting SeqCVD, the deep learning on sequential data part of my thesis.

The proposal has two deep learning aims: first, predicting cardiovascular risk from visit history; second, learning representations of lab test sequences. I combine both into one concrete task called T2: given the record up to the end of this visit, predict whether the patient's next visit will record a new cardiovascular diagnosis.

The model I built is SeqLab. It is both the answer to aim 2 (how to represent lab sequences) and the tool for aim 1.

I will go in this order: data, task, model, evaluation, and only then results. At the end I briefly mention two side tasks, T1 and T3."""


# =============================================================== 2. Raw data
def s_raw():
    ctx = context()
    raw = ctx["raw"]
    B, C = raw["sources"]
    head("Data · 1/4", "Raw data: lab records from one hospital, 2020–2025",
         "Two sets of lab files with the same column layout, plus a table of diagnosis codes per visit and an imaging "
         "report table.")
    a, b, c = st.columns([1, 1, 1.25])
    with a:
        card("Table B · combined file",
             f"<p><b>{vn(B['rows'])}</b> rows<br>{vn(B['patients'])} patients · {vn(B['visits'])} visits<br>1 file</p>",
             tag="Source 1")
    with b:
        card("Table C · one file per patient",
             f"<p><b>{vn(C['rows'])}</b> rows<br>{vn(C['patients'])} patients · {vn(C['visits'])} visits<br>"
             f"{vn(C['files'])} files</p>", tag="Source 2")
    with c:
        card("The two sources overlap",
             f"<p>Merged, there are <b>{vn(raw['rows_before_dedup'])}</b> rows, of which "
             f"<b>{vn(raw['duplicate_rows_removed'])}</b> are duplicates. After deduplication, "
             f"{vn(raw['rows_after_dedup'])} results remain for {vn(raw['union_patients'])} patients.</p>",
             tag="After merging")
    st.write("")
    l, r = st.columns([1.15, 1])
    with l:
        html("""<div class="card"><h4>What each lab row contains</h4>
<table class="simple">
<tr><th>Group</th><th>Columns</th></tr>
<tr><td>Identifiers, time</td><td><code>mabn</code>, <code>mavaovien</code>, <code>maql</code>, <code>id_*</code>, <code>ddmmyyyy</code></td></tr>
<tr><td>Lab test</td><td><code>maxn</code>, <code>tenxn</code>, <code>idxetnghiem</code>, <code>ketqua</code></td></tr>
<tr><td>Vital signs</td><td><code>mach</code>, <code>huyetap</code>, <code>nhietdo</code>, <code>cannang</code>, <code>chieucao</code></td></tr>
<tr><td>Patient</td><td><code>tuoi</code>, <code>phai</code>, <code>loaibenhan</code></td></tr>
<tr><td>Diagnosis</td><td><code>maicd</code> (ICD code), <code>chandoan</code> (free text)</td></tr>
</table></div>""")
    with r:
        y = ctx["corpus"]["lab_years"]
        fig = go.Figure(go.Bar(x=list(y), y=list(y.values()), marker=dict(color="#2a78d6", cornerradius=4),
                               text=[f"{v / 1e3:.0f}k" for v in y.values()], textposition="outside", cliponaxis=False,
                               hovertemplate="Year %{x}: %{y:,} results<extra></extra>"))
        fig.update_yaxes(title="Lab results", range=[0, max(y.values()) * 1.18])
        html('<div style="font-weight:700;margin-bottom:-.4rem">Lab results per year</div>')
        st.plotly_chart(fig_layout(fig, 290), config=CFG, width="stretch")
    why("<p>The data are spread evenly over 6 years, so both kinds of test are possible: splitting by patient, and "
        "training on earlier years then testing on later ones. The second is closer to how the model would really be "
        "used.</p>")


N_RAW = """The raw data come as two sets of lab files. Table B is one large file; table C is thousands of small files, one per patient. They share the same columns but overlap heavily: almost 290 thousand rows are duplicates.

Each row is one lab result, with vital signs measured at the same time, age, sex, record type, ICD code and a free-text diagnosis field. Alongside that there is a table of diagnosis codes per visit and an imaging report table.

The number of results per year is fairly even, around 110 to 150 thousand. This matters because I want to test the model over time: train on earlier years, test on a later one."""


# =============================================================== 3. What to keep, what to drop
def s_select():
    k = context()["corpus"]["by_kind"]
    head("Data · 2/4", "Keep only what the doctor has at prediction time; drop anything that contains the answer",
         "Many columns would make the scores look better if included, but falsely so, because they reveal the "
         "diagnosis.")
    a, b = st.columns(2)
    with a:
        card("Kept", ul([
            f"<b>Lab results</b>: {k[0]['keys']} types, {vn(k[0]['events'])} results. This is the main source.",
            f"<b>Vital signs</b>: {k[1]['keys']} types (pulse, blood pressure, temperature, weight, height...).",
            f"<b>ICD codes from earlier visits</b>: {k[2]['keys']} distinct codes. Disease history is a strong signal.",
            f"<b>Visits</b> and treatment-episode type ({k[3]['keys']} types), taken per episode at prediction time.",
            "<b>Age at the visit</b> and <b>sex</b>.",
        ]), "keep")
    with b:
        card("Dropped", ul([
            "<b>Free-text diagnosis field</b> (<code>chandoan</code>) and <b>imaging conclusions</b>: they state the "
            "diagnosis outright.",
            "<b>ICD codes of the visit being predicted</b>: these are the answer.",
            "<b>Record type stored per visit</b>: reveals whether the patient is later admitted.",
            "<b>Lab table pre-filtered by ICD</b> (table A): which rows were kept depends on the diagnosis, i.e. on "
            "the label.",
        ]), "drop")
    why(ul([
        "This is a prediction task, so the input may only contain information available before the prediction time.",
        "An early run had exactly this kind of bug: the visit-type token of the current visit revealed a future "
        "inpatient admission. I removed that token and reran; results barely changed, but since then every column has "
        "been checked against this rule.",
    ]))


N_SELECT = """My rule for choosing data is simple: anything the doctor does not yet have at prediction time must not go in.

Kept: lab tests, vital signs, diagnosis codes from earlier visits, visit information, age and sex.

Dropped: the free-text diagnosis field and imaging conclusions, because they state the diagnosis directly. The record type stored per visit is also dropped, because it tells you whether the patient was later admitted. And table A was already filtered by ICD, so the mere presence of a row carries information about the label.

I did hit one bug of this kind in the first version. I fixed it and reran, and results barely changed. Since then I check every column against this rule."""


# =============================================================== 4. Processing
def s_process():
    ctx = context()
    raw, c = ctx["raw"], ctx["corpus"]
    head("Data · 3/4", "From over 1 million raw rows to one timeline per patient",
         "Each patient is represented as a time-ordered sequence of events. An event is one measurement, one diagnosis "
         "code or one visit.")
    flow([
        ("Step 1", "Merge B + C", f"{vn(raw['rows_before_dedup'])} rows"),
        ("Step 2", "Deduplicate", f"by order ID + test type, {vn(raw['duplicate_rows_removed'])} rows removed"),
        ("Step 3", "Attach time", "the first 12 digits of each ID encode year-month-day-hour-minute-second"),
        ("Step 4", "Age at visit", "the files store age at extraction (19 Mar 2025); it must be shifted back to the visit date"),
        ("Step 5", "Merge events", "labs + vital signs + ICD codes + visits"),
        ("Result", f"{vn(c['events'])} events", f"{vn(c['patients'])} patients · {vn(c['visits'])} visits"),
    ], highlight=(5,))
    st.write("")
    l, r = st.columns([1, 1.1])
    with l:
        kinds = c["by_kind"]
        lab = [KINDS_EN.get(k["name"], k["name"]) for k in kinds][::-1]
        val = [k["events"] for k in kinds][::-1]
        fig = go.Figure(go.Bar(x=val, y=lab, orientation="h", marker=dict(color="#2a78d6", cornerradius=4),
                               text=[vn(v) for v in val], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:,}<extra></extra>"))
        fig.update_xaxes(type="log", title="Events (log scale)", range=[4, 6.3])
        html('<div style="font-weight:700;margin-bottom:-.4rem">Composition of the event corpus</div>')
        st.plotly_chart(fig_layout(fig, 250), config=CFG, width="stretch")
    with r:
        html("""<div class="card"><h4>Example sequence excerpt (hypothetical patient)</h4>
<table class="simple">
<tr><th>Time</th><th>Kind</th><th>Code</th><th style="text-align:right">Value</th></tr>
<tr><td>03/2023 08:12</td><td>Visit</td><td>outpatient</td><td class="num">–</td></tr>
<tr><td>03/2023 08:40</td><td>Vital sign</td><td>systolic BP</td><td class="num">152</td></tr>
<tr><td>03/2023 09:05</td><td>Lab test</td><td>Creatinine</td><td class="num">1.4</td></tr>
<tr><td>03/2023 09:05</td><td>Lab test</td><td>NT-proBNP</td><td class="num">820</td></tr>
<tr><td>03/2023</td><td>ICD code</td><td>I10</td><td class="num">–</td></tr>
<tr><td>05/2023 07:55</td><td>Visit</td><td>inpatient</td><td class="num">–</td></tr>
<tr><td colspan="4" style="color:#7a7f8a">... continues up to the prediction time</td></tr>
</table></div>""")
    why(ul([
        "No fixed time grid and no imputation of missing values: an event exists only if the measurement really happened.",
        "Values are normalised using the distribution of the training portion only, one lab type at a time, so no "
        "information leaks from the test portion.",
    ]))


N_PROCESS = """Processing has a few main steps.

Merge the two sources, then deduplicate by order ID and test type, which leaves about 800 thousand results.

Time: I take it from the first 12 digits of the hospital ID, which encode year-month-day-hour-minute-second. That gives the sampling time, not just the date.

Age: the age column is age at data extraction in 2025, so it has to be shifted back to the visit date; otherwise a 2020 patient would be 5 years too old.

Finally, labs, vital signs, ICD codes and visits are merged into a single sequence per patient, about 920 thousand events in total. Labs make up most of them.

The key point: I do not use a time grid and I do not impute missing values. An event exists when a measurement was taken."""


# =============================================================== 5. Target ICD codes
ICD_GROUPS = [
    ("Hypertension", [("I10", "Essential (primary) hypertension")]),
    ("Coronary artery disease", [("I20", "Angina pectoris"), ("I21", "Acute myocardial infarction"),
                                 ("I25", "Chronic ischaemic heart disease")]),
    ("Heart failure", [("I50", "Heart failure")]),
    ("Heart rhythm", [("I48", "Atrial fibrillation and flutter"), ("I46", "Cardiac arrest")]),
]


def s_icd():
    t2 = context()["tasks"]["T2"]
    new, n = t2["new_by_code"], t2["samples"]
    head("Data · 4/4", "Prediction targets: 7 cardiovascular ICD-10 codes",
         "ICD codes are grouped at the 3-character level, e.g. I21.0 to I21.9 all count as I21. Each code is a separate "
         "yes/no question: is this code recorded for the first time at the next visit?")
    cols = st.columns([1, 2.4, 1, 1.7])
    for col, (grp, codes) in zip(cols, ICD_GROUPS):
        tiles = ""
        for code, nm in codes:
            main = code in EVAL_CODES
            badge = ('<span class="pill b">main metric</span>' if main
                     else '<span class="pill">too few, excluded</span>')
            tiles += (f'<div class="card {"accent" if main else ""}" style="flex:1 1 120px;padding:.8rem 1rem">'
                      f'<div style="font-size:1.6rem;font-weight:800;color:{"#1d5fae" if main else "#7a7f8a"}">{code}</div>'
                      f'<div style="font-weight:600;color:#16181d;margin:.1rem 0 .35rem 0">{nm}</div>'
                      f'<div style="color:#4a4f5a;font-size:.92rem">{vn(new[code])} new cases · '
                      f'{fmt(new[code] / n * 100, 1)}%</div><div style="margin-top:.35rem">{badge}</div></div>')
        with col:
            html(f'<div class="kicker" style="color:#4a4f5a">{grp}</div>'
                 f'<div style="display:flex;gap:.6rem;flex-wrap:wrap;margin-top:.5rem">{tiles}</div>')
    html(f'<div class="note">New cases counted over the {vn(n)} T2 visit pairs. "New" means the code has never appeared '
         'in any earlier visit or the current visit of that patient.</div>')
    st.write("")
    a, b = st.columns(2)
    with a:
        card("Why these 7 codes", "<p>The same code set as the RAD study, so the studies are comparable. These are the 7 "
             "chapter IX codes (diseases of the circulatory system) with the most admissions in the RAD cohort, from "
             "114 admissions (I46) to 2,072 (I10), covering the four main groups of cardiovascular disease.</p>")
    with b:
        card("Why the primary metric uses only 5 codes", "<p>Rule set in advance: a code needs at least 50 new cases "
             "overall and at least 10 new cases in every yearly test period. I48 (55 cases) and I46 (43 cases) do not "
             "qualify. With a handful of cases per period their AP swings too much and skews the average. The model "
             "still predicts all 7 codes.</p>")
    why("<p>Grouping at 3 characters gives each code enough cases while keeping a clear clinical meaning. Finer groups, "
        "e.g. splitting infarction into anterior and inferior wall, would leave only a few dozen cases each, not enough "
        "to train or evaluate.</p>")


N_ICD = """Before the task itself, let me be clear about what the model predicts.

There are 7 cardiovascular ICD-10 codes in four groups: hypertension; coronary artery disease, covering angina, acute myocardial infarction and chronic ischaemic heart disease; heart failure; and rhythm disorders, covering atrial fibrillation and cardiac arrest.

Codes are grouped at the 3-character level; for example every kind of acute myocardial infarction counts as I21. Finer groups would have too few cases.

I use the same code set as the RAD study so that the studies can be compared. These are the 7 cardiovascular codes with the most admissions in the cohort.

The model predicts all 7 codes, but the headline score uses 5. Atrial fibrillation and cardiac arrest have only 55 and 43 new cases; spread over 4 test years that is a few cases per year, and AP would jump around. I set this rule before looking at any results."""


# =============================================================== 6. Task T2
def timeline_fig():
    fig = go.Figure()
    visits = [(0, 3, "Visit 1", "I10"), (104, 110, "Visit 2 (current)", "I25"), (156, 160, "Visit 3 (+40 days)", "I20")]
    for x0, x1, lab, code in visits:
        future = x0 > 120
        fig.add_shape(type="rect", x0=x0, x1=x1 + 6, y0=0.25, y1=0.75, line=dict(width=0),
                      fillcolor="#fde3d7" if future else "#dbe9fb")
        fig.add_annotation(x=(x0 + x1 + 6) / 2, y=1.05, text=f"<b>{lab}</b>", showarrow=False, font=dict(size=13))
        fig.add_annotation(x=(x0 + x1 + 6) / 2, y=0.5, text=f"records <b>{code}</b>", showarrow=False,
                           font=dict(size=13, color="#c2410c" if future else "#1d5fae"))
    random.seed(3)
    xs = [random.uniform(0, 3) for _ in range(6)] + [random.uniform(104, 116) for _ in range(9)]
    fig.add_trace(go.Scatter(x=xs, y=[random.uniform(-0.25, 0.05) for _ in xs], mode="markers",
                             marker=dict(size=8, color="#2a78d6", line=dict(width=2, color="white")),
                             hoverinfo="skip", showlegend=False))
    fig.add_annotation(x=58, y=-0.12, text="lab tests, vital signs (blue dots)", showarrow=False,
                       font=dict(size=12, color="#4a4f5a"))
    fig.add_vline(x=124, line=dict(color="#16181d", width=2, dash="dash"))
    fig.add_annotation(x=124, y=1.38, text="<b>Prediction time</b>", showarrow=False, bgcolor="white",
                       font=dict(size=13))
    fig.add_annotation(x=60, y=1.38, text="← the model sees", showarrow=False, font=dict(color="#1d5fae"))
    fig.add_annotation(x=152, y=1.38, text="must predict →", showarrow=False, font=dict(color="#c2410c"))
    fig.update_xaxes(range=[-5, 172], title="days", showgrid=False)
    fig.update_yaxes(visible=False, range=[-0.35, 1.5])
    return fig_layout(fig, 250, showlegend=False)


def s_task():
    t2 = context()["tasks"]["T2"]
    head("Task", "T2: which never-before-seen cardiovascular diagnoses will the next visit record?",
         "Each sample is a pair of consecutive visits of the same patient. The current visit and everything before it "
         "are the input; the next visit provides the answer.")
    st.plotly_chart(timeline_fig(), config=CFG, width="stretch")
    l, r = st.columns([1, 1.15])
    with l:
        html("""<div class="card"><h4>Labels for the example above</h4>
<table class="simple"><tr><th>Code</th><th>Seen before?</th><th>Next visit records it?</th><th>Label</th></tr>
<tr><td>I10</td><td>yes</td><td>yes</td><td><b>0</b> (no longer new)</td></tr>
<tr><td>I25</td><td>yes</td><td>no</td><td><b>0</b></td></tr>
<tr><td>I20</td><td>no</td><td>yes</td><td><b style="color:#c2410c">1</b> (new code)</td></tr>
<tr><td>I50, I21, ...</td><td>no</td><td>no</td><td><b>0</b></td></tr></table>
<p style="margin-top:.6rem">One visit can have several new codes at once, so each code is a separate yes/no question.</p></div>""")
    with r:
        codes = list(t2["new_by_code"])
        n = [t2["new_by_code"][c] for c in codes]
        cols = ["#2a78d6" if c in EVAL_CODES else "#c9ccd2" for c in codes]
        fig = go.Figure(go.Bar(x=codes, y=n, customdata=[CODES_EN[c] for c in codes],
                               marker=dict(color=cols, cornerradius=4), cliponaxis=False,
                               text=[f"{v} ({fmt(v / t2['samples'] * 100, 1)}%)" for v in n], textposition="outside",
                               hovertemplate="%{x} · %{customdata}<br>%{y} new cases<extra></extra>"))
        fig.update_yaxes(title="Cases with a new code", range=[0, max(n) * 1.22])
        html(f'<div style="font-weight:700;margin-bottom:-.4rem">Positive cases among {vn(t2["samples"])} visit pairs '
             f'({vn(t2["patients"])} patients)</div>')
        st.plotly_chart(fig_layout(fig, 270), config=CFG, width="stretch")
    why(ul([
        "This is the real question when a patient comes back, and it directly addresses aim 1 of the proposal.",
        "The label is clear-cut: a code recorded at the next visit that was never recorded before. The two visits are "
        "at least 1 day apart (median about 37 days).",
        "I48 and I46 have too few cases (55 and 43), so they are still predicted but left out of the primary average; "
        "the primary metric averages over the 5 blue codes.",
    ]))


N_TASK = """This is the main task. Look at the example: this patient had hypertension recorded at visit 1, and chronic ischaemic heart disease added at visit 2. I stand at the end of visit 2; the model sees everything before the dashed line.

The question is: will the next visit, here 40 days later, record any cardiovascular code never seen before? Here it is I20, angina. I10 and I25 were already there, so they do not count as new.

I have 12,514 such visit pairs from nearly 2,500 patients. Positives are rare: only about 2 to 7% per code. I48 and I46 have only a few dozen cases, so I keep them out of the average so the headline number is not noisy.

One thing to be clear about: the label is a newly recorded code, not the date the disease started. And only patients who come back can be evaluated."""


# =============================================================== 7. Why read sequences
def s_why_seq():
    head("Idea", "Why read the sequence instead of summarising history into a table?",
         "The usual approach computes min, max, mean and last value for each lab test and feeds them to a tree model "
         "(HGB). It is fast and strong, but it loses order and timing.")
    l, r = st.columns([1.2, 1])
    with l:
        x = ["Test 1", "Test 2", "Test 3", "Test 4"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[1.0, 1.2, 1.6, 1.4], name="Patient A: rose, now falling",
                                 mode="lines+markers", line=dict(color="#2a78d6", width=2.5), marker=dict(size=9)))
        fig.add_trace(go.Scatter(x=x, y=[1.6, 1.2, 1.0, 1.4], name="Patient B: fell, now rising",
                                 mode="lines+markers", line=dict(color="#eb6834", width=2.5), marker=dict(size=9)))
        fig.update_yaxes(title="Creatinine (mg/dL)", range=[0.8, 1.8])
        html('<div style="font-weight:700;margin-bottom:-.4rem">Two creatinine courses (hypothetical example)</div>')
        st.plotly_chart(fig_layout(fig, 300), config=CFG, width="stretch")
        html('<div class="note">Both have the same <b>min 1.0 · max 1.6 · mean 1.3 · last value 1.4 · 4 tests</b>. '
             'A summary table sees these two patients as identical, even though their trends point in opposite '
             'directions.</div>')
    with r:
        card("Three difficulties of this data", ul([
            "<b>Irregular sampling</b>: every patient and every test has its own rhythm.",
            "<b>Very sparse</b>: there are 353 lab types, but each visit runs only a few dozen.",
            "<b>Rare positives</b>: 2–7% per code, i.e. only a few hundred cases to learn from.",
        ]))
        st.write("")
        card("Two ways to read history", "<p><b>HGB</b>: a 451-column summary table (60 common labs × statistics, most "
             "recent value, past codes, vital signs, age, sex).<br><br><b>SeqLab</b>: reads the event sequence "
             "directly, each value paired with its measurement time.</p>")
    why("<p>A sequence keeps the trend and timing that a summary table throws away. Whether it actually predicts "
        "better has to be tested, which is why HGB is always the main comparator.</p>")


N_WHY_SEQ = """This is why I took the sequence route.

The common approach summarises each lab test into a few numbers: min, max, mean, last value, and feeds them to a tree model such as HGB. It is very strong, and I keep it as the main competitor.

But look at these two patients: the same 4 creatinine measurements, the same min, max, mean and last value. To a summary table they are identical. In reality one is falling and the other is rising.

Hospital data has three more difficulties: irregular sampling, extreme sparsity, and rare positives. A sequence model handles the first two naturally. The third I tackle with pretraining, on a later slide.

I do not assume the sequence approach must win. That is the question to be tested."""


# =============================================================== 8. Architecture overview
@st.cache_resource
def arch_data_uri_en():
    return "data:image/png;base64," + base64.b64encode((STATIC / "seqcvd_architecture_en.png").read_bytes()).decode()


def s_overview():
    head("Model · overview", "SeqCVD architecture: from patient history to 7 probabilities",
         "Read left to right: events → embedding → SeqLab encoder → pretraining (①) and fine-tuning for T2 (②). "
         "The bottom row is the final system: an ensemble of 21 SeqLab networks blended with HGB.")
    html(f'<img src="{arch_data_uri_en()}" alt="SeqCVD architecture diagram" '
         'style="width:100%;height:auto;display:block;border-radius:8px;background:#fff">')
    links = "".join(
        f'<a class="dl-btn" href="app/static/{f}" download="{f}" target="_blank">{lab}</a>'
        for lab, f in [("Download PNG (English)", "seqcvd_architecture_en.png"),
                       ("Download PDF (English)", "seqcvd_architecture_en.pdf"),
                       ("Download PNG (Vietnamese)", "seqcvd_architecture_vi.png")])
    html(f'<div class="dl-row">{links}</div>')


N_OVERVIEW = """This is the big picture; I will follow the figure from left to right.

The first box is the input: the patient's full history up to the prediction time τ, covering labs, vital signs, ICD codes and visits, plus age and sex. Each event is a triple: what was measured, the value, and when.

Yellow box: each event is embedded with five components that are summed into a 96-dimensional vector.

Blue box: the SeqLab encoder, three Transformer layers followed by attention pooling into a single vector z that represents the whole patient.

On the right are two stages. Stage one, in green, is label-free pretraining: mask values and reconstruct them, and forecast the next blood draw. Stage two, in orange, is fine-tuning for T2 with 7 outputs.

The bottom row is the final system: average 21 SeqLab networks and blend them with HGB, which reads the summary table; the blending weight is chosen on validation. The output is 7 probabilities; the headline score uses the 5 highlighted codes.

The next slides go into each part and explain why it was chosen."""


# =============================================================== 9. SeqLab
def s_arch():
    head("Model · SeqLab", "SeqLab: each event becomes a token, a Transformer reads the whole sequence, 7 probabilities come out",
         "Builds on the STraTS idea (each measurement is a type–value–time triple) and extends it to read diagnosis "
         "codes and visits in the same sequence.")
    flow([
        ("Input", "≤ 256 most recent events", "+ age, sex. When truncating, keep up to the 128 most recent codes/visits first"),
        ("Token", "5 components", "measurement type · value · time · event kind · part of current visit?"),
        ("Embedding", "96-dim vector", "one embedding per component, summed"),
        ("Transformer", "3 layers × 4 heads", "every event can attend to every other event in the history"),
        ("Pooling", "attention pooling", "learns a weight per event → patient vector z"),
        ("Output", "7 sigmoids", "one probability per code, no need to sum to 1"),
    ], highlight=(1, 4))
    st.write("")
    html("""<div class="card"><h4>Each choice and its reason</h4><table class="simple">
<tr><th style="width:28%">Choice</th><th>Why</th></tr>
<tr><td>Type–value–time tokens</td><td>No time grid, no imputation of missing values. Irregular sampling stops being a problem.</td></tr>
<tr><td>ICD codes and visits in the sequence</td><td>Disease history is the strongest signal for a new code; in one sequence the model learns the links between labs and diagnoses by itself.</td></tr>
<tr><td>Truncation that favours codes/visits</td><td>Long-term patients have thousands of results; keeping only the 256 most recent results would lose the whole disease history.</td></tr>
<tr><td>Transformer</td><td>Lab tests months apart can relate directly, without passing through every step as in an RNN.</td></tr>
<tr><td>Attention pooling</td><td>The model picks out the important events (e.g. a high troponin) instead of averaging everything.</td></tr>
<tr><td>Small model (96 dims, 3 layers)</td><td>With only a few thousand patients, a compact model overfits less. A longer window (512 events) was tried and was not better.</td></tr>
</table></div>""")


N_ARCH = """This is the SeqLab architecture, left to right.

Each event becomes a token with five parts: which test it is, its value, when it was measured, what kind of event it is, and whether it belongs to the current visit. Each part has its own embedding, summed into a 96-dimensional vector.

A 3-layer Transformer lets every event attend to every other. A weighted pooling layer then produces a single vector representing the whole record, and 7 sigmoid heads give 7 probabilities.

This part is inherited from STraTS, and I say so clearly so as not to claim it as my own. What I added: putting diagnosis codes and visits into the same sequence, a truncation scheme that preserves disease history, the next-blood-draw forecasting pretraining task, and the way the final system is combined.

The model is deliberately small, because the data cover only a few thousand patients."""


# =============================================================== 10. Pretraining
def s_pretrain():
    f = forecast()
    better = int((f.mse_model < f.mse_locf).sum())
    gain = 1 - f.mse_model.sum() / f.mse_locf.sum()
    head("Model · pretraining", "Pretrain on unlabelled data first, then learn task T2",
         "T2 labels give only a few hundred positives per code, but the corpus holds tens of thousands of history "
         "windows that need no label. SeqLab first learns to \"read\" lab tests from those.")
    a, b, c = st.columns(3)
    with a:
        card("Task 1 · Mask and reconstruct", "<p>Randomly mask part of the values in the sequence and make the model "
             "recover them from the surrounding events. Like the way BERT learns language.</p>", "accent",
             "Self-supervised")
    with b:
        card("Task 2 · Forecast the next blood draw", "<p>From the history, predict the result of the first lab test / "
             "vital sign afterwards (within 30 days). This forces the model to understand trends.</p>", "accent",
             "Self-supervised")
    with c:
        card("Then · Fine-tune for T2", "<p>Keep the pretrained encoder, attach a 7-code head and keep training with T2 "
             "labels. One group of encoders also learns an extra masked-ICD-code task.</p>", "accent", "Supervised")
    st.write("")
    l, r = st.columns([1.35, 1])
    with l:
        top = f.sort_values("n", ascending=False).head(14).iloc[::-1]
        g = top.mse_gain_vs_locf * 100
        labs = [LABS_EN.get(x, x) for x in top["name"]]
        labs = [x.split("(")[0].strip() for x in labs]
        fig = go.Figure(go.Bar(x=g, y=labs, orientation="h", cliponaxis=False,
                               marker=dict(color=["#2a78d6" if v > 0 else "#e34948" for v in g], cornerradius=4),
                               text=[f"{v:.0f}%" for v in g], textposition="outside",
                               hovertemplate="%{y}<br>error reduced by %{x:.1f}%<extra></extra>"))
        fig.update_xaxes(title="Error reduction vs \"repeat the last value\" (%)", range=[0, max(g) * 1.2])
        html('<div style="font-weight:700;margin-bottom:-.4rem">Next-blood-draw forecasting, 14 most frequent lab tests</div>')
        st.plotly_chart(fig_layout(fig, 400), config=CFG, width="stretch")
    with r:
        kpi(f"{gain * 100:.0f}%", "lower error (MSE) than repeating the last value", "blue")
        st.write("")
        kpi(f"{better}/{len(f)}", "lab tests forecast better", "green")
        st.write("")
        html('<div class="note">The baseline is "the value will equal the previous one", a hard baseline to beat for '
             'lab tests. Creatinine is roughly level (mean absolute error 0.346 vs 0.345 mg/dL).</div>')
    why("<p>This is the evidence for aim 2: the learned SeqLab representation already contains information about how "
        "lab values evolve, before it has seen any disease label. In task T1, almost all of SeqLab's benefit also "
        "comes from this pretraining step.</p>")


N_PRETRAIN = """The problem with T2 is that labels are scarce: only a few hundred positives per code. Unlabelled data, however, is plentiful: each split has about 45 thousand history windows usable for pretraining.

So I first have SeqLab do two self-supervised tasks. Task one masks values and reconstructs them, like BERT. Task two predicts the result of the next blood draw. Task two forces the model to understand trends rather than just memorise values.

The result of task two is itself the evidence for aim 2 of the proposal: compared with simply repeating the previous value, the error drops by about 29%, and it is better for 72 of 76 lab tests. Creatinine is roughly level, because creatinine is inherently stable.

Only after that do I attach the 7-code prediction head and fine-tune with T2 labels."""


# =============================================================== 11. Final system
def s_system():
    head("Model · final system", "Final system: average of 21 SeqLab networks, blended with HGB",
         "A single network varies quite a lot with the random seed. Averaging many networks is more stable, and "
         "blending with trees keeps it robust when training data are scarce.")
    flow([
        ("SeqLab branch", "21 networks", "9 trained on T2 alone · 3 from encoders with masked ICD codes · 9 trained jointly on T2 + a 1-year side task"),
        ("", "Average", "mean score of the 21 networks"),
        ("Tree branch", "HGB", "451-column summary table, average of 5 HGB models per code"),
        ("Blend", "w·SeqLab + (1−w)·HGB", "blended on the logit scale; w chosen on validation for each split"),
        ("Result", "7 probabilities", "one per code"),
    ], highlight=(3,))
    st.write("")
    a, b = st.columns(2)
    with a:
        card("Why many networks", ul([
            "Same configuration, different seed, and AP changes noticeably. Averaging many networks reduces this noise.",
            "The benefit saturates fairly early: in task T1, going from 12 to 15–18 networks did not help.",
            "Mixing different training schemes (alone, masked codes, joint) adds more diversity than changing seeds only.",
        ]))
    with b:
        card("Why blend with HGB", ul([
            "Trees are very strong on tabular data and robust with little data; networks need more data.",
            "The two branches read the data in different ways, so they make different mistakes and compensate for each other.",
            "The weight is chosen on validation only, never on test. In some periods the system chose 100% SeqLab, so it is "
            "not always 50–50.",
        ]))
    why("<p>In the experiments, SeqLab alone is only on par with HGB. Blended, it clearly beats HGB under both "
        "evaluation schemes (results slide). So the two ways of reading history truly complement each other.</p>")


N_SYSTEM = """The final system has two branches.

The SeqLab branch is the average of 21 networks. The reason is that a single network is very sensitive to the seed; change the seed and AP moves up and down. When I increased the number of networks, the benefit saturated at around 12 to 21. These 21 networks are also not identical: one group is trained on T2 alone, one uses encoders that also learned a masked ICD code task, and one is trained jointly on T2 with a one-year side task.

The second branch is HGB on the 451-column summary table.

The two branches are blended on the logit scale with a weight chosen on validation. I do not look at test when choosing. In some periods the system chose 100% SeqLab by itself.

Why blend: SeqLab alone is only on par with HGB, but the two make mistakes in different places, so together they beat both."""


# =============================================================== 12. Evaluation
def rolling_fig():
    ctx = context()
    rows = [s for s in ctx["multitask_fixed_splits"] if s["task"] == "T2"]
    labels = {"roll2021": "2021", "roll2022": "2022", "roll2023": "2023", "roll2024-25": "2024–25"}
    end = {"roll2021": 2021, "roll2022": 2022, "roll2023": 2023, "roll2024-25": 2024}
    fig = go.Figure()
    for s in rows:
        y = f"Test period {labels[s['split']]}"
        e = end[s["split"]]
        w = 2 if s["split"] == "roll2024-25" else 1
        fig.add_trace(go.Bar(y=[y], x=[e - 2020], base=[2020], orientation="h", marker=dict(color="#8db7ea", cornerradius=4),
                             name="Train + select (past)", showlegend=s["split"] == "roll2021",
                             hovertemplate=f"Train: {vn(s['train']['samples'])} samples · select: {vn(s['validation']['samples'])} samples<extra></extra>"))
        fig.add_trace(go.Bar(y=[y], x=[w], base=[e], orientation="h", marker=dict(color="#eb6834", cornerradius=4),
                             name="Test", showlegend=s["split"] == "roll2021",
                             text=[f"{vn(s['test']['samples'])} samples"], textposition="inside",
                             hovertemplate=f"Test: {vn(s['test']['samples'])} samples<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(range=[2020, 2026], dtick=1, title="year")
    fig.update_yaxes(autorange="reversed")
    return fig_layout(fig, 260)


def s_eval():
    head("Evaluation", "Two ways to test, one primary metric",
         "Both keep the test data fully separate from the training data. Every model choice is made on validation only.")
    l, r = st.columns(2)
    with l:
        card("Scheme 1 · Patient-level 5-fold CV", "<p>Patients are split into 5 groups. Each round uses 3 groups to "
             "train, 1 to select, 1 to test, then rotates. A patient is never in both training and test.</p>", "accent")
        z = go.Figure()
        roles = ["train", "train", "train", "select", "test"]
        col = {"train": "#8db7ea", "select": "#c9ccd2", "test": "#eb6834"}
        for i in range(5):
            rr = roles[-i:] + roles[:-i] if i else roles
            for j, role in enumerate(rr):
                z.add_trace(go.Bar(y=[f"Round {i + 1}"], x=[1], base=[j], orientation="h", showlegend=(i == 0 and j in (0, 3, 4)),
                                   name=role, marker=dict(color=col[role], line=dict(color="white", width=2)),
                                   hovertemplate=f"Group {j + 1}: {role}<extra></extra>"))
        z.update_layout(barmode="overlay")
        z.update_xaxes(visible=False)
        z.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_layout(z, 220), config=CFG, width="stretch")
    with r:
        card("Scheme 2 · Train on the past, test on the future", "<p>Each period trains only on earlier years and tests "
             "on the following year. The 4 test periods are pooled to compute metrics. This is closest to real "
             "deployment, and harder than CV.</p>", "accent")
        st.plotly_chart(rolling_fig(), config=CFG, width="stretch")
    a, b = st.columns([1.1, 1])
    with a:
        card("Primary metric: macro AP (AUPRC)", "<p>Compute AP for each of the 5 main codes and average. AP measures "
             "whether the cases with a new code are ranked at the top of the list. Random guessing gives an AP equal to "
             "the positive rate, about 0.05.</p>")
    with b:
        card("Secondary metrics (Results tab)", "<p>AUROC, Accuracy, Balanced acc., Recall, Specificity, Precision, NPV, "
             "F1, MCC, Brier. For threshold-based metrics, every method flags exactly as many cases as the positive rate "
             "on validation. 95% confidence intervals by patient-level bootstrap.</p>")
    why("<p>Positives are only about 5%, so a model that always says \"no\" still reaches about 95% Accuracy. AP is not "
        "fooled that way, so I use AP as the primary measure, and every keep / drop decision is based on validation "
        "AP.</p>")


N_EVAL = """I test in two ways.

The first is 5-fold CV split by patient: 3 folds to train, 1 to select the model, 1 to test, rotating. Splitting by patient means nobody appears in both training and test.

The second is harder and more realistic: train only on earlier years, test on the following year. There are 4 periods: 2021, 2022, 2023 and 2024–25. The 2021 period has only one year of training data, so it is very hard.

The primary metric is macro AP: AP per code, averaged over the 5 codes. The reason: positives are only 5%, so Accuracy is nearly meaningless. A model that says "no" to everyone still gets 95%.

I still report all other metrics in the Results tab. For a fair comparison, every method uses the same thresholding rule: the number of flagged cases equals the positive rate on validation."""


# =============================================================== 13. Main result
def s_result():
    head("Results", "The SeqLab + HGB system has the highest AP under both evaluation schemes")
    task = "T2"
    rows = []
    for ev in ("cv", "future"):
        t = macro_table(task, ev)["AUPRC"]
        best_seq = max((k for k in t.index if k in PUBLISHED_SEQ), key=lambda k: t[k])
        rows.append((ev, t, best_seq))
    l, r = st.columns([1.35, 1])
    with l:
        fig = go.Figure()
        order = [("hgb", "HGB"), ("best", "Best published sequence model"), (NEURAL[task], "SeqLab"),
                 (OUR[task], "SeqLab + HGB")]
        colors = {"hgb": GROUP_COLOR["Mô hình cây"], "best": GROUP_COLOR["Mô hình chuỗi EHR đã công bố"],
                  NEURAL[task]: GROUP_COLOR["SeqLab"], OUR[task]: GROUP_COLOR["Hệ cuối"]}
        xs = ["Patient-level CV", "Future years"]
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
        html('<div style="font-weight:700;margin-bottom:-.4rem">macro AP over 5 codes (y axis starts at 0.10)</div>')
        st.plotly_chart(fig_layout(fig, 380), config=CFG, width="stretch")
    with r:
        v, lo, hi = ci(task, "cv", "hgb")
        kpi(f"{pts(v)} AP points", f"above HGB in CV · 95% CI [{pts(lo)}; {pts(hi)}]", "blue")
        st.write("")
        v, lo, hi = ci(task, "future", "hgb")
        kpi(f"{pts(v)} AP points", f"above HGB in future years · 95% CI [{pts(lo)}; {pts(hi)}]", "blue")
        st.write("")
        cv_best, fu_best = rows[0][2], rows[1][2]
        v1 = ci(task, "cv", cv_best)[0]
        v2 = ci(task, "future", fu_best)[0]
        kpi(f"{pts(v1)} / {pts(v2)}", f"above the best published sequence model ({name(cv_best)} / {name(fu_best)}), "
            "CV / future years", "green")
    why(ul([
        "The confidence intervals of all the differences above lie above 0.",
        "SeqLab alone beats HGB by about +0.5 points, but its confidence interval still touches 0, so I only claim it "
        "is on par with HGB. The clear benefit comes from blending the two branches.",
        "In future years every model's AP drops compared with CV; the ranking stays the same.",
    ]), "How to read this result")


N_RESULT = """This is the main T2 result. The axis is macro AP.

In CV, the SeqLab + HGB system reaches about 0.159, 1.2 points above HGB, with a confidence interval from 0.8 to 1.7 that does not touch 0. In future years the margin is smaller, about 0.6 points, but still significant.

Compared with the best sequence model from published work, the system is about 2 points higher in CV and more than 1 point higher in future years.

I want to be honest about one point: SeqLab alone is only on par with HGB, not clearly better. The significant benefit appears when the two branches are blended. My reading is that the sequence holds information the summary table lacks, but trees still exploit the summary table better."""


# =============================================================== 14. SOTA
def s_sota():
    head("Comparison", "Ranking 13 methods on the same data and the same splits",
         "The published EHR sequence models (RETAIN, Doctor AI, GRU-D, BEHRT, STraTS) were reimplemented on our data, "
         "with the same event inputs and the same test sets. The full table of all metrics is in the Results tab.")
    ev = st.radio("Evaluation", ["cv", "future"], format_func=lambda e: {"cv": "Patient-level CV",
                  "future": "Future years"}[e], horizontal=True, key="sota_ev", label_visibility="collapsed")
    l, r = st.columns([1.5, 1])
    with l:
        st.plotly_chart(ranking_bar("T2", ev, "AUPRC", height=470), config=CFG, width="stretch")
        group_legend("T2", macro_table("T2", ev).index)
    with r:
        card("Takeaways", ul([
            "SeqLab ranks above every published EHR sequence model, under both evaluation schemes.",
            "Tree models (HGB, LightGBM, XGBoost) also beat most published sequence models. "
            "On data of this size, trees are the toughest competitor, which is why HGB stays in the final system.",
            "The frequency prior reaches an AP of only about 0.05: this is the floor.",
        ]))
        st.write("")
        html('<div class="note">Note: the published models are reimplementations for this task, not the authors\' '
             'original code. BEHRT here reads diagnosis codes only and is not pretrained as in the original; RETAIN, '
             'Doctor AI and GRU-D read data aggregated per visit.</div>')


N_SOTA = """This is the ranking of 13 methods by AP.

The green group are the published EHR sequence models: RETAIN, Doctor AI, GRU-D, BEHRT and STraTS. I reimplemented them on exactly our data and splits to compare fairly. SeqLab ranks above all of them, in CV and in future years.

One notable point: tree models such as HGB, LightGBM and XGBoost also beat almost all published sequence models. This matches the Labrador paper from 2024: on lab data, trees are a very hard baseline to beat. That is why I keep HGB in the system.

I also state the limitation clearly: these are reimplementations, not the authors' original code. I do not compare with numbers from the papers, because the data and labels differ."""


# =============================================================== 15. Reading the numbers
def s_read():
    head("Reading the numbers right", "An AP of 0.16 sounds low, but it must be set against a positive rate of about 5%")
    c = curves()["T2"]
    l, r = st.columns([1.25, 1])
    with l:
        ev = "cv"
        per = {x["cls"]: x for x in c[ev]["perclass"] if x["method"] == OUR["T2"]}
        prev = [c[ev]["curves"][k][OUR["T2"]]["prevalence"] for k in EVAL_CODES]
        ap = [per[k]["AUPRC"] for k in EVAL_CODES]
        xs = [f"<b>{k}</b><br>{wrap(CODES_EN[k], 14)}" for k in EVAL_CODES]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=xs, y=prev, name="Positive rate (= random-guess AP)", marker=dict(color="#c9ccd2", cornerradius=4),
                             text=[fmt(v) for v in prev], textposition="outside", cliponaxis=False))
        fig.add_trace(go.Bar(x=xs, y=ap, name="Final system AP", marker=dict(color="#2a78d6", cornerradius=4),
                             text=[fmt(v) for v in ap], textposition="outside", cliponaxis=False))
        fig.update_layout(barmode="group", bargap=0.3)
        fig.update_yaxes(title="AP", range=[0, max(ap) * 1.2])
        fig.update_xaxes(tickangle=0, tickfont=dict(size=11))
        html('<div style="font-weight:700;margin-bottom:-.4rem">AP per code vs random guessing (CV)</div>')
        st.plotly_chart(fig_layout(fig, 360), config=CFG, width="stretch")
        ratio = sum(ap) / sum(prev)
        html(f'<div class="note">On average AP is about <b>{fmt(ratio, 1)} times</b> the random-guessing level. I20 '
             '(angina) is the easiest; I25 is the hardest, and also the code with the fewest cases among the 5.</div>')
    with r:
        fu = {x["cls"]: x for x in c["future"]["perclass"] if x["method"] == OUR["T2"]}["I20"]
        alerts = fu["TP"] + fu["FP"]
        card("Example: I20 in future years", f"""<table class="simple">
<tr><td>Visits that truly added I20</td><td class="num"><b>{fu['positive']}</b></td></tr>
<tr><td>Visits flagged by the system</td><td class="num">{alerts}</td></tr>
<tr><td>Correct flags</td><td class="num">{fu['TP']}</td></tr>
<tr><td>Missed</td><td class="num">{fu['FN']}</td></tr>
<tr><td>Accuracy</td><td class="num">{fmt(fu['Accuracy'])}</td></tr>
<tr><td>Recall</td><td class="num">{fmt(fu['Recall'])}</td></tr>
<tr><td>Precision</td><td class="num">{fmt(fu['Precision'])}</td></tr>
</table><p style="margin-top:.6rem">Accuracy is high because most visits have no new code. In practice the system catches
only about 1 in 8 cases when it may flag only a few visits.</p>""")
    why("<p>I do not present this system as a tool ready for clinical use. What the results show is that, under the same "
        "conditions, reading the sequence and blending with trees ranks at-risk patients better than every approach "
        "tried. The task itself is very hard: we only know whether the next visit recorded a new code, not when the "
        "disease began.</p>", "Plainly")


N_READ = """This slide is here to prevent misreading the numbers.

An AP of 0.16 sounds low, but random guessing is only about 0.05. On average the system is about 3 times that level. I20 is the easiest, I25 the hardest.

On the right is I20 in future years, when the system may flag only as many visits as the positive rate. An Accuracy of 0.91 sounds great, but out of 845 real cases the system caught only 99. That is why I do not use Accuracy to compare.

To be plain: this is not yet a clinical tool. My conclusion is narrower: on the same data and with the same scoring, reading the sequence and blending with trees ranks risk better than every method I tried."""


# =============================================================== 16. T1, T3
def s_others():
    head("Also", "Two side tasks: T1 and T3",
         "Same SeqLab, same data, but so far I have not seen enough value to make them headline results.")
    a, b = st.columns(2)
    d1 = []
    for ev in ("cv", "future"):
        x = deltas("T1", ev, "AUPRC").assign(tau=lambda z: metrics("T1", ev).loc[z.index, "tau"])
        x = x[x.b == "RAD system (published)"].sort_values("tau")
        d1.append([fmt(v * 100, 1, True) for v in x["value"]])
    t3 = {ev: macro_table("T3", ev)["AUPRC"] for ev in ("cv", "future")}
    v, lo, hi = ci("T3", "future", "hgb")
    with a:
        card("T1 · Within the same admission", f"""<p>At 2 / 6 / 12 hours after admission, predict the cardiovascular codes
that will be added during this admission. Adding SeqLab to the RAD system (submitted paper) raises AP in all 6 cells, with
every confidence interval above 0.</p>
<table class="simple" style="margin-top:.5rem"><tr><th>AP difference (points)</th><th>2 h</th><th>6 h</th><th>12 h</th></tr>
<tr><td>CV</td><td>{d1[0][0]}</td><td>{d1[0][1]}</td><td>{d1[0][2]}</td></tr>
<tr><td>Future years</td><td>{d1[1][0]}</td><td>{d1[1][1]}</td><td>{d1[1][2]}</td></tr></table>
<p style="margin-top:.6rem"><b>Caveat:</b> the time a lab result becomes available is currently inferred from the order ID
and has not been verified. This cannot yet be called a real "early warning after 2 hours".</p>""", tag="RAD study")
    with b:
        card("T3 · New codes within 1 year", f"""<p>Like T2 but looking further ahead: new codes within the next 1–365 days.</p>
<table class="simple" style="margin-top:.5rem"><tr><th>macro AP</th><th>CV</th><th>Future years</th></tr>
<tr><td>HGB</td><td>{fmt(t3['cv']['hgb'])}</td><td>{fmt(t3['future']['hgb'])}</td></tr>
<tr><td>SeqLab</td><td>{fmt(t3['cv']['seqlab_t3mix12'])}</td><td>{fmt(t3['future']['seqlab_t3mix12'])}</td></tr>
<tr class="us"><td>SeqLab + HGB</td><td>{fmt(t3['cv'][OUR['T3']])}</td><td>{fmt(t3['future'][OUR['T3']])}</td></tr></table>
<p style="margin-top:.6rem"><b>Caveat:</b> in future years SeqLab alone is clearly worse than HGB; the blend is only
{pts(v)} points above HGB, with a confidence interval [{pts(lo)}; {pts(hi)}] that still contains 0. The early periods have
too little training data.</p>""", tag="One year")
    why("<p>I keep both tasks in the methods but not as main conclusions. T1 needs the data-availability times verified; "
        "T3 needs more data or a better way to learn in periods with few samples.</p>")


N_OTHERS = """Two side tasks, briefly.

T1 is the within-admission task tied to the RAD study. Adding SeqLab raises AP in all 6 cells, with statistical significance. But I have not verified when lab results are actually in the doctor's hands, so I do not yet call it early warning.

T3 is new codes within one year. It helps in CV, but in future years SeqLab alone is clearly worse than HGB, and the blend is only on par with HGB. I think the main cause is that the early periods have too little training data, but that is still a hypothesis.

So for now I treat both as secondary results."""


# =============================================================== 17. Summary
def s_end():
    head("Summary", "What is done, what is still uncertain, and next steps")
    a, b, c = st.columns(3)
    with a:
        card("Done", ul([
            "A corpus of 920 thousand time-stamped events, checked for label leakage.",
            "SeqLab + pretraining: forecasts the next blood draw 29% better than repeating the old value (aim 2).",
            "T2: the SeqLab + HGB system has the highest AP, significantly above HGB in both CV and future years (aim 1).",
            "SeqLab beats every reimplemented published EHR sequence model.",
        ]), "keep")
    with b:
        card("Still uncertain", ul([
            "One hospital, no external validation yet.",
            "The label is a newly recorded code, not the onset date; only returning patients are included.",
            "SeqLab alone is only on par with HGB.",
            "Not yet separated how much of the gain comes from pretraining, from the sequence, or from the number of networks.",
            "Test results were seen across many development rounds; an independent final evaluation is needed.",
        ]), "drop")
    with c:
        card("Next steps", ul([
            "Ablations: pretraining vs random initialisation, 1 vs 21 networks, directly on T2.",
            "Also report results for patients who never had a code before (first-time risk).",
            "Freeze the pipeline and run one final evaluation, then external validation.",
            "Write up: merge into the RAD manuscript or a separate paper for aims 1 + 2.",
        ]), "accent")


N_END = """To sum up.

Done: a leakage-checked sequential data corpus; SeqLab with pretraining, forecasting lab values about 29% better than the simple baseline, which meets aim 2; and on T2, the SeqLab + HGB blend is best, significantly above HGB under both evaluation schemes, which meets aim 1.

Still uncertain: one hospital; the label is a recorded code, not the onset date; SeqLab alone is only on par with HGB; and I have not yet separated where the gain comes from. Also, over many development rounds I looked at test results, so an independent final evaluation is needed.

Next steps are the ablation experiments, one final evaluation, and then writing the paper."""


SLIDES = [  # (section, slide title, render function, speaker notes) — same order as slides.SLIDES
    ("Opening", "Opening", s_intro, N_INTRO),
    ("Data", "Raw data", s_raw, N_RAW),
    ("Data", "Keep or drop", s_select, N_SELECT),
    ("Data", "Data processing", s_process, N_PROCESS),
    ("Data", "Target ICD codes", s_icd, N_ICD),
    ("Task", "Task T2", s_task, N_TASK),
    ("Task", "Why read sequences", s_why_seq, N_WHY_SEQ),
    ("Model", "Architecture overview", s_overview, N_OVERVIEW),
    ("Model", "SeqLab architecture", s_arch, N_ARCH),
    ("Model", "Pretraining", s_pretrain, N_PRETRAIN),
    ("Model", "Final system", s_system, N_SYSTEM),
    ("Results", "Evaluation", s_eval, N_EVAL),
    ("Results", "T2 results", s_result, N_RESULT),
    ("Results", "Versus SOTA", s_sota, N_SOTA),
    ("Results", "Reading the numbers", s_read, N_READ),
    ("Wrap-up", "T1 and T3", s_others, N_OTHERS),
    ("Wrap-up", "Summary", s_end, N_END),
]
