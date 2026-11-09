"""SeqCVD — app thuyết trình tiến độ (T2). Chạy: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="SeqCVD · Thuyết trình", page_icon="🫀", layout="wide",
                   initial_sidebar_state="collapsed")

from common import html, inject_css  # noqa: E402
import results  # noqa: E402
from slides import SLIDES  # noqa: E402

inject_css()

# Phím ← → (hoặc PageUp / PageDown) để chuyển trang chiếu.
# Trên Streamlit Cloud app nằm trong một khung của trang bọc ngoài, nên khi trang bọc giữ focus thì phím không tới
# document của app: gắn thêm bộ nghe vào mọi khung cha cùng origin, còn nút luôn tìm trong document của app.
# Khung nhỏ này bị dựng lại khi đổi trang và bộ nghe của khung cũ chết theo, nên mỗi lần chạy phải thay bộ nghe cũ
# bằng bộ nghe mới (không dùng cờ "đã gắn").
KEYS = """<!doctype html><html><body><script>
const app = window.parent.document;
function onKey(e) {
  const tag = (e.target.tagName || '').toLowerCase();
  if (['input', 'textarea', 'select'].includes(tag) || e.target.isContentEditable) return;
  let label = null;
  if (e.key === 'ArrowRight' || e.key === 'PageDown') label = 'Tiếp →';
  if (e.key === 'ArrowLeft' || e.key === 'PageUp') label = '← Trước';
  if (!label) return;
  const btn = Array.from(app.querySelectorAll('button')).find(b => b.innerText.trim() === label);
  if (btn && !btn.disabled) { e.preventDefault(); btn.click(); }
}
let w = window.parent;
while (true) {
  try {
    const d = w.document;
    if (typeof d._seqcvdKeys === 'function') d.removeEventListener('keydown', d._seqcvdKeys);
    d._seqcvdKeys = onKey;
    d.addEventListener('keydown', onKey);
  } catch (err) { break; }
  if (w === w.parent) break;
  w = w.parent;
}
</script></body></html>"""


def go_to(i):
    st.session_state.slide = max(0, min(len(SLIDES) - 1, i))
    st.session_state.jump = st.session_state.slide
    st.query_params["s"] = str(st.session_state.slide + 1)


def label(j):
    sec, title, _, _ = SLIDES[j]
    return f"{j + 1}. {title}" if sec == title else f"{j + 1}. {sec} · {title}"


def toolbar(i):
    """Thanh chọn trang: ô thả xuống + bảng tất cả trang theo phần."""
    st.session_state.jump = i
    a, b, _ = st.columns([2.2, 1, 2.8])
    a.selectbox("Đi tới trang", range(len(SLIDES)), key="jump", format_func=label, label_visibility="collapsed",
                on_change=lambda: go_to(st.session_state.jump))
    with b.popover("▦ Tất cả trang", width="stretch"):
        sections = list(dict.fromkeys(s[0] for s in SLIDES))
        cols = st.columns(len(sections))
        for col, sec in zip(cols, sections):
            col.markdown(f"**{sec}**")
            for j, (s, title, _, _) in enumerate(SLIDES):
                if s == sec:
                    col.button(f"{j + 1}. {title}", key=f"grid{j}", on_click=go_to, args=(j,), width="stretch",
                               type="primary" if j == i else "secondary")


def presentation():
    if "slide" not in st.session_state:
        try:
            st.session_state.slide = max(0, min(len(SLIDES) - 1, int(st.query_params.get("s", 1)) - 1))
        except ValueError:
            st.session_state.slide = 0
    i = st.session_state.slide
    _, title, render, notes = SLIDES[i]

    with st.sidebar:
        st.markdown("### Mục lục")
        for j in range(len(SLIDES)):
            st.button(label(j), key=f"toc{j}", on_click=go_to, args=(j,), width="stretch",
                      type="primary" if j == i else "secondary")

    toolbar(i)
    html(f'<div class="progress-wrap"><div class="progress-bar" style="width:{(i + 1) / len(SLIDES) * 100:.1f}%">'
         f'</div></div>')
    render()

    st.write("")
    with st.expander("Ghi chú để nói", expanded=False):
        st.markdown(notes)

    a, m, b = st.columns([1, 4, 1])
    # Tính theo trang hiện tại lúc bấm (không theo i của lần vẽ trước) để bấm dồn không bị lùi/đứng sai trang.
    a.button("← Trước", on_click=lambda: go_to(st.session_state.slide - 1), disabled=i == 0, width="stretch")
    m.markdown(f'<div style="text-align:center;color:#7a7f8a;padding-top:.45rem">{i + 1} / {len(SLIDES)} · {title} '
               f'· dùng phím ← → để chuyển trang</div>', unsafe_allow_html=True)
    b.button("Tiếp →", on_click=lambda: go_to(st.session_state.slide + 1), disabled=i == len(SLIDES) - 1, width="stretch",
             type="primary")
    st.iframe(KEYS, height=1)


pg = st.navigation([st.Page(presentation, title="Thuyết trình", icon="🎤", default=True),
                    st.Page(results.page, title="Kết quả & SOTA", icon="📊", url_path="ket-qua")],
                   position="top")
pg.run()
