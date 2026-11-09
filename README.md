# SeqCVD · App thuyết trình T2

Bản tách riêng của `phd_heart/publication/seqcvd/thuyet_trinh_T2_20261102/` để deploy lên Streamlit Community Cloud.
Chỉ chứa số liệu tổng hợp (chỉ số, đường cong, số đếm); không có dữ liệu bệnh nhân.

Chạy trên máy: `pip install -r requirements.txt` rồi `streamlit run app.py`.

## Deploy (Streamlit Community Cloud)

1. Vào https://share.streamlit.io, đăng nhập bằng GitHub.
2. **Create app → Deploy a public app from GitHub**:
   - Repository: `ndthong2411/demo_heart`, Branch: `main`, Main file path: `app.py`
   - App URL: chọn tên, ví dụ `seqcvd-t2`
   - Advanced settings → Python version: **3.11**
3. Bấm **Deploy**. Lần đầu cài thư viện mất khoảng 2–4 phút.

Repo private thì app cũng private: vào **Share** của app để mời người xem bằng email.

## Cập nhật

Sửa ở bản gốc trong `phd_heart`, chép các file thay đổi sang đây, `git commit` và `git push`. Streamlit Cloud tự cập nhật.
