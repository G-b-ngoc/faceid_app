import streamlit as st
import pandas as pd
from datetime import datetime
import pytz
import os
import io
import cv2
import pickle
import time

import streamlit as st

st.title("FaceID Online")

st.write("Nhận dữ liệu từ web camera")

# --- 1. CẤU HÌNH HỆ THỐNG ---
FILE_LOP = "danh_sach_lop.xlsx"
mui_gio_vn = pytz.timezone('Asia/Ho_Chi_Minh')

st.set_page_config(page_title="Hệ thống Quản lý + FaceID", page_icon="📱", layout="wide")

# --- 2. HÀM AI (NHẬN DIỆN KHUÔN MẶT THEO STT) ---
def xuly_ai_tra_may():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("face_model.yml")
        with open("labels.pickle", "rb") as f:
            label_dict = pickle.load(f)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception as e:
        st.error(f"Lỗi hệ thống AI: {e}. Kiểm tra file model và labels!")
        return None

    cap = cv2.VideoCapture(0) 
    # cap = cv2.VideoCapture("http://192.168.137.117:81/stream")
    found_stt = None
    start_time = time.time()
    
    # Quét trong vòng 30 giây
    while (time.time() - start_time) < 30:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face_gray = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face_gray)

            # Nếu độ tin cậy tốt (Confidence thấp là tốt)
            if confidence < 65: 
                found_stt = label_dict.get(label, None)
                if found_stt:
                    cap.release()
                    cv2.destroyAllWindows()
                    return str(found_stt) # Trả về STT (ví dụ '01')

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Dang quet... (Conf: {int(confidence)})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("He thong FaceID - Dang tra may", frame)
        if cv2.waitKey(1) == 27: break # Nhấn ESC để thoát
        
    cap.release()
    cv2.destroyAllWindows()
    return None

# --- 3. QUẢN LÝ DỮ LIỆU ---
def load_data():
    if os.path.exists(FILE_LOP):
        try:
            df = pd.read_excel(FILE_LOP)
            df['STT'] = df['STT'].astype(str).apply(lambda x: x.split('.')[0].strip().zfill(2))
            for col in ['HoTen', 'TrangThai', 'GioCat', 'GioTra']:
                if col in df.columns:
                    df[col] = df[col].astype(str).replace('nan', '').str.strip()
            return df
        except Exception as e:
            st.error(f"Lỗi đọc Excel: {e}")
            return None
    return None

df = load_data()

# --- 4. GIAO DIỆN CHÍNH ---
st.title("📱 QUẢN LÝ ĐIỆN THOẠI X FACE ID")

if df is not None:
    tab_thu, tab_tra, tab_bc = st.tabs(["📥 THU MÁY", "📤 TRẢ MÁY", "📊 BÁO CÁO"])

    # --- TAB THU MÁY (GIỮ NGUYÊN) ---
    with tab_thu:
        st.subheader("📥 Trạm Thu Máy (Đầu giờ)")
        with st.form(key='form_thu', clear_on_submit=True):
            ma_thu = st.text_input("Nhập STT nộp máy rồi Enter:")
            if st.form_submit_button("Xác nhận Thu"):
                stt = ma_thu.strip().zfill(2)
                if stt in df['STT'].values:
                    idx = df.index[df['STT'] == stt][0]
                    if df.at[idx, 'TrangThai'] == "✅ Đã cất":
                        st.warning(f"⚠️ Bạn {df.at[idx, 'HoTen']} đã nộp máy rồi!")
                    else:
                        df.at[idx, 'TrangThai'] = "✅ Đã cất"
                        df.at[idx, 'GioCat'] = datetime.now(mui_gio_vn).strftime("%H:%M %d/%m")
                        df.to_excel(FILE_LOP, index=False)
                        st.success(f"✅ Đã thu máy của: {df.at[idx, 'HoTen']}")
                        st.rerun()
                else:
                    st.error("❌ Không tìm thấy STT này!")

    # --- TAB TRẢ MÁY (LOGIC AI MỚI THEO STT) ---
    with tab_tra:
        st.subheader("📤 Trạm Trả Máy FaceID")
        col_ai, col_tay = st.columns(2)
        
        with col_ai:
            st.info("Cách 1: Tự động")
            if st.button("🚀 BẬT CAMERA QUÉT MẶT"):
                with st.spinner("Đang mở camera..."):
                    stt_nhan_dien = xuly_ai_tra_may()
                
                if stt_nhan_dien:
                    stt_chuan = stt_nhan_dien.strip().zfill(2)
                    match = df[df['STT'] == stt_chuan]
                    
                    if not match.empty:
                        idx = match.index[0]
                        ten_hs = df.at[idx, 'HoTen']
                        if df.at[idx, 'TrangThai'] == "✅ Đã cất":
                            df.at[idx, 'TrangThai'] = "🏠 Đã trả"
                            df.at[idx, 'GioTra'] = datetime.now(mui_gio_vn).strftime("%H:%M %d/%m")
                            df.to_excel(FILE_LOP, index=False)
                            st.balloons()
                            st.success(f"🎊 Chào {ten_hs}, mời bạn nhận máy!")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(f"🚫 Bạn {ten_hs} chưa nộp máy!")
                    else:
                        st.error(f"❌ Nhận diện được STT {stt_chuan} nhưng không có trong danh sách lớp!")

        with col_tay:
            st.info("Cách 2: Thủ công")
            with st.form(key='form_tra_tay', clear_on_submit=True):
                ma_tra = st.text_input("Nhập STT để trả tay:")
                if st.form_submit_button("Xác nhận Trả tay"):
                    stt = ma_tra.strip().zfill(2)
                    if stt in df['STT'].values:
                        idx = df.index[df['STT'] == stt][0]
                        if df.at[idx, 'TrangThai'] == "✅ Đã cất":
                            df.at[idx, 'TrangThai'] = "🏠 Đã trả"
                            df.at[idx, 'GioTra'] = datetime.now(mui_gio_vn).strftime("%H:%M %d/%m")
                            df.to_excel(FILE_LOP, index=False)
                            st.success(f"🏠 Đã trả máy cho: {df.at[idx, 'HoTen']}")
                            st.rerun()
                        else: st.error("Bạn này chưa nộp máy!")

    # --- TAB BÁO CÁO (GIỮ NGUYÊN) ---
    with tab_bc:
        st.subheader("📊 Trạng thái nộp máy lớp")
        st.dataframe(df, use_container_width=True)
        
        if st.button("🔄 Reset Ngày Mới"):
            df['TrangThai'] = "Chưa nộp"
            df['GioCat'] = ""; df['GioTra'] = ""
            df.to_excel(FILE_LOP, index=False)
            st.rerun()
else:
    st.warning("⚠️ Vui lòng copy file 'danh_sach_lop.xlsx' vào thư mục code!")
