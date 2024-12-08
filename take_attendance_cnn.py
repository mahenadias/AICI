import os
import pickle
import time
from datetime import datetime

import cv2
import dlib  # Untuk mendeteksi facial landmarks
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from PIL import Image  # Untuk menampilkan gambar di Streamlit


# Fungsi untuk memuat model dan data pengguna
def load_model_and_data():
    model = load_model('data/face_recognition_cnn.h5')
    with open('data/users.pkl', 'rb') as f:
        users = pickle.load(f)
    return model, users

# Fungsi untuk mencatat kehadiran ke dalam file CSV
def record_attendance(name, user_id, major):
    file_path = 'data/attendance.csv'
    
    # Jika file tidak ada atau file kosong, buat file baru dengan header yang benar
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        df = pd.DataFrame(columns=["Nama", "NIM", "Prodi", "Waktu Kehadiran"])
        df.to_csv(file_path, index=False)
    
    # Tambahkan data kehadiran baru
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{
        "Nama": name,
        "NIM": user_id,
        "Prodi": major,
        "Waktu Kehadiran": current_time
    }])

    df = pd.read_csv(file_path)
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(file_path, index=False)

# Fungsi untuk memprediksi wajah
def predict_face(face, model, users, threshold=0.6):  
    face = cv2.resize(face, (64, 64))  # Sesuaikan ukuran wajah dengan model input
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    face = face.astype('float32') / 255.0  # Normalisasi
    face = np.expand_dims(face, axis=-1)  # Tambahkan dimensi channel (grayscale)
    face = np.expand_dims(face, axis=0)  # Tambahkan dimensi batch

    predictions = model.predict(face)
    max_pred = np.max(predictions)

    # Periksa apakah prediksi lebih besar atau sama dengan threshold
    if max_pred >= threshold:
        predicted_class = np.argmax(predictions)  # Ambil index prediksi tertinggi
        predicted_name = users[predicted_class]['name']
        user_id = users[predicted_class]['user_id']
        major = users[predicted_class]['major']
        return predicted_name, user_id, major, max_pred
    else:
        return "Tidak Dikenali", None, None, max_pred

# Fungsi untuk mendeteksi pergerakan wajah
def detect_movement(prev_frame, current_frame, threshold=1000):
    if prev_frame is None:
        return False, current_frame
    diff = cv2.absdiff(prev_frame, current_frame)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    
    _, diff_thresh = cv2.threshold(diff_blur, 25, 255, cv2.THRESH_BINARY)
    
    movement_amount = np.sum(diff_thresh)
    return movement_amount > threshold, current_frame

# Fungsi untuk membuat folder jika belum ada
def ensure_user_directory(user_id):
    user_dir = os.path.join("data", "Bukti Kehadiran", user_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    return user_dir

# Fungsi untuk mengambil kehadiran dengan menggunakan kamera
def take_attendance():
    st.title("Sistem Absensi Pengenalan Wajah")
    st.write("Aktifkan kamera untuk melakukan absensi.")

    # Opsi threshold yang bisa diatur oleh pengguna
    threshold = st.slider("Atur Threshold Pengakuan Wajah", min_value=0.1, max_value=1.0, value=0.6, step=0.05)

    model, users = load_model_and_data()

    # Inisialisasi kamera
    cap = cv2.VideoCapture(1)
    frame_placeholder = st.empty()  # Tempatkan untuk video feed
    comment_placeholder = st.empty()  # Tempatkan untuk komentar

    # Load dlib untuk mendeteksi facial landmarks
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(cv2.data.haarcascades + 'shape_predictor_68_face_landmarks.dat')

    # Variabel untuk mencatat kehadiran dan deteksi durasi pengenalan
    last_recorded_time = 0
    recognition_start_time = None
    recognized_name = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Gagal membuka kamera")
            break

        current_time = time.time()

        # Proses wajah dengan deteksi
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Inisialisasi default untuk variabel `comment`
        comment = "Tidak ada wajah yang terdeteksi."

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]

                # Gunakan fungsi prediksi wajah dengan threshold yang ditentukan pengguna
                name, user_id, major, confidence = predict_face(face, model, users, threshold=threshold)

                # Deteksi landmark wajah menggunakan dlib
                faces_dlib = detector(gray)
                for face_dlib in faces_dlib:
                    landmarks = predictor(gray, face_dlib)
                    landmarks_data = np.array([[p.x, p.y] for p in landmarks.parts()])

                if name != "Tidak Dikenali":
                    if recognized_name == name:
                        # Jika nama sama, cek durasi pengenalan
                        if recognition_start_time is None:
                            recognition_start_time = current_time
                        elif current_time - recognition_start_time >= 3:
                            # Jika wajah dikenali selama lebih dari 3 detik
                            if current_time - last_recorded_time >= 5:
                                # Pastikan folder pengguna ada
                                user_dir = ensure_user_directory(user_id)

                                # Simpan bukti gambar dengan nama file berbasis identitas dan waktu
                                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                evidence_filename = os.path.join(user_dir, f"{name}_{timestamp}.jpg")
                                cv2.imwrite(evidence_filename, frame)

                                # Rekam kehadiran
                                record_attendance(name, user_id, major)
                                comment = f"Wajah dikenali dan kehadiran tercatat untuk {name}."
                                last_recorded_time = current_time  # Update waktu terakhir pencatatan

                            # Reset waktu pengenalan agar tidak terus menerus mencatat
                            recognition_start_time = None
                    else:
                        # Nama berubah, reset waktu pengenalan
                        recognized_name = name
                        recognition_start_time = current_time

                    # Tampilkan kotak wajah dan identitas secara realtime
                    cv2.putText(frame, f"Nama: {name}", (x, y-60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"NIM: {user_id}", (x, y-40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"Prodi: {major}", (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    # Cetak akurasi (confidence) ke terminal VSCode
                    print(f"Nama: {name}, Confidence: {confidence:.2f}")

                else:
                    recognized_name = None
                    recognition_start_time = None
                    comment = "Wajah tidak dikenali."

        # Tampilkan komentar yang sinkron dengan pencatatan kehadiran
        comment_placeholder.write(comment)

        # Konversi frame ke format RGB untuk Streamlit dan tampilkan di localhost
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        frame_placeholder.image(img_pil, use_column_width=True)

        # Tambahkan jeda untuk mengurangi beban memori
        time.sleep(0.1)

    cap.release()

# Fungsi utama untuk Streamlit
def main():
    take_attendance()

if __name__ == "__main__":
    main()