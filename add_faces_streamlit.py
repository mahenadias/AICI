import os
import pickle
import subprocess

import cv2
import dlib  # Import dlib untuk deteksi facial landmarks
import numpy as np
import streamlit as st  # TEST


def load_existing_data():
    if os.path.exists('data/faces_data.pkl') and os.path.exists('data/users.pkl'):
        with open('data/faces_data.pkl', 'rb') as f:
            faces_data = pickle.load(f)
        with open('data/users.pkl', 'rb') as f:
            users = pickle.load(f)
    else:
        faces_data = []  
        users = []
    return faces_data, users


def save_data(faces_data, users):
    if not os.path.exists('data'):
        os.makedirs('data')

    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)

    with open('data/users.pkl', 'wb') as f:
        pickle.dump(users, f)


def main():
    st.title("Tambah Wajah Baru ke Dataset")

    # Input nama, NIM, dan prodi pengguna
    name = st.text_input("Masukkan Nama:")
    user_id = st.text_input("Masukkan NIM:")
    major = st.text_input("Masukkan Prodi:")
    add_face_button = st.button("Tambah Wajah")
    train_button = st.button("Mulai Training")

    faces_data, users = load_existing_data()

    if add_face_button and name and user_id and major:
        cap = cv2.VideoCapture(1)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load dlib's face detector and shape predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(cv2.data.haarcascades + 'shape_predictor_68_face_landmarks.dat')

        face_samples = []
        count = 0  
        FRAME_WINDOW = st.image([])

        while count < 100:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal membuka kamera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Deteksi wajah dengan dlib
            faces_dlib = detector(gray)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                resized_face = cv2.resize(face, (64, 64))
                face_samples.append(resized_face)
                count += 1

                # Deteksi landmark wajah menggunakan dlib
                for face in faces_dlib:
                    landmarks = predictor(gray, face)

                    # Landmark hanya digunakan untuk fitur, tidak ditampilkan di kamera
                    landmarks_data = np.array([[p.x, p.y] for p in landmarks.parts()])

                # Gambarkan bounding box wajah (tanpa landmark)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{count}/100", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            FRAME_WINDOW.image(frame, channels="BGR")

            if count >= 100:
                break

        cap.release()
        st.success(f"100 gambar wajah berhasil disimpan untuk {name}.")

        # Tambahkan wajah dan pengguna baru ke data yang sudah ada
        faces_data.append(np.array(face_samples))  # Append new face data
        users.append({'name': name, 'user_id': user_id, 'major': major})  # Append new user info

        save_data(faces_data, users)

    if train_button:
        st.info("Memulai proses training di backend...")
        try:
            process = subprocess.Popen(['python', 'train_cnn_model.py'])
            process.wait()
            if process.returncode == 0:
                st.success("Proses training selesai! Model telah berhasil disimpan.")
            else:
                st.error("Proses training gagal. Periksa log di terminal backend.")
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == '__main__':
    main()
