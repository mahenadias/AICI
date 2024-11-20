import os
import pickle
import subprocess

import cv2
import numpy as np
import streamlit as st


def load_existing_data():
    # Check if faces_data.pkl and users.pkl exist and load them
    if os.path.exists('data/faces_data.pkl') and os.path.exists('data/users.pkl'):
        with open('data/faces_data.pkl', 'rb') as f:
            faces_data = pickle.load(f)
        with open('data/users.pkl', 'rb') as f:
            users = pickle.load(f)
    else:
        faces_data = []  # Initialize empty if no data found
        users = []
    return faces_data, users


def save_data(faces_data, users):
    # Save face data
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

    faces_data, users = load_existing_data()  # Load existing face data and users

    if add_face_button and name and user_id and major:
        # Buka kamera
        cap = cv2.VideoCapture(1)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        face_samples = []
        count = 0  # Counter untuk sampel wajah
        FRAME_WINDOW = st.image([])  # Placeholder Streamlit

        while count < 100:  # Ambil 100 gambar wajah
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal membuka kamera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                resized_face = cv2.resize(face, (64, 64))  # Resize gambar ke 64x64
                face_samples.append(resized_face)
                count += 1

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

        save_data(faces_data, users)  # Simpan data

    if train_button:
        st.info("Memulai proses training di backend...")
        try:
            # Jalankan program train_cnn_model.py secara terpisah
            process = subprocess.Popen(['python', 'train_cnn_model.py'])
            process.wait()  # Tunggu proses selesai
            if process.returncode == 0:
                st.success("Proses training selesai! Model telah berhasil disimpan.")
            else:
                st.error("Proses training gagal. Periksa log di terminal backend.")
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == '__main__':
    main()
