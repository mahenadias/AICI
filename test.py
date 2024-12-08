from pymongo import MongoClient

# Koneksi ke MongoDB
url = "mongodb+srv://ningimasaza:hsWpcl865Vwfe9H9@cluster1.csmpu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
client = MongoClient(
    url
)  # Ganti dengan URL MongoDB Anda jika menggunakan server berbeda

# db = client["SPK"]  # Nama database
# collection_face = db["presensi"]  # Collection untuk data face recognition
# collection_activity = db["tracking"]  # Collection untuk data aktivitas

# # # 2. Mengambil Data (Query Data)
# # print("\nMencari data berdasarkan nama:")
# # data_ditemukan = collection_face.find_one({"Nama": "Mahendra Adiastoro"})
# # print(data_ditemukan)

# # 3. Mengambil Semua Data
# print("\nSemua data dalam collection:")
# for doc in collection_face.find():
#     print(doc)

# Send a ping to confirm a successful connection
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)