# Mengimpor library
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Memuat data
print("Memuat data...")

# Memuat dataset
try:
    ratings = pd.read_csv('book-crossings/BX-Book-Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    books = pd.read_csv('book-crossings/BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv('book-crossings/BX-Users.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
except:
    # Alternatif jika parameter on_bad_lines tidak tersedia
    ratings = pd.read_csv('book-crossings/BX-Book-Ratings.csv', sep=';', encoding='latin-1', error_bad_lines=False)
    books = pd.read_csv('book-crossings/BX-Books.csv', sep=';', encoding='latin-1', error_bad_lines=False)
    users = pd.read_csv('book-crossings/BX-Users.csv', sep=';', encoding='latin-1', error_bad_lines=False)

# Membersihkan nama kolom
ratings.columns = ratings.columns.str.strip()
books.columns = books.columns.str.strip()
users.columns = users.columns.str.strip()

print(f"Bentuk asli ratings: {ratings.shape}")
print(f"Bentuk asli books: {books.shape}")

# Prapemrosesan data
print("\nMelakukan prapemrosesan data...")

# Menyaring pengguna dengan setidaknya 200 rating
jumlah_rating_per_pengguna = ratings['User-ID'].value_counts()
ratings = ratings[ratings['User-ID'].isin(jumlah_rating_per_pengguna[jumlah_rating_per_pengguna >= 200].index)]

# Menyaring buku dengan setidaknya 100 rating
jumlah_rating_per_buku = ratings['ISBN'].value_counts()
ratings = ratings[ratings['ISBN'].isin(jumlah_rating_per_buku[jumlah_rating_per_buku >= 100].index)]

print(f"Bentuk ratings setelah penyaringan: {ratings.shape}")

# Menggabungkan buku dengan rating
buku_dengan_rating = pd.merge(ratings, books[['ISBN', 'Book-Title']], on='ISBN', how='left')

# Menghapus baris dimana Judul Buku adalah NaN
buku_dengan_rating = buku_dengan_rating.dropna(subset=['Book-Title'])

# Menghapus duplikat
buku_dengan_rating = buku_dengan_rating.drop_duplicates(['User-ID', 'Book-Title'])

# Membuat tabel pivot
pivot_rating_buku = buku_dengan_rating.pivot(
    index='Book-Title', 
    columns='User-ID', 
    values='Book-Rating'
).fillna(0)

print(f"Bentuk tabel pivot: {pivot_rating_buku.shape}")
print(f"Jumlah buku unik: {len(pivot_rating_buku)}")

# Mengkonversi ke matriks sparse
matriks_rating_buku = csr_matrix(pivot_rating_buku.values)

# Membangun model KNN
print("\nMembangun model KNN...")
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
model_knn.fit(matriks_rating_buku)

# Fungsi untuk mendapatkan rekomendasi
def dapatkan_rekomendasi(judul_buku=""):
    try:
        # Memeriksa apakah buku ada dalam dataset
        if judul_buku not in pivot_rating_buku.index:
            # Mencari judul yang mirip
            buku_yang_cocok = [idx for idx in pivot_rating_buku.index if judul_buku.lower() in idx.lower()]
            if not buku_yang_cocok:
                return [judul_buku, []]
            judul_buku = buku_yang_cocok[0]
            print(f"Menggunakan judul terdekat: {judul_buku}")
        
        indeks_buku = list(pivot_rating_buku.index).index(judul_buku)
        
        # Mendapatkan tetangga terdekat
        jarak, indeks = model_knn.kneighbors(
            pivot_rating_buku.iloc[indeks_buku, :].values.reshape(1, -1)
        )
        
        # Menyiapkan rekomendasi
        rekomendasi = []
        for i in range(1, len(jarak.flatten())):  # Mulai dari 1 untuk mengecualikan buku itu sendiri
            buku_rekomendasi = pivot_rating_buku.index[indeks.flatten()[i]]
            nilai_jarak = jarak.flatten()[i]
            rekomendasi.append([buku_rekomendasi, nilai_jarak])
        
        # Mengurutkan berdasarkan jarak (terdekat pertama) dan mengambil 5 teratas
        rekomendasi = sorted(rekomendasi, key=lambda x: x[1])[:5]
        
        return [judul_buku, rekomendasi]
    
    except Exception as e:
        print(f"Error: {e}")
        return [judul_buku, []]

# Pengujian dengan buku yang diminta
print("\n" + "="*50)
print("Pengujian dengan buku yang diminta:")
print("="*50)
hasil_uji = dapatkan_rekomendasi("The Queen of the Damned (Vampire Chronicles (Paperback))")
print("\nHasil:")
print(hasil_uji)

# Memeriksa apakah hasil sesuai dengan format yang diharapkan
if hasil_uji[1]:
    print(f"\nDitemukan {len(hasil_uji[1])} rekomendasi")
    for i, (buku, jarak) in enumerate(hasil_uji[1], 1):
        print(f"{i}. {buku[:50]}... - Jarak: {jarak:.4f}")

# Pengujian dengan beberapa buku lain
print("\n" + "="*50)
print("Pengujian tambahan:")
print("="*50)

buku_uji = [
    "Where the Heart Is",
    "The Da Vinci Code",
    "The Fellowship of the Ring (The Lord of the Rings, Part 1)"
]

for buku in buku_uji:
    print(f"\nMenguji: '{buku}'")
    hasil = dapatkan_rekomendasi(buku)
    if hasil[1]:
        print(f"Ditemukan {len(hasil[1])} rekomendasi")
        for i, (buku_rek, jarak) in enumerate(hasil[1][:3], 1):  # Menampilkan 3 pertama
            print(f"  {i}. {buku_rek[:40]}... - {jarak:.4f}")
    else:
        print("Tidak ada rekomendasi ditemukan")

# Fungsi bantuan untuk mencari judul buku
def cari_judul_buku(query, hasil_maks=10):
    kecocokan = [judul for judul in pivot_rating_buku.index if query.lower() in judul.lower()]
    return kecocokan[:hasil_maks]

# Contoh pencarian
print("\n" + "="*50)
print("Contoh pencarian:")
print("="*50)
print("\nBuku yang mengandung 'Harry Potter':")
for buku in cari_judul_buku("Harry Potter", 5):
    print(f"  - {buku}")

print("\nBuku yang mengandung 'Vampire':")
for buku in cari_judul_buku("Vampire", 5):
    print(f"  - {buku}")

print("\nSelesai!")