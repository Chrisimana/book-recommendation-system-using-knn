# Import libraries
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")

# Load datasets
try:
    ratings = pd.read_csv('book-crossings/BX-Book-Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    books = pd.read_csv('book-crossings/BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv('book-crossings/BX-Users.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
except:
    # Fallback if on_bad_lines parameter doesn't exist
    ratings = pd.read_csv('book-crossings/BX-Book-Ratings.csv', sep=';', encoding='latin-1', error_bad_lines=False)
    books = pd.read_csv('book-crossings/BX-Books.csv', sep=';', encoding='latin-1', error_bad_lines=False)
    users = pd.read_csv('book-crossings/BX-Users.csv', sep=';', encoding='latin-1', error_bad_lines=False)

# Clean column names
ratings.columns = ratings.columns.str.strip()
books.columns = books.columns.str.strip()
users.columns = users.columns.str.strip()

print(f"Original ratings shape: {ratings.shape}")
print(f"Original books shape: {books.shape}")

# Data preprocessing
print("\nPreprocessing data...")

# Filter users with at least 200 ratings
user_rating_counts = ratings['User-ID'].value_counts()
ratings = ratings[ratings['User-ID'].isin(user_rating_counts[user_rating_counts >= 200].index)]

# Filter books with at least 100 ratings
book_rating_counts = ratings['ISBN'].value_counts()
ratings = ratings[ratings['ISBN'].isin(book_rating_counts[book_rating_counts >= 100].index)]

print(f"Ratings shape after filtering: {ratings.shape}")

# Merge books with ratings
books_with_ratings = pd.merge(ratings, books[['ISBN', 'Book-Title']], on='ISBN', how='left')

# Remove rows where Book-Title is NaN
books_with_ratings = books_with_ratings.dropna(subset=['Book-Title'])

# Remove duplicates
books_with_ratings = books_with_ratings.drop_duplicates(['User-ID', 'Book-Title'])

# Create pivot table
book_ratings_pivot = books_with_ratings.pivot(
    index='Book-Title', 
    columns='User-ID', 
    values='Book-Rating'
).fillna(0)

print(f"Pivot table shape: {book_ratings_pivot.shape}")
print(f"Number of unique books: {len(book_ratings_pivot)}")

# Convert to sparse matrix
book_ratings_matrix = csr_matrix(book_ratings_pivot.values)

# Build KNN model
print("\nBuilding KNN model...")
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
model_knn.fit(book_ratings_matrix)

# Function to get recommendations
def get_recommends(book_title=""):
    try:
        # Check if book exists in our dataset
        if book_title not in book_ratings_pivot.index:
            # Try to find similar titles
            matching_books = [idx for idx in book_ratings_pivot.index if book_title.lower() in idx.lower()]
            if not matching_books:
                return [book_title, []]
            book_title = matching_books[0]
            print(f"Using closest match: {book_title}")
        
        book_idx = list(book_ratings_pivot.index).index(book_title)
        
        # Get nearest neighbors
        distances, indices = model_knn.kneighbors(
            book_ratings_pivot.iloc[book_idx, :].values.reshape(1, -1)
        )
        
        # Prepare recommendations
        recommendations = []
        for i in range(1, len(distances.flatten())):  # Start from 1 to exclude self
            rec_book = book_ratings_pivot.index[indices.flatten()[i]]
            distance = distances.flatten()[i]
            recommendations.append([rec_book, distance])
        
        # Sort by distance (closest first) and take top 5
        recommendations = sorted(recommendations, key=lambda x: x[1])[:5]
        
        return [book_title, recommendations]
    
    except Exception as e:
        print(f"Error: {e}")
        return [book_title, []]

# Test with the required book
print("\n" + "="*50)
print("Testing with required book:")
print("="*50)
test_result = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
print("\nResult:")
print(test_result)

# Check if result matches expected format
if test_result[1]:
    print(f"\nFound {len(test_result[1])} recommendations")
    for i, (book, dist) in enumerate(test_result[1], 1):
        print(f"{i}. {book[:50]}... - Distance: {dist:.4f}")

# Test with some other books
print("\n" + "="*50)
print("Additional tests:")
print("="*50)

test_books = [
    "Where the Heart Is",
    "The Da Vinci Code",
    "The Fellowship of the Ring (The Lord of the Rings, Part 1)"
]

for test_book in test_books:
    print(f"\nTesting: '{test_book}'")
    result = get_recommends(test_book)
    if result[1]:
        print(f"Found {len(result[1])} recommendations")
        for i, (book, dist) in enumerate(result[1][:3], 1):  # Show first 3
            print(f"  {i}. {book[:40]}... - {dist:.4f}")
    else:
        print("No recommendations found")

# Helper function to search books
def search_book_titles(query, max_results=10):
    """Search for books containing the query string"""
    matches = [title for title in book_ratings_pivot.index if query.lower() in title.lower()]
    return matches[:max_results]

# Example search
print("\n" + "="*50)
print("Search examples:")
print("="*50)
print("\nBooks containing 'Harry Potter':")
for book in search_book_titles("Harry Potter", 5):
    print(f"  - {book}")

print("\nBooks containing 'Vampire':")
for book in search_book_titles("Vampire", 5):
    print(f"  - {book}")

print("\nDone!")