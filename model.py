import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(products_path='products.csv', reviews_path='reviews.csv'):
    products_df = pd.read_csv(products_path)
    reviews_df = pd.read_csv(reviews_path)

    products_df['title'] = products_df['title'].fillna('').astype(str) if 'title' in products_df.columns else ''
    products_df['asin'] = products_df['asin'].fillna('').astype(str) if 'asin' in products_df.columns else ''
    products_df['price_value'] = pd.to_numeric(products_df['price_value'], errors='coerce') if 'price_value' in products_df.columns else 0
    products_df['rating_count'] = pd.to_numeric(products_df['rating_count'], errors='coerce') if 'rating_count' in products_df.columns else 0
    products_df['rating_stars'] = products_df['rating_stars'].fillna('').astype(str) if 'rating_stars' in products_df.columns else ''
    products_df['product_description'] = products_df['product_description'].fillna('').astype(str) if 'product_description' in products_df.columns else ''

    reviews_df['productASIN'] = reviews_df['productASIN'].fillna('').astype(str) if 'productASIN' in reviews_df.columns else ''
    reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce') if 'rating' in reviews_df.columns else 0
    reviews_df['reviewText'] = reviews_df['reviewText'].fillna('').astype(str) if 'reviewText' in reviews_df.columns else ''
    reviews_df['reviewTitle'] = reviews_df['reviewTitle'].fillna('').astype(str) if 'reviewTitle' in reviews_df.columns else ''
    reviews_df['sentiment_score'] = pd.to_numeric(reviews_df['sentiment_score'], errors='coerce') if 'sentiment_score' in reviews_df.columns else 0

    similarity_matrix = build_similarity(products_df)
    return products_df, reviews_df, similarity_matrix


def build_similarity(products_df):
    if 'title' not in products_df.columns or products_df.empty:
        return None
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products_df['title'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_recommendations(product_title, products_df, similarity_matrix, num_recommendations=4):
    if similarity_matrix is None or 'title' not in products_df.columns or product_title not in products_df['title'].values:
        return pd.DataFrame()

    idx = products_df[products_df['title'] == product_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    product_indices = [i[0] for i in sim_scores]
    return products_df.iloc[product_indices].reset_index(drop=True)
