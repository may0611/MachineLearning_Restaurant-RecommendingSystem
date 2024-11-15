import pandas as pd
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Load and preprocess data
data = pd.read_csv("../RecommenderSystem/review_category.csv")  # 데이터 파일을 로드합니다.

# 텍스트 데이터를 하나의 컬럼으로 합칩니다.
data['combined_text'] = data['text_nouns'].apply(eval).astype(str) + data['text_adjectives'].apply(eval).astype(str)

# 사용자별로 벡터 그룹화 (user_id를 기준으로)
user_data = data.groupby('user_id').agg({
    'stars': 'mean',  # 평균 평점
    'business_id': lambda x: list(x),  # 리뷰한 비즈니스 목록
    'combined_text': ' '.join  # 텍스트 병합
}).reset_index()

# Step 2: 사용자 1000명 랜덤 샘플링
sampled_users = random.sample(list(user_data['user_id']), 1000)
sampled_user_data = user_data[user_data['user_id'].isin(sampled_users)].reset_index(drop=True)  # 인덱스 리셋

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
sampled_user_text_vectors = vectorizer.fit_transform(sampled_user_data['combined_text'])

# Step 3: KNN 모델 학습
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(sampled_user_text_vectors)

# Step 4: 추천 시스템 함수 (상위 5개 추천)
def recommend(user_id):
    # 사용자 ID로 인덱스를 찾음
    user_row = sampled_user_data[sampled_user_data['user_id'] == user_id]
    if user_row.empty:
        print(f"User ID {user_id} not found in the sampled data.")
        return []
    
    user_index = user_row.index[0]  # 행 위치를 찾음
    
    # 유사한 사용자 검색
    distances, indices = knn.kneighbors(sampled_user_text_vectors[user_index], n_neighbors=5)
    
    # 추천 목록 생성
    recommended_businesses = set()
    for idx in indices[0]:
        similar_user_businesses = sampled_user_data.iloc[idx]['business_id']
        recommended_businesses.update(similar_user_businesses)
    
    # 현재 사용자가 이미 리뷰한 비즈니스는 제외
    reviewed_businesses = set(sampled_user_data[sampled_user_data['user_id'] == user_id]['business_id'].iloc[0])
    recommendations = list(recommended_businesses - reviewed_businesses)
    
    return recommendations[:5]  # 상위 5개 아이템만 추천

# Step 5: RMSE 계산 함수
def calculate_rmse():
    true_ratings = []
    predicted_ratings = []

    for user_id in sampled_users:
        # 해당 사용자의 실제 리뷰 데이터 가져오기
        user_reviews = data[data['user_id'] == user_id]
        
        for _, row in user_reviews.iterrows():
            business_id = row['business_id']
            true_rating = row['stars']
            
            # 예측 평점 계산
            similar_users = recommend(user_id)
            neighbor_ratings = data[(data['user_id'].isin(similar_users)) & (data['business_id'] == business_id)]['stars']
            
            if not neighbor_ratings.empty:
                predicted_rating = neighbor_ratings.mean()  # 이웃의 평균 평점을 예측 평점으로 사용
            else:
                predicted_rating = data[data['business_id'] == business_id]['stars'].mean()  # 전체 평균 평점 사용
                
            true_ratings.append(true_rating)
            predicted_ratings.append(predicted_rating)

    # RMSE 계산
    rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    print(f"\nTraining RMSE: {rmse:.4f}")

# Step 6: 테스트 RMSE 계산 함수
def calculate_test_rmse(test_size=0.2):
    true_ratings = []
    predicted_ratings = []

    test_data = data[data['user_id'].isin(sampled_users)]
    train_data, test_data = train_test_split(test_data, test_size=test_size, random_state=42)

    for _, row in test_data.iterrows():
        user_id = row['user_id']
        business_id = row['business_id']
        true_rating = row['stars']

        # 예측 평점 계산
        similar_users = recommend(user_id)
        neighbor_ratings = data[(data['user_id'].isin(similar_users)) & (data['business_id'] == business_id)]['stars']

        if not neighbor_ratings.empty:
            predicted_rating = neighbor_ratings.mean()  # 이웃의 평균 평점을 예측 평점으로 사용
        else:
            predicted_rating = data[data['business_id'] == business_id]['stars'].mean()  # 전체 평균 평점 사용

        true_ratings.append(true_rating)
        predicted_ratings.append(predicted_rating)

    # 테스트 RMSE 계산
    test_rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    print(f"\nTest RMSE: {test_rmse:.4f}")

# RMSE 계산
calculate_rmse()
calculate_test_rmse()

# Example: 랜덤 사용자에게 추천
random_user_id = random.choice(sampled_users)
recommended_items = recommend(random_user_id)

# 추천된 아이템을 보기 좋게 출력
print(f"\nRecommended businesses for user {random_user_id}:")

for rank, business_id in enumerate(recommended_items, 1):
    print(f"Rank {rank}: Business ID {business_id}")
