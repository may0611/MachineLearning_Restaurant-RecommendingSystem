import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise import accuracy
import random
import numpy as np

# 데이터셋 로드
data_path = "merged_data2.csv"
full_data = pd.read_csv(data_path)  # CSV 파일을 pandas DataFrame으로 읽기

# 10000개의 샘플을 랜덤으로 추출 (train dataset)
# 정확도를 올리기 위해 샘플 개수를 1만에서 3만으로 증가시킴
# sampled_data = full_data.sample(n=30000, random_state=0)  # random_state=0를 통해 일관된 테스트를 거쳤다는 것을 보장
sampled_data = full_data.sample(n=30000)
# Reader 클래스는 데이터의 평점 범위를 설정하는데 사용됨
# 여기서는 평점 범위가 1에서 5로 설정됨
reader = Reader(rating_scale=(1, 5))

# 데이터셋을 Surprise 라이브러리에서 사용할 수 있는 형식으로 변환
# 'user_id', 'business_id', 'stars' 열을 사용하여 데이터셋을 생성
data = Dataset.load_from_df(sampled_data[['user_id', 'business_id', 'stars']], reader)

# 유저 기반 협업 필터링 모델(KNN)
sim_options = {
    'name': 'cosine',
    'user_based': True,  # 유저 기반 협업 필터링
    'k': 40,  # k 값 조정, 정확도를 올리기 위해
    'min_support': 5  # 최소 지원 아이템 수
}

# KNNBasic 알고리즘을 사용하여 모델을 생성
algo = KNNBasic(sim_options=sim_options)

# 모델 학습
algo.fit(data.build_full_trainset())  # trainset만 사용하여 학습

# test dataset 로드
test_data_path = "review.csv"  # test 데이터 파일 경로

# test_data 로드 (user_id, business_id, stars, name 열 포함)
test_data = pd.read_csv(test_data_path, usecols=['user_id', 'stars', 'name'])

# 별점이 'like'와 같은 문자로 되어 있는 잘못된 행을 제거
test_data = test_data.drop(7601)  # 'like' 문자가 있는 행 삭제

# test_data의 'stars' 컬럼을 float로 변환 (문자열을 숫자형으로 변환)
test_data['stars'] = pd.to_numeric(test_data['stars'], errors='coerce')

# test_data의 NaN 값 처리
test_data = test_data.dropna(subset=['stars'])  # 'stars' 컬럼이 NaN인 행을 삭제

# test_data 열 이름 확인
print("test_data의 열 이름 확인:", test_data.columns)

# 'test_data'는 이미 user_id, business_id, stars 열이 있기 때문에 그대로 사용
test_data = test_data[['user_id', 'name', 'stars']]

# Surprise 라이브러리에서 사용할 수 있도록 데이터 형식 변환
test_data_surprise = Dataset.load_from_df(test_data[['user_id', 'name', 'stars']], reader)

# 테스트 데이터셋을 튜플 형태로 변환 (Surprise에서 요구하는 형태)
testset = [(row['user_id'], row['name'], row['stars']) for idx, row in test_data.iterrows()]

# 모델 평가 (테스트 데이터 사용)
predictions_test = algo.test(testset)

# RMSE 계산 (Test RMSE)
rmse_test = accuracy.rmse(predictions_test)
print(f"Test RMSE: {rmse_test}")  # RMSE 값 출력

# 1. 테스트 데이터에서 랜덤으로 사용자 선택
random_user = random.choice(test_data['user_id'].unique())
print(f"추천을 위한 랜덤 사용자 ID: {random_user}")

# 2. 해당 사용자가 평점을 매긴 아이템 목록을 추출
user_ratings = test_data[test_data['user_id'] == random_user]
rated_items = user_ratings['name'].unique()

# 3. 사용자가 평점을 매기지 않은 아이템 목록을 추출
# unrated_items = test_data[~test_data['name'].isin(rated_items)]['name'].unique()

# 3-1. 사용자가 평점을 매긴 아이템을 포함한 모든 아이템 목록을 추출해서 대신 사용
unrated_items = test_data['name'].unique()

# 4. 사용자가 평점 매지 않은 아이템에 대해 예측 평점 계산
predicted_ratings = [(item, algo.predict(random_user, item).est) for item in unrated_items]

# 5. 예측 평점이 높은 순으로 정렬하여 상위 k개의 추천 아이템 추출
k = 5
recommended_items = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:k]

# 6. 추천된 아이템 출력
print("추천 아이템:")
for item, rating in recommended_items:
    print(f"아이템 ID: {item}, 예측 평점: {rating}")

# 7. 랜덤한 한 유저에 대한 rank 평가
print('\nRank 평가')

# precision@k
true_positive = sum((item[0] in user_ratings.index for item in recommended_items))
precision = true_positive / k
print(f'precision@{k} of user {random_user}: {precision}')

# recall@k
recall = true_positive / len(user_ratings)
print(f'recall@{k} of user {random_user}: {recall}')

# 8. 모든 유저에 대한 rank 평가

# Mean Average Precision@k
test_df = test_data.drop_duplicates(subset=['user_id'])  # make user ids unique
test_user_ids = test_df['user_id']
average_precision_sum = 0

for test_user_id in test_user_ids:
    user_ratings = test_data[test_data['user_id'] == test_user_id]
    unshaped_rated_items = user_ratings['name'].unique()
    rated_items = []
    for item in unshaped_rated_items:
        rated_items += item

    all_items = test_data['name'].unique()
    predicted_ratings = [(item, algo.predict(random_user, item).est) for item in all_items]

    recommended_items = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:k]

    # calculate precision in range 1 to k
    precision_sum = 0
    for k_i in range(1, k + 1):
        # calculate Average Precision
        # if item is rated by user, relevance is 1 and otherwise 0
        # 0 make Average Precision to 0 so, calculate only when relevance is 1
        if recommended_items[k_i - 1][0] in rated_items:
            true_positive_of_i = sum((item in rated_items for item in recommended_items[:k_i]))
            precision_of_i = true_positive_of_i / k_i
            precision_sum += precision_of_i

    average_precision = precision_sum / len(rated_items)
    average_precision_sum += average_precision

mean_average_precision = average_precision_sum / len(test_user_ids)
print(f'mean average precision@{k}: {mean_average_precision}')
