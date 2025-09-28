"""
코드 실행
"""

import logging
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import parse_netflix_file
from preprocessing import preprocessing
from train import als_train
from recommendation import get_recommendations
from evaluation import precision_at_k

from fastapi import FastAPI


# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s"
# )

# if __name__ == "__main__":
#     logging.info("코드 실행 시작")
#     df = parse_netflix_file(max_movie_id=100, max_user_id=10000)

#     train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
#     P_train, C_train, user_to_idx, item_to_idx = preprocessing(train_df)
#     P_test, C_test, _, _ = preprocessing(test_df)  # train에 있는 사용자 대상 평가 (? coldestart는 안 되나?)

#     k = 20  # latent factor dimension
#     lambda_reg = 0.1

#     X = np.random.normal(scale=0.01, size=(len(user_to_idx), k))
#     Y = np.random.normal(scale=0.01, size=(len(item_to_idx), k))

#     X, Y = als_train(X, Y, C_train, P_train, lambda_reg, n_iter=10)

#     user_id = 0
#     pred_items = get_recommendations(X, Y, user_id=user_id, k=10)
#     print(pred_items)


#     true_items = P_train[user_id].nonzero()[1]
#     print(true_items)

#     print(precision_at_k(pred_items, true_items))
    
#     logging.info("코드 실행 완료")

app = FastAPI()

model = None
P_train = None

@app.post("/train")
def train():

    df = parse_netflix_file(max_movie_id=100, max_user_id=10000)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    P_train, C_train, user_to_idx, item_to_idx = preprocessing(train_df)

    k = 20
    X = np.random.normal(scale=0.01, size=(len(user_to_idx), k))
    Y = np.random.normal(scale=0.01, size=(len(item_to_idx), k))
    print(1)
    X, Y = als_train(X, Y, C_train, P_train, lambda_reg=0.1, n_iter=100)
    print(2)

    print(3)

    return {"status": "success", "message": "학습 완료"}


@app.get("/recommend/{user_id}/{k}")  # async?
def  show_recommendations(user_id: int, k: int):
    recommendations = get_recommendations(
        model["X"],
        model["Y"],
        user_id,
        k,
        P_train=P_train  # 그냥 P?
    )

    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "k": k
    }
