"""
하이퍼 파라미터 탐색
"""

import numpy as np
from train import als_train
from evaluation import evaluate_model
from preprocessing import build_matrices
from data_loader import parse_netflix_file


def grid_search(P, C, param_grid: dict, n_iter: int = 10, top_k: int = 10):
    """
    여러 파라미터 조합을 탐색하여 best parameter 찾기.

    Args:
        P: preference matrix
        C: confidence matrix
        param_grid: {"rank": [...], "lambda": [...], "alpha": [...]}
        n_iter: ALS 반복 횟수
        top_k: 추천 평가 cut-off

    Returns:
        best_params: dict
        best_score: float
    """
    best_score = -1
    best_params = None

    for k in param_grid["rank"]:
        for lam in param_grid["lambda"]:
            for alpha in param_grid["alpha"]:
                # 행렬 생성
                df = parse_netflix_file(max_movie_id=50, max_user_id=500)
                P_new, C_new, _, _ = build_matrices(df, alpha=alpha)

                # 초기화
                X = np.random.normal(scale=0.01, size=(P_new.shape[0], k))
                Y = np.random.normal(scale=0.01, size=(P_new.shape[1], k))

                # 학습
                X, Y = als_train(X, Y, C_new, P_new, lambda_reg=lam, n_iter=n_iter)

                # 평가
                metrics = evaluate_model(X, Y, P_new, P_new, k=top_k)  # TODO: split 적용
                score = (metrics["precision"] + metrics["recall"]) / 2

                print(f"rank={k}, lambda={lam}, alpha={alpha} → {metrics}")

                if score > best_score:
                    best_score = score
                    best_params = {"rank": k, "lambda": lam, "alpha": alpha}

    return best_params, best_score


if __name__ == "__main__":
    # 랜덤 서치 확인
    param_grid = {
        "rank": [10, 20],
        "lambda": [0.01, 0.1],
        "alpha": [0.1, 0.4]
    }

    best_params, best_score = grid_search(None, None, param_grid, n_iter=5, top_k=10)
    print("Best params:", best_params)
    print("Best score:", best_score)
