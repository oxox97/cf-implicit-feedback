"""
추천 결과 반환
"""

import numpy as np


def get_recommendations(
    X: np.ndarray,
    Y: np.ndarray,
    user_id: int,
    k: int = 10
) -> list[int]:
    """
    사용자별 추천 아이템 리스트
    # 이미 본 것 제외?
    """
    scores = np.matmul(X[user_id], Y.T)
    recommendations = np.argsort(scores)[::-1][:k].tolist()
    return recommendations


