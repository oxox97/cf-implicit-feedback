"""
모델 평가 지표
- Precision
- Recall
- MAP
- NDCG
"""

import numpy as np


def precision_at_k(
    pred_items: list[int],
    true_items: list[int],
    k: int = 10
) -> float:
    """
    추천된 상위 k개 아이템 중 실제 선호하는 아이템의 비율 측정.

    Args:
        pred_items : 추천된 아이템 리스트
        true_items: 실제로 사용자가 선호한 아이템 집합
        k: 상위 추천 개수

    Returns:
        Precision@K (0.0 ~ 1.0)
    """
    pred_topk = pred_items[:k]
    hit_count = len(set(pred_topk) & set(true_items))
    return hit_count / k


def recall_at_k(
    pred_items: list[int],
    true_items: list[int],
    k: int = 10
) -> float:
    """
    실제 선호 아이템 중 상위 K개 추천에 포함된 비율 측정.
    """
    pred_topk = pred_items[:k]
    hit_count = len(set(pred_topk) & set(true_items))
    return hit_count / len(true_items)


def average_precision_at_k(
    pred_items: list[int],
    true_items: list[int],
    k: int = 10
) -> float:
    """
    정답 아이템을 만날 때마다 precision@idx 기록, 평균.
    MAP는 여러 유저에 대한 평균
    """
    pred_topk = pred_items[:k]
    score = 0.0
    hit_count = 0

    for idx, item in enumerate(pred_topk, start=1):
        if item in set(true_items):
            hit_count += 1
            score += precision_at_k(pred_topk, true_items, idx)
    
    return score / min(len(set(true_items)), k)


def ndcg_at_k(
    pred_items: list[int],
    true_items: list[int],
    k: int = 10
) -> float:
    """
    NDCG@K : Normalized Discounted Cumulative Gain.
    추천된 아이템의 순위별 적합도를 로그 스케일로 감쇠하여 순위 품질 평가.
    """
    pred_topk = pred_items[:k]

    # DCG 계산
    dcg = 0.0
    for idx, item in enumerate(pred_topk, start=1):
        if item in set(true_items):
            dcg += 1.0 / np.log2(idx + 1)
    
    # IDCG 계산
    max_hits = min(len(true_items), k)
    idcg = sum(1 / np.log2(i + 1) for i in range(1, max_hits + 1))

    return dcg / idcg


if __name__ == "__main__":
    pred_items = [2, 4, 11]
    true_items = [2, 4, 6, 8, 10]
    print(precision_at_k(pred_items, true_items))
    print(recall_at_k(pred_items, true_items))
    print(average_precision_at_k(pred_items, true_items))
    print(ndcg_at_k(pred_items, true_items))