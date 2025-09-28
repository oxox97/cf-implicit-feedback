"""
데이터 전처리
"""

import pandas as pd
from scipy.sparse import csr_matrix

def build_matrices(
    df: pd.DataFrame,
    alpha: float,
) -> tuple[csr_matrix, csr_matrix, dict[int, int], dict[int, int]]:
    """
    사용자-아이템 상호작용 데이터에서 P, C 행렬 생성.
    
    Args:
        df: user_id, item_id, watch_count 칼럼을 포함한 데이터프레임
        alpha: confidence scaling factor
    
    Returns:
        P: 선호도 행렬 (0/1)
        C: 신뢰도 행렬 (1 + alpha * watch_count)
        user_to_idx: 사용자 -> 인덱스 매핑 딕셔너리
        item_to_idx: 아이템 -> 인덱스 매핑 딕셔너리
    """

    # preference, confidence 칼럼 추가
    df["preference"] = 1
    df["confidence"] = 1 + alpha * df["watch_count"]

    # 유저, 아이템 매핑
    user_to_idx = {int(u): i for i, u in enumerate(df["user_id"].unique())}
    item_to_idx = {int(i): j for j, i in enumerate(df["item_id"].unique())}

    df["user_idx"] = df["user_id"].map(user_to_idx)
    df["item_idx"] = df["item_id"].map(item_to_idx)

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    # 행렬 생성
    # P: preference matrix
    P = csr_matrix(
        (df["preference"], (df["user_idx"], df["item_idx"])),
        shape=(n_users, n_items)
    )

    # C: confidence matrix
    C = csr_matrix(
        (df["confidence"], (df["user_idx"], df["item_idx"])),
        shape=(n_users, n_items)
    )

    return P, C, user_to_idx, item_to_idx


if __name__ == "__main__":
    import data_loader
    df = data_loader.parse_netflix_file(max_movie_id=100, max_user_id=10000)
    P, C, user_to_idx, item_to_idx = build_matrices(df=df, alpha=0.4)
    print(P.shape, P.nnz)


