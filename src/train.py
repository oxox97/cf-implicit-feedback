"""
모델 학습
"""

import numpy as np
from scipy.sparse import csr_matrix


def update_user_latent_matrix(
    X: np.ndarray, 
    Y: np.ndarray,
    C: csr_matrix,
    P: csr_matrix,
    lambda_reg: float
) -> np.ndarray:
    """
    User latent matrix 업데이트.
    x_u = (Y^T Y + Y^T (C_u - I) Y + lambda_reg * I)^(-1) Y^T C_u p_u

    Args:
        X: User latent matrix (n_users, k)
        Y: Item latent matrix (n_items, k)
        C: Confidence matrix of shape (n_users, n_items)
            For user u, c_u = C[u] (vector), C_u = diag(c_u)
        P: Preference matrix of shape (n_users, n_items)
        lambda_reg: Regularization parameter

    Returns:
        Updated user latent matrix X of shape (n_users, k)
    """
    n_users, k = X.shape
    n_items = Y.shape[0]
    # 유저에 상관없이 공통 행렬 미리 계산
    YtY = np.matmul(Y.T, Y)  # (k, k)
    I_k = np.identity(k)  # (k, k)
    
    # 유저별로 업데이트
    for u in range(n_users):
        c_u = C[u].toarray().flatten()  # (n_items,)
        p_u = P[u].toarray().flatten()  # (n_items,)
        C_u = np.diag(c_u)  # (n_items, n_items)  # 벡터 브로드캐스팅 방법 확인 필요 c_u[:None]?

        A = YtY + np.matmul(Y.T, np.matmul(C_u - np.identity(n_items), Y)) + lambda_reg * I_k  # identity 말고 vector로 변경하여 overflow 해결 필요

        b = np.matmul(Y.T, np.matmul(C_u, p_u))

        X[u] = np.linalg.solve(A, b)  # LU Decomposition
    
    return X


def update_item_latent_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    C: csr_matrix,
    P: csr_matrix,
    lambda_reg: float
) -> np.ndarray:
    """
    Item latent matrix 업데이트.
    y_i = (X^T X + X^T (C_i - I) X + lambda_reg)^(-1) X^T C_i p_i

    Args:
        X: User latent matrix (n_users, k)
        Y: Item latent matrix (n_items, k)
        C: Confidence matrix of shape (n_users, n_items)
        P: Preference matrix of shape (n_users, n_items)
        lambda_reg: Regulization parameter

    Returns:
        Updated item latent matrix Y of shape (n_items, k)
    """
    n_items, k = Y.shape
    n_users = X.shape[0]
    XtX = np.matmul(X.T, X)  # (k, k)
    I_k = np.identity(k)  # (k, k)
    
    # 아이템별로 업데이트
    for i in range(n_items):
        c_i = C[:, i].toarray().flatten()  # (n_users,)
        p_i = P[:, i].toarray().flatten()  # (n_users,)
        C_i = np.diag(c_i)  # (n_users, n_users)  # 벡터 브로드캐스팅 방법 확인 필요 c_i[:None]?
        
        A = XtX + np.matmul(X.T, np.matmul(C_i - np.identity(n_users), X)) + lambda_reg * I_k
        
        b = np.matmul(X.T, np.matmul(C_i, p_i))
        
        Y[i] = np.linalg.solve(A, b)  # LU Decomposition
    
    return Y


def compute_loss(
    X: np.ndarray,
    Y: np.ndarray,
    C: csr_matrix,
    P: csr_matrix,
    lambda_reg: float,
) -> float:
    """
    ALS 목적 함수 계산
    J = sum_{u,i} C_ui * (P_ui - x_u^T y_i)^2 + lambda_reg (||X||^2 + ||Y||^2)
    """
    pred = np.matmul(X, Y.T)
    diff = P.toarray() - pred
    loss = np.sum(C.toarray() * (diff ** 2))
    loss += lambda_reg * (np.sum(X**2) + np.sum(Y**2))
    return loss


def als_train(
    X: np.ndarray,
    Y: np.ndarray,
    C: csr_matrix,
    P: csr_matrix,
    lambda_reg: float,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ALS 학습.
    """
    for it in range(n_iter):
        X = update_user_latent_matrix(X, Y, C, P, lambda_reg)
        Y = update_item_latent_matrix(X, Y, C, P, lambda_reg)
        loss = compute_loss(X, Y, C, P, lambda_reg)

        if (it+1) % 10 == 0:
            print(f"[Iter {it+1}/{n_iter}] Loss : {loss:.4f}")

    return X, Y


if __name__ == "__main__":
    import data_loader
    import preprocessing

    df = data_loader.parse_netflix_file(max_movie_id=100, max_user_id=10000)
    P, C, user_to_idx, item_to_idx = preprocessing.build_matrices(df=df, alpha=0.4)

    k = 20  # latent factor dimension
    lambda_reg = 0.1
    n_iter = 30

    X = np.random.normal(scale=0.01, size=(len(user_to_idx), k))
    Y = np.random.normal(scale=0.01, size=(len(item_to_idx), k))

    als_train(X, Y, C, P, lambda_reg=lambda_reg, n_iter=n_iter)


