"""
데이터 로드
"""


import kagglehub
import pandas as pd
import os
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def parse_netflix_file(max_movie_id: int = None, max_user_id: int = None) -> pd.DataFrame:
    """
    넷플릭스 시청 데이터로 대체하여 사용.
    샘플링하여 가져오기.

    Args:
        max_movie_id: 영화 ID 샘플링 최대값
        max_user_id: 사용자 ID 샘플링 최대값
    
    Returns:
        pd.DataFrame: 넷플릭스 시청 데이터
            - user_id: 사용자 ID
            - item_id: 영화 ID
            - watch_count: 시청 횟수 (rating(explicit) -> watch_count(implicit)으로 변경하여 사용)
    """
    path = kagglehub.dataset_download("netflix-inc/netflix-prize-data")
    file_path = os.path.join(path, "combined_data_1.txt")

    data = []
    current_movie = None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.endswith(":"):
                current_movie = int(line[:-1])
                if max_movie_id is not None and current_movie > max_movie_id:  # movieId sampling (sorted)
                    break
            else:
                current_user, rating, date = line.split(",")
                if max_user_id is not None and int(current_user) > max_user_id:  # userId sampling
                    continue
                data.append([int(current_user), current_movie, int(rating)])
    return pd.DataFrame(data, columns=["user_id", "item_id", "watch_count"])


if __name__ == "__main__":

    logging.info("데이터 로드 시작")
    df = parse_netflix_file(max_movie_id=100, max_user_id=10000)

    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"\n{df.head()}")
    logging.info(f"\n{df['watch_count'].describe()}")
    logging.info("데이터 로드 완료")