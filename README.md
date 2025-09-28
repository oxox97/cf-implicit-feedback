# 📖 Collaborative Filtering for Implicit Feedback Datasets
Hu, Koren, Volinsky (2008)

---

## 1. 논문 개요
- **제목**: Collaborative Filtering for Implicit Feedback Datasets  
- **저자**: Yifan Hu, Yehuda Koren, Chris Volinsky  
- **발표**: IEEE International Conference on Data Mining (ICDM), 2008  
- **요약**:
  - Explicit feedback(평점) 대신 Implicit feedback(조회, 클릭, 구매 등)을 추천에 활용하는 방법 제시
  - Confidence weighting 도입 → 행동 빈도에 따라 가중치 부여
  - ALS(Alternating Least Squares)를 이용하여 효율적인 학습

---

## 2. 피드백 특징
- **Explicit feedback**: 사용자가 명시적으로 준 정보 (예: 영화 평점)  
- **Implicit feedback**: 행동 로그 기반 정보 (예: 클릭, 조회, 구매 이력)
  - 부정적 피드백이 명확하지 않음 (안 봤다고 해서 싫어하지 않을 수 있음)
  - 데이터가 훨씬 풍부하지만, 노이즈가 많음  
- -> confidence weight 사용
---

## 3. 모델 설명
### Latent Factor Model
- 사용자와 아이템을 잠재 요인 공간(latent factors)에 매핑
- 예측:  
  $$
  \hat{r}_{ui} = x_u^T y_i
  $$

### Confidence Weighting
- 이 논문의 핵심 아이디어:  
  $$
  C_{ui} = 1 + \alpha \cdot r_{ui}
  $$
  - $r_{ui}$: implicit feedback (예: 클릭 횟수)  
  - $C_{ui}$: 신뢰도(클릭이 많을수록 더 강하게 반영)

### 목적 함수
- 손실 함수 (Weighted squared error):
  $$
  \min_{x_*, y_*} \sum_{u,i} C_{ui} \cdot (p_{ui} - x_u^T y_i)^2 + \lambda ( \|x_u\|^2 + \|y_i\|^2 )
  $$
  - $p_{ui}$: preference (행동이 있으면 1, 없으면 0)

### 학습 방법
- ALS(Alternating Least Squares)로 최적화
- 각 사용자 벡터/아이템 벡터를 번갈아가며 업데이트

---

## 4. 데모
- **데이터셋**: Netflix Prize dataset을 Implicit 데이터로 변환하여 사용 

---

## 6. 내 생각
- 광고 로그는 대부분 **implicit feedback** (노출, 클릭, 전환)  
- 이 논문 아이디어는 광고 추천/광고 타게팅에 직접 적용 가능  
- 예:
  - 클릭이 많을수록 높은 confidence
  - 전환(conversion) 같은 신호는 더 큰 가중치로 반영 가능
- 현대 딥러닝 기반 모델(DIN, SASRec 등)도 결국 implicit 신호를 다루는데, 이 논문이 그 출발점 중 하나라고 볼 수 있음

---

## 7. 참고 자료
- [원문 논문 PDF](https://ieeexplore.ieee.org/document/4781121)  
- [Implicit 라이브러리 (Python)](https://github.com/benfred/implicit)