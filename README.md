# XAI
XAI 설명 가능한 인공지능, 인공지능을 해부하다 / 위키북스

## 01 이야기를 열며
XAI는 기존 인공지능 위에 설명성을 부여하는 기법
1. 기존 머신러닝 모델에 설명 가능한 기능 추가
2. 머신러닝 모델에 HCI(Human Computer Interaction) 기능 추가
3. XAI를 통한 현재 상황이 개선

## 02 실습환경 구축
<pre><code>pip install -r requirements.txt</code></pre>


## 03 XAI 개발 준비

## 04 의사 결정 트리
### PDP
### XGBoost
- 장점
1. 훌륭한 그래디언트 부스팅 라이브러리, 벙렬 처리를 사용하기 때문에 학습과 분류가 빠름
2. 유연성이 좋음, 평가함수를 포함해 다양한 커스텀 최적화 옵션 제공
3. greedy algorithm을 사용해 자동으로 forest의 가지를 침. 따라서 과적합이 잘 일어나지 않음
4. 다른 알고리즘과 연계 활용성이 좋음

- The Curse of Dimensionality

데이터의 차원이 증가할수록 해당 공간의 크기가 기하급수적으로 증가하기 때문에 공간 속 데이터 밀도가 희박해지는 상황

- 기본 원리

Y= α×M(X) + β×G(X) + γ×H(X) + error
greedy algorithm을 통해 classifier M, G, H 발견
weight parameter는 분산 처리를 사용해 각 classifier의 최적 반영 가중치

- CART 알고리즘

분류 트리 분석과 회귀 트리 분석을 동시에 사용해서 트리 만듦


- 파라미터
1. general parameter: 도구 모양 결정
2. booster parameter: 트리마다 가지칠 때 적용하는 옵션
3. learning task parameter: 최적화 성능 결정

- 피쳐중요도

트리의 깊이에 따라 모델의 의사결정 방식과 정확도가 함께 변하기 때문에 중요하다


- 부분 의존성 플롯 그리기

피처의 수치 변화에 따라 모델에 기여하는 정도가 어떻게 달라지는지 확인 가능
궁금한 피처가 모델에 긍정적인/부정적인 영향을 미치는지 파악하게 도움
특정 피처에 대해 여유분을 함께 표시함으로써 피처간 독립을 보장하지 못하는 환경에서 어느 정도 모델에 오차가 있을 수 있는지 확인 가능


## 05 대리 분석

## 06 필터 시각화


## 07 LPR(Layer-wise Relevance Propagation)


## 08 실전분석1: 의사 결정 트리와 XAI


## 09 실전분석2: LPR과 XAI


## 10 이야기를 닫으며


## 11 참고자료
### Candlestick chart
- 캔들스틱 하나는 일간, 월간, 연간 집계된 통계 누적치 될 수 있다
- 스틱 안에는 최댓값, 최솟값, 평균, 중간값 등 다양한 통계 누적 가능
- 상위25%=upper shadow, 하위25%=lower shadow
### confusion matrix
|        |예측 음성|예측 양성|
|:------:|:------:|:------:|
|실제 양성|   TP   |   FN   |
|실제 음성|   FP   |   TN   |
- 정확도 (Accuracy) = (TP+TN)/ALL
- 에러율 (Error) = (FN+FP)/ALL
- 정밀성 (Precision) = TP/(TP+FP)
- 민감도 (Sensitivity, Recall) = TP/((TP+FN)
- 특이성 (Specificity) = TN/(TN+FP)
- 낙제율 (Fallout) = TP/(FP+TN)
- F1-점수 (F1-score) = 2×(1/(1/recall+1/precision))
### Normalization
### Regularization
### Standardization
