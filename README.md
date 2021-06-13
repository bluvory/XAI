# XAI
XAI 설명 가능한 인공지능, 인공지능을 해부하다 / 위키북스

## 01 이야기를 열며
XAI는 기존 인공지능 위에 설명성을 부여하는 기법
1. 기존 머신러닝 모델에 설명 가능한 기능 추가
2. 머신러닝 모델에 HCI(Human Computer Interaction) 기능 추가
3. XAI를 통한 현재 상황이 개선
## 02 실습환경 구축
<pre><code>pip install -r requirements.txt<\code><\pre>
## 03 XAI 개발 준비
## 04 의사 결정 트리
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
