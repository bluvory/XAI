# XAI
XAI 설명 가능한 인공지능, 인공지능을 해부하다 / 위키북스

## 01 이야기를 열며
XAI는 기존 인공지능 위에 설명성을 부여하는 기법
1. 기존 머신러닝 모델에 설명 가능한 기능 추가
2. 머신러닝 모델에 HCI(Human Computer Interaction) 기능 추가
3. XAI를 통한 현재 상황이 개선



## 02 실습환경 구축
<pre><code>pip install -r requirements.txt</code></pre>
https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko


## 03 XAI 개발 준비
- 머신러닝: 학습 방식을 먼저 입력하고 기계가 스스로 로직을 만들어가게 제작하는 과정
- 머신러닝은 연산 과정이 많을수록, 절차가 깊어질수록 학습 과정이 복잡해짐 왜냐하면 학습을 위해 처리해야 할 매개변수가 많아지기 때문
- 블랙박스: 머신러닝 모델의 의사 결정 과정을 인간이 직접 이해할 수 없을 때
- XAI: 머신러닝 모델의 블랙박스 성향을 인간이 이해할 수 있는 수준까지 분해하는 기술
- 즉 인공지능의 해석 가능성이란 머신러닝이 내린 결정의 근거를 인간이 이해할 수 있게 변이하는 과정
- XAI vs 시각화
  - XAI는 대리분석, 부분 의존성 플롯, 유사도 분석, 피처 중요도 등의 기법으로 데이터와 모델 설명
  - 즉 XAI의 핵심은 해석 가능성 (왜 해당 모델을 신뢰해야하는지, 모델이 왜 특정 결정을 했는지에 관한 근거 찾기, 어떤 결과가 예상되는지 판단)
  - 시각화와 도해법(Graphical Method)은 모델을 분석한 이후에 추가로 수행하는 



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
### 대리 분석 개론 Surrogate Analysis
인공지능모델이 너무복잡해서분석이 불가능할때유사한 기능을 흉내 내는 인공지능 모델 여러 개를 대리로 만들어서 본래 모델을 분석하는 기법
근사치 모델, 반응 표면 기법(RSM), 에뮬레이터 등의 이름으로 불림

- 장점: 모델 애그노스틱 (model-agnostic technology) 즉 모델에 대한 지식 없이도 학습 가능

- 분석해야 할 모델을 f, 대리 분석 모델은 f를 흉내내는 모델 g라고 할 때 모델 g를 결정하는 조건
1. 모델 f보다 학습하기 쉬움
2. 설명 가능
3. 모델 f를 유사하게 흉내냄

- 모델 g를 학습 시키는 과정 두가지
1. 글로벌 대리 분석: 모델 f를 학습시킬 때처럼 학습 데이터 전부를 모델 g 학습
  - 장점: 유연함, 직관적, 다양한 XAI 기법을 자유롭게 적용 가능, 블랙박스 모델을 이해하고 있지 않아도 measure function을 가지고 모델 f가 어떻게 학습됐는지 설명 가능
  - 단점: 모델 f를 직접 설명하는 게 아니라 간접적으로 설명하기 때문에 g 모델의 정확도와 g 모델의 해석 방향에 결함 존재 가능, measure function의 설명 가능성 판단 기준이 주관적, 데이터 편향 됐을 위험이 있으므로 데이터가 일반적이고 그 크기가 효율적인지 재고 요함
2. 로컬 대리 분석: 데이터 라벨별로 또는 학습 데이터의 일부만 추려서 모델 g 학습 (ex. LIME)


### 잘 알려진 XAI 알고리즘과 알고리즘 각각의 특성을 정리한 표
|알고리즘      |선형성|단조함수 유무|PDP Interaction|목표|
|-------------|------|------------|---------------|---|
|선형 회귀     |있음  |단조함수     |불가능         |회귀|
|로지스틱 회귀 |없음  |단조함수     |불가능          |분류|
|의사 결정 트리|없음  |일부         |가능           |분류, 회귀|
|나이브 베이즈 |없음  |단조함수     |불가능          |분류|
|K-최근접 이웃 |없음  |단조함수 아님|불가능          |분류, 회귀|

### LIME
LIME: Local Interpretable Model-agnostic Explanations
모델이 현재 데이터의 어떤 영역을 집중해서 분석했고 어떤 영역을 분류 근거로 사용했는지 알려주는 XAI 기법

- 장점
1. LIME은 머신러닝 알고리즘에 관계없이 XAI를 적용할 수 있다. 입력 데이터를 변형해서 설명 가능성을 조사하기 때문에 모델의 종류에 구애받지 않는다 딥러닝 기법이나 비싼 그래픽 카드를 사용하지 않아도 적용 가능
2. 매트릭스로 표현 가능한 데이터에 대해 작동함, 서브모듈러를 찾고 그것을 설명하기 때문에 결과가 직관적
3. 다른 XAI 기법과 비교했을 때 매우 가벼움




## 06 필터 시각화
### 딥러닝이란
딥러닝: 입력 계층과 출력 계층 사이에 은닉 계층을 수많은 비선형 방정식으로 조합해 학습하는 머신러닝 알고리즘

딥러닝은 은닉 계층이 문자 그대로 hidden되어 있어서 모델을 이해하기가 어렵다

### 이미지 필터 시각화
Visualizing Image Filters, 이미지 필터 시각화

:학습된 신경망 모델에 이미지가 입력됐을 때 각 은닉 계층마다 인풋 이미지에 어떻게 반응하는지 시각적으로 확인하는 기법

이미지가 신경망 입력층을 통과 ➔ 각 은닉층에 들어옴 ➔ 필터(은닉층에 복수로 존재하는 활성 함수 다발)가 입력 이미지를 통과시키며 활성화 함수 연산 ➔ 계산 결과를 앞으로 전달




## 07 LPR(Layer-wise Relevance Propagation)




## 08 실전분석1: 의사 결정 트리와 XAI

### 신용대출 분석 데이터(loan) 설명

데이터는 loanData.csv에 저장되어 있다. loanData는 19개의 칼럼으로 되어 있으며, 첫 번째 칼럼부터 18번째 칼럼까지는 대출 신청을 위한 사용자 정보가 저장되어 있으며 19번째 칼럼에는 대출 승인 여부가 이진(binary) 형태로 저장되어 있다.

### 칼럼 설명

데이터 칼럼이 의미하는 내용은 다음과 같다.

1. id: 고객 아이디
2. gender: 대출 신청자 성별
3. age: 대출 신청자 나이
4. married: 결혼 유무
5. dependents: 가족 수
6. education: 학력
7. self_employed: 자영업 유무
8. business_type: 국세청 기준 대출 신청인 업종 코드
9. applicant_income: 대출 신청인 수입
10. applicant_work_period: 대출 신청인 근무 기간
11. coapplicant_income: 배우자 수입
12. credit_history: 금융서비스(대출) 이용 횟수
13. credit_amount: 대출중인 금액
14. property_area: 주거지 종류(Urban: 도시, Semiurban: 준도시, Rural: 시골)
15. property_type: 주거지 소유 여부(1: 자가, 2: 월세, 3: 전세, 4: 기타)
16. credit_rate: 신용등급
17. loan_amount: 대출 금액
18. loan_term: 대출 상환 기간
19. loan_status: 대출 승인 여부


### 피처 중요도와 SHAP
- SHAP
  - 전체 대출 신청자들 간의 상대적 피처 중요도 표시, 어떤 유저의 대출 신청자 수입 피처가 대출 여부에 결정적이었다면 섀플리 영향도는 매우 크게 계산
  - 장점: 개별 데이터를 한눈에 요약하듯이 볼 수 있음
  - 단점: 극단치(outlier)에 취약 
- 피처 중요도: 신용 등급의 중요도를 측정할 때 다른 조건들은 그대로 두고 오직 신용 등급만 바꿈
  - 각 피처가 서로 의존적일 때 과소평가된 결과를 출력하며 상대적으로 다른 피처가 과대평가될 수 있음



### 기타

8번 칼럼의 국세청 업종 코드는 이곳(https://www.venturein.or.kr/popup/BusinessCode.do)에서 확인할 수 있다.


## 09 실전분석2: LPR과 XAI

### JAFFE 감정분석 데이터(Emotion)

![JAFFE example](http://www.kasrl.org/KA_004.jpg)

JAFFE 감정분석 데이터 중 놀람(SUP)에 대응하는 이미지 한 장.

### 데이터 설명

JAFFE(The Japanese Femail Facial Expression) 데이터베이스는 일본인 여성(학생들)의 얼굴 사진과 감정을 정량적인 수치로 표시한 데이터베이스다.

### 데이터 다운로드 방법

JAFFE 데이터베이스는 연구 목적으로만 자유롭게 사용될 수 있으며, 각 사용자들은 모두 라이센스에 동의해야 한다. 라이센스에 대한 동의는 [이 링크](http://www.kasrl.org/jaffedb_info.html)에서 하고 다운 받을 수 있다.

위 링크에서 라이센스 사용에 대한 동의를 한 이후 이미지를 다운 받고 "./Ch2.emotion/jaffe"라는 하위폴더에 이미지를 삽입한다.

### 칼럼 설명

이미지에 대응하는 데이터 칼럼이 텍스트파일 형태로 저장되어 있다. 텍스트파일은 "./Ch2.emotion/jaffe/jaffe_labels.txt"에 저장되어 있다.

데이터의 첫 번째와 두 번째 행은 이미지에 대한 설명이다. 이후 행에 대하여 각 칼럼별로 의미하는 바는 다음과 같다.

1. id: 이미지 고유값
2. HAP: 행복
3. SAD: 슬픔
4. SUR: 놀람
5. ANG: 분노
6. DIS: 실망
7. FEA: 두려움
8. PIC: 이미지 이름


### 기타

JAFFE 이미지 라이센스는 Michel J. Lyons 교수에게 있다. 이미지를 다운받은 사용자들의 이미지 사용에 대한 책임은 전적으로 이용자에게 있음을 고지한다.

Michael J. Lyons, Shigeru Akemastu, Miyuki Kamachi, Jiro Gyoba.
Coding Facial Expressions with Gabor Wavelets, 3rd IEEE International Conference on Automatic Face and Gesture Recognition, pp. 200-205 (1998).


## 단원별 코랩 데이터

구글 코랩(Google Colab)은 브라우저에서 파이썬을 작성하고 실행할 수 있는 클라우드 기반 주피터 노트북 개발 환경입니다. 코랩과 인터넷만 있으면 예제 코드를 직접 따라하고 실행할 수 있습니다.

|단원|내용|URL|
|---|:----:|---:|
|04 |의사 결정 트리|http://bit.ly/391EmTS|
|05| 대리 분석|http://bit.ly/2RNP6zv|
|06| 필터 시각화| http://bit.ly/37M96YV|
|07| LRP| http://bit.ly/37SDpwX|
|08| 실전 분석1: 의사 결정 트리와 XAI| http://bit.ly/2vBcYxB 또는 <br> 현재 Github Repo의 `Ch1.Loan` 참고|
|09| 실전 분석2: LRP와 XAI| 현재 Github Repo의 `Ch2.Emotion` 참고|


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
