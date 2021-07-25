# Movie_Review 영화리뷰 감정분석 학습(긍정, 부정판단)
## 텍스트 데이터 분석(자연어 처리, KoNLPy)
자연어 처리, 한글을 분석할 떄는 한글 형태소 분석기인 KoNLPy를 사용하여 텍스트를 분석, 처리하여 원하는 활용도로 데이터 처리

### 구조
1. KoNLPy 라이브러리의 Okt 형태소 분석기를 사용하여 데이터 전처리
2. 영화 리뷰를 긍정, 부정으로 2진분류할 수 있도록 학습

### 학습
 - 영화평과 그 영화평에 긍정 혹은 부정으로 나타내져있는 2열(리뷰, 평가)의 데이터를 사용
 - 데이터를 토큰화(문장을 단어 형태로 토큰화)
 - 토큰화된 단어의 사용빈도를 확인(Matplotlib을 사용하여 시각화)
 - 자주사용된 단어 상위 100개를 원핫인코딩
 - 단어에 사용에 따른 긍정, 부정 분류 학습
 - 긍정적인 리뷰에 사용된 단어, 부정적인 리뷰에 사용된 단어를 학습하고 이에 따라 입력된 리뷰를 판별
<img src="https://user-images.githubusercontent.com/87750521/126892084-82aaf776-8baf-4a6b-9de4-c133240f8c4c.png" width="500" height="330">

### 결과
 - 아주 재미 없어요를 입력하였을 때 96%의 정확도로 부정임을 판별
 - 단어의 포함 여부를 바탕으로 학습하였기 때문의 중의적 표현, 비꼬는 듯한 리뷰에서의 정확도는 다소 떨어지는 것으로 보여짐
<img src="https://user-images.githubusercontent.com/87750521/126892154-e167b670-fa8c-4c3e-9134-2b529f29fb39.png" width="500" height="80">

