## 2월 2째 주 주간보고서

### 2월 10일 (월)

* 한영 번역 데모 제작하신 분께 메일로 문의한 결과 다음과 같이 답변 해주심.

    1) 코드 : Fairseq , Tensor2Tensor 보는 것 추천
    2) 데이터 단위 : Subword Tokenization - BPE, SentencePiece
    3) 전처리 : Parallel Corpus Filtering 모듈(박찬준), 글자수 80 글자 미만 이런 제한


### 2월 11일 (화)

* 전처리 공부 팀 / 코드 수정 팀 나누었다.
* [Pytorch seq2seq + attention translation tutorial](https://tutorials.pytorch.kr/beginner/torchtext_translation_tutorial.html) 수정 (문장 입력 파트) 
  + 띄어쓰기 단위로 split, 문장의 단어 개수를 제한하여 인코더, 디코더에 사용 
  + AI Hub 데이터셋 일부 돌려보았으나 성능은 낮은 편. (특히 앞 부분 번역은 잘하다가 뒤에 잘 못하는 경우 있었음)
 
* tatoeba 데이터셋 추가 다운로드 ([tatoeba](https://www.manythings.org/anki/)) -> 약 3300 문장, 대체로 짧은 구어체.
  * tatoeba 데이터셋 간단 전처리하여 AI HUB 데이터셋과 결합
  
* 데이터셋 EDA 진행

  1. 중복 여부 파악, 제거
    + 중복 제거 전 : 1,606,026 문장
    + 중복 제거 후 : 1,603,529 문장 (2497개 중복 제거) 
       * AI Hub 데이터셋, tatoeba 데이터셋 합치고 중복 제거한 파일을 'datalist.csv'로 저장.
       <br>
  2. 문장 별 단어 길이 파악
    + 한글 문장 중 가장 많은 단어 개수 가진 문장 : 89개, 영어 문장 중 가장 많은 단어 개수 가진 문장 : 993개의 단어 가지고 있었음.
    + boxplot 그려본 결과 다음과 같음. (좌 : 한글 문장에서의 단어 개수 , 우 : 영어 문장에서의 단어 개수)
    <img src="https://postfiles.pstatic.net/MjAyMDAyMTJfOTQg/MDAxNTgxNDQxOTkyMjQ3.NEh0tc23WgWDioZCOyfqoFaHHieialwnZTD-AV7iTfQg.M9tCUlNLFkQ7A9ws6jmRDdbJM7OKu2SlZeRIKknGDOwg.PNG.wazoskee/image.png?type=w773" width="300" height="200">
    <img src="https://postfiles.pstatic.net/MjAyMDAyMTJfNTEg/MDAxNTgxNDQxODg3MzU0.o2uv_ryt81C1aL-4CISPuPgbBy9fjl8NjR8AUZY_JsMg.0AkUWJEKcUATLwuvlIqILzt9jZDrJVOv5-ZHbQWqaR0g.PNG.wazoskee/image.png?type=w773" width="300" height="200">
    
    + 아웃라이어 처리 고민 필요.
    
* subword tokenization 공부하기 좋아보이는 [링크](https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46)
  + BPE, SentencePiece 내용 있음.

#### 논의 사항

1. 새로운 데이터셋 사용 여부 ? 대화체. 사용하기로 함

2. 여러 문장이 같이 묶여 있는 경우, : 와 ; 로 연결된 긴 문장의 경우 처리 방법?
  ==> 코드로 구분하기에 예외사항이 많기에 단어 / 글자 수로 제한을 두어 전처리 할 듯.
