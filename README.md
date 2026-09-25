# 03A FontCLIP Logo Prescreener v1.0

## 목적
K-코스메틱 영문 로고타입 후보군을 **FontCLIP으로 사전 분석**하여 Aaker(1997)의 15 facets / 5 dimensions에 대한 계산적 의미 프로파일을 만들고, 최종 연구표본이 특정한 로고타입 인상에 편중되지 않도록 **maximum-variation sampling을 보조**하는 Streamlit 앱입니다.

이 앱은 브랜드의 실제 개성을 확정하거나 인간 지각을 대체하기 위한 도구가 아닙니다.

## 현재 연구 흐름에서의 위치

`시장 랭킹 참고 → K-코스메틱 로고 후보 수집 → B/W·1024×1024·실제 로고 약 800px 표준화 → FontCLIP 사전분석 → 프로파일 다양성 확인 → 공식 텍스트·패키지·웹사이트 확보 여부 확인 → 최종 연구대상 확정`

기존 `02 Brand Personality Analyzer`(공식 브랜드 텍스트 분석)와 역할이 다르므로 별도 모듈 `03A`로 분리합니다.

## 중요한 수정사항
업로드된 기존 `app(2).py`는 FontCLIP 이미지 분석기가 아니라 `SentenceTransformer(paraphrase-multilingual-MiniLM-L12-v2)`를 사용한 **텍스트 기반 02 Brand Personality Analyzer v1.8**이었습니다. 따라서 로고 이미지 업로드, FontCLIP 체크포인트 로딩, 이미지-텍스트 cosine similarity 계산이 없었습니다.

본 v1.0은 다음을 새로 구현합니다.

1. ZIP 안의 PNG/JPG/WebP 로고 이미지 일괄 인식
2. 1024×1024, PNG, 비백색 foreground 약 800px 조건 자동 점검
3. 연구자의 A/B/C 로고 적합성 기록
4. 공식 FontCLIP GitHub 소스와 공개 checkpoint 사용
5. Aaker 15 facets를 고정 prompt(`{facet} font`)로 변환
6. 로고 이미지 ↔ 15 facets raw cosine similarity 계산
7. facet 평균으로 Aaker 5차원 프로파일 구성
8. 5차원 프로파일 간 cosine distance 산출
9. deterministic maximin 방식의 maximum-variation shortlist 제공
10. 분석결과, pairwise distance, shortlist, 실행 재현정보 CSV 저장

## FontCLIP 사용 근거
- FontCLIP: Tatsukawa et al. (2024), *FontCLIP: A Semantic Typography Visual-Language Model for Multilingual Font Applications*.
- 공식 저장소: `https://github.com/yukistavailable/FontCLIP`
- FontCLIP은 CLIP을 typography-specific knowledge로 fine-tuning하여 font image와 semantic text attribute를 동일한 latent space에서 비교합니다.
- 공식 font retrieval 구현은 이미지/텍스트 임베딩 간 cosine similarity를 이용합니다.

## Aaker 분석 기준

- Sincerity: Down-to-earth, Honest, Wholesome, Cheerful
- Excitement: Daring, Spirited, Imaginative, Up-to-date
- Competence: Reliable, Intelligent, Successful
- Sophistication: Upper-class, Charming
- Ruggedness: Outdoorsy, Tough

**주의:** Aaker 척도 자체가 FontCLIP용으로 개발된 것이 아니므로 `Aaker facet → FontCLIP prompt` 적용은 본 연구의 탐색적 조작화입니다. 결과는 인간의 브랜드 개성 점수가 아니라 **계산적 의미 유사도**로 해석합니다.

## 주요 분석값

### Primary metric
`raw cosine similarity`

### Dimension score
각 dimension에 속하는 facet cosine similarity의 산술평균.

예:
`Excitement = mean(Daring, Spirited, Imaginative, Up-to-date)`

### Diversity shortlist
5차원 raw profile의 크기 차이에 덜 민감하도록 profile 간 **cosine distance**를 계산하고, 이미 선택된 표본과의 최소거리를 최대화하는 `maximin` 방식으로 서로 다른 사례를 우선 배치합니다.

이 shortlist는 **최종 선정 자동판정이 아닙니다.** 공식 브랜드 텍스트, 패키지, 웹사이트 자료 확보 가능성을 추가 확인한 뒤 연구자가 확정해야 합니다.

## 실행환경
Python 3.11 또는 3.12 권장.

처음 실행 시 앱이:
1. FontCLIP 공식 GitHub `main`의 현재 commit SHA를 조회
2. 그 commit의 소스 archive를 다운로드
3. FontCLIP 공식 `setup_data.py`에 기재된 Google Drive checkpoint를 다운로드
4. 실제 사용한 commit SHA와 checkpoint SHA256을 결과 메타데이터에 기록

합니다.

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
- Python 3.11/3.12 권장
- GitHub 및 Google Drive outbound download가 가능해야 함
- checkpoint는 저장소에 직접 포함하지 않음

## 연구방법 서술 시 권장 표현
“FontCLIP 사전분석을 통해 후보 로고타입의 브랜드 개성을 확정하였다”가 아니라,

> “최종 연구대상 선정에 앞서 후보 영문 로고타입의 시각적·의미적 프로파일 분포가 특정 유형에 편중되는지를 확인하기 위해 FontCLIP 기반 사전분석을 실시하였다. 이 결과는 표본의 다양성을 확보하기 위한 보조기준으로 사용하였으며, 지향 브랜드 개성 또는 인간평가 결과는 표본선정에 사용하지 않았다.”

라고 기술하는 것을 권장합니다.
