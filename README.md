# 03A FontCLIP Logo Prescreener v1.2

## 역할
이 앱은 **영문 로고타입의 FontCLIP 사전선별(prescreening)만 수행**합니다.
공식 텍스트 수집, 패키지/웹사이트 분석, 일치도/일관성 계산, 인간평가는 포함하지 않습니다.

## 절차
1. 표준화 로고 ZIP 업로드
2. 1024×1024 / PNG / 문자영역 약 800px 조건 확인
3. 연구자 A/B/C 적격성 검토
4. FontCLIP 실행
5. FontCLIP 이미지 임베딩 cosine distance 기반 다양성 확인
6. **Aaker 15 facets에 대해 positive / negative prompt 및 contrast 계산**
7. **Aaker 15 facets contrast를 5차원으로 요약하여 참고 프로파일 제시**
8. maximin 방식 shortlist 생성
9. 전체 결과 CSV 다운로드

## 산출물
- 03A_01_logo_standardization_review.csv
- 03A_02_fontclip_prescreen_results.csv
- 03A_03_fontclip_image_embeddings.csv
- 03A_04_fontclip_pairwise_distance_matrix.csv
- 03A_05_fontclip_pairwise_distance_long.csv
- 03A_06_fontclip_diversity_shortlist.csv
- 03A_07_aaker_15facet_scores_wide.csv
- 03A_08_aaker_15facet_scores_long.csv
- 03A_09_aaker_5D_reference.csv
- 03A_10_prompt_definition.csv
- 03A_11_run_metadata.csv

## 배포
- Streamlit Community Cloud
- Python 3.12 권장
- `app.py`를 main file로 지정

## 방법상 주의
- shortlist의 1차 기준은 **FontCLIP 이미지 임베딩 간 cosine distance**입니다.
- Aaker 15 facets 및 5차원 값은 **참고 프로파일**이며, 최종 연구대상 확정값이 아닙니다.
- positive / negative prompt는 서체 의미를 상대적으로 보기 위한 보조 장치입니다.
- 최종 연구대상은 이 앱에서 확정하지 않습니다.
