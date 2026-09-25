# 03A FontCLIP Logo Prescreener v1.1

## v1.1 변경사항
- 모든 분석 결과를 **개별 CSV**로 다운로드 가능
- 모든 CSV를 하나의 ZIP으로 일괄 다운로드 가능
- 15 facet 점수와 5차원 프로파일을 별도 CSV로 분리
- pairwise cosine distance를 matrix / long format 두 형식으로 저장
- Aaker facet → FontCLIP prompt 정의 CSV 저장
- shortlist가 생성되지 않아도 분석결과 CSV 다운로드 가능
- Streamlit Cloud Python 3.12 배포를 위해 CPU PyTorch 버전 고정
- FontCLIP 원 코드의 `pkg_resources` 호환을 위해 `setuptools==80.8.0` 고정

## 생성되는 CSV
1. `03A_01_logo_standardization_review.csv` — 이미지 규격과 연구자 A/B/C 판정
2. `03A_02_fontclip_all_analysis_results.csv` — 로고별 전체 분석값 통합본
3. `03A_03_fontclip_15facet_scores.csv` — Aaker 15 facet cosine similarity
4. `03A_04_fontclip_5D_profiles.csv` — Aaker 5차원, 주차원, 다양성 지표, PCA 좌표
5. `03A_05_fontclip_pairwise_distance_matrix.csv` — 브랜드×브랜드 5D cosine distance 행렬
6. `03A_06_fontclip_pairwise_distance_long.csv` — pairwise distance long format
7. `03A_07_fontclip_diversity_shortlist.csv` — maximin maximum-variation shortlist
8. `03A_08_fontclip_prompt_definition.csv` — 사용한 facet 및 고정 prompt
9. `03A_09_fontclip_run_metadata.csv` — 앱 버전, FontCLIP commit/checkpoint, 분석 조건

## 연구용 핵심값
Primary metric은 **raw image-text cosine similarity**입니다. 5차원 점수는 각 Aaker dimension에 포함된 facet cosine similarity의 산술평균입니다.

## 배포 환경
Streamlit Community Cloud에서는 **Python 3.12**를 사용하세요.
