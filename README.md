# NeuroWriter

뇌파(EEG/PSG) + 딥러닝 기반 신경과/정신과 연구 주제를 입력하면, PubMed 실제 논문을 인용하며 영어 Introduction을 자동 생성하는 Streamlit 대화형 웹앱.

## 기술 스택

- Python 3.11+
- Streamlit (대화형 채팅 UI)
- OpenAI API (GPT-4o)
- PubMed E-utilities API + PMC API (논문 검색/메타데이터)
- SQLite (논문 메타데이터 캐싱)

## 설치 및 실행

### 1. 프로젝트 클론 및 의존성 설치

```bash
cd neurowriter
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env.example` 파일을 참고하여 `.env` 파일을 생성하고 OpenAI API 키를 입력하세요.

```bash
cp .env.example .env
# .env 파일을 열고 OPENAI_API_KEY를 입력하세요
```

### 3. Streamlit 앱 실행

```bash
streamlit run app.py
```

브라우저가 자동으로 열리지 않으면 `http://localhost:8501`로 접속하세요.

## 사용 방법

1. 연구 주제를 한 줄로 입력 (예: "뇌파 딥러닝 분석 기반 주요우울장애의 항우울제 치료 반응성 예측 연구")
2. 시스템이 주제를 분석하고 PubMed에서 관련 논문을 수집
3. 핵심 논문을 자동으로 선별하고 Introduction을 생성
4. 생성된 Introduction 및 Reference List 확인
5. 필요시 추가 수정 요청

## 프로젝트 구조

```
neurowriter/
├── app.py                    # Streamlit 메인 앱
├── config.py                 # API keys, 설정
├── requirements.txt
├── .env                      # 환경변수 (OPENAI_API_KEY)
│
├── core/
│   ├── __init__.py
│   ├── topic_parser.py       # 주제 분석
│   ├── pubmed_client.py      # PubMed 검색 + 메타데이터 수집
│   ├── citation_scorer.py    # 논문 중요도 scoring
│   ├── intro_generator.py    # Introduction 생성 orchestration
│   ├── fact_checker.py       # 팩트체크 파이프라인
│   └── llm_client.py         # LLM 추상화 레이어
│
├── prompts/
│   ├── __init__.py
│   ├── topic_parsing.py      # 주제 파싱 프롬프트
│   ├── intro_generation.py   # Intro 생성 프롬프트
│   ├── fact_checking.py      # 팩트체크 프롬프트
│   └── revision.py           # 수정 요청 프롬프트
│
└── utils/
    ├── __init__.py
    ├── pubmed_utils.py       # PubMed XML 파싱
    └── cache.py              # SQLite 캐싱
```

## 주요 기능

### 1. 지능형 주제 분석
- 사용자 입력을 자동으로 분석하여 질환, 데이터 유형, 방법론, 예측 대상 등 추출

### 2. PubMed 논문 수집
- 5가지 카테고리로 체계적 논문 검색
- 각 카테고리별 최대 20편 수집
- 메타데이터 캐싱으로 중복 검색 방지

### 3. 지능형 논문 선별
- 저널 impact, citation count, recency 등을 종합적으로 평가
- 각 카테고리별 최고 점수 논문 자동 선택

### 4. AI 기반 Introduction 생성
- GPT-4o를 활용한 자연스러운 영어 작성
- 실제 PubMed 논문 인용 기반 생성
- 4-6문단, 800-1200단어 규모

### 5. 팩트체크
- PMID 실존 여부 확인
- 인용 내용과 원 논문 대조
- 수치 검증 및 대표성 확인

## API 키 설정

OpenAI API 키는 https://platform.openai.com/api-keys에서 발급받을 수 있습니다.

## 주의사항

- PubMed API는 초당 3개 이하의 요청 제한이 있습니다
- API key 없이 사용 시 초당 1개 이하로 제한됩니다
- 생성된 Introduction은 반드시 팩트체크를 거쳐야 합니다

## 라이센스

MIT License

## 문의

문제 또는 기능 제안은 GitHub Issues를 통해 보고해주세요.
