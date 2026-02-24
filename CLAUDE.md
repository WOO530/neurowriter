# NeuroWriter

AI 기반 의학 논문 Introduction 자동 생성 에이전트.
연구 주제를 입력하면 PubMed 논문을 검색·분석하여 학술적 Introduction을 생성한다.

## 프로젝트 목적

- EEG/PSG + 딥러닝 기반 신경과학·정신의학 연구 논문의 Introduction 작성 자동화
- 실제 PubMed 논문만 인용 (hallucination 방지)
- 자체 품질 평가 및 팩트체크로 신뢰도 확보

## 핵심 파이프라인

```
연구 주제 입력
  → 주제 파싱 & 계층적 분석 (topic_parser)
  → 15+ 검색 쿼리 생성 & PubMed Deep Research (deep_researcher)
  → 100-200편 수집 → 30-50편 선별 (citation_scorer)
  → Introduction 생성 (intro_generator, GPT-4o)
  → 8개 기준 루브릭 자체 평가 (self_evaluator)
  → 팩트체크 & 수정 (fact_checker, revision)
```

## 프로젝트 구조

```
neurowriter/
├── app.py                 # Streamlit 메인 UI
├── config.py              # 설정 (API 키, 상수)
├── core/
│   ├── llm_client.py      # LLM 추상화 (OpenAI 구현)
│   ├── topic_parser.py    # 주제 분석 & 쿼리 생성
│   ├── pubmed_client.py   # PubMed API 검색
│   ├── deep_researcher.py # Deep Research 파이프라인
│   ├── citation_scorer.py # 논문 스코어링 & 선별
│   ├── intro_generator.py # Introduction 생성 오케스트레이션
│   ├── self_evaluator.py  # 8기준 루브릭 평가
│   └── fact_checker.py    # PMID 검증 & 내용 일치 확인
├── prompts/               # LLM 프롬프트 모듈
│   ├── topic_parsing.py
│   ├── intro_generation.py
│   ├── self_evaluation.py
│   ├── fact_checking.py
│   └── revision.py
├── utils/
│   ├── cache.py           # SQLite 캐싱
│   └── pubmed_utils.py    # PubMed XML 파싱, Vancouver 포맷
├── .env                   # API 키 (커밋 금지)
└── requirements.txt
```

## 기술 스택

- **Python 3.11+**, Streamlit, OpenAI API (gpt-4o), PubMed E-utilities
- **주요 패턴:** Factory (LLM 클라이언트), DI, JSON-first LLM 출력, rate limiting

## 실행 방법

```bash
pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY 설정
streamlit run app.py
```

## 개발 방향

- Introduction 품질 향상에 집중
- 가져오는 논문의 quality control 강화
- 실제 논문 내용이 Introduction에 정확히 반영되었는지 검증 강화
- Self-evolving 구조 구현 (스스로 개선하는 시스템)

## 작업 시 유의사항

- `.env` 파일은 절대 커밋하지 않는다
- PubMed API rate limit 준수 (0.5초 딜레이)
- LLM 출력은 반드시 JSON 파싱 + 에러 핸들링
- 인용은 Vancouver 스타일, 실제 PMID가 있는 논문만 사용
- 현재 혼자 쓰는 프로젝트이나 추후 타인에게 공개 예정 (Streamlit UI)
