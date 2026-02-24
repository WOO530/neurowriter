"""NeuroWriter - Streamlit Web Application (Interactive Self-Evolving Pipeline)"""
import streamlit as st
import logging
from typing import Optional
from core.pipeline_orchestrator import PipelineOrchestrator, MAX_EVOLUTION_ITERATIONS
from core.fact_checker import FactChecker
from core.llm_client import get_llm_client
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL)

# Configure page
st.set_page_config(
    page_title="NeuroWriter",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.stChatMessage {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------

def initialize_session():
    """Initialize session state variables"""
    defaults = {
        "history": [],
        "current_topic": None,
        "generation_result": None,
        "fact_check_result": None,
        "chat_messages": [],
        "current_intro": None,
        "current_references": None,
        "collected_papers": None,
        "parsed_topic": None,
        # Pipeline state machine
        "pipeline_state": "IDLE",
        "pipeline_iteration": 0,
        "topic_analysis": None,
        "search_queries": [],
        "paper_pool": [],
        "landscape": {},
        "reference_pool": [],
        "writing_strategy": {},
        "introduction_text": "",
        "evaluation_result": {},
        "iteration_history": [],
        # Orchestrator recreation
        "api_key_stored": "",
        "model_stored": "",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_pipeline():
    """Reset pipeline state to IDLE"""
    keys = [
        "pipeline_state", "pipeline_iteration", "topic_analysis",
        "search_queries", "paper_pool", "landscape", "reference_pool",
        "writing_strategy", "introduction_text", "evaluation_result",
        "iteration_history", "generation_result", "current_topic",
        "fact_check_result",
    ]
    defaults = {
        "pipeline_state": "IDLE",
        "pipeline_iteration": 0,
        "topic_analysis": None,
        "search_queries": [],
        "paper_pool": [],
        "landscape": {},
        "reference_pool": [],
        "writing_strategy": {},
        "introduction_text": "",
        "evaluation_result": {},
        "iteration_history": [],
        "generation_result": None,
        "current_topic": None,
        "fact_check_result": None,
    }
    for k in keys:
        st.session_state[k] = defaults.get(k)


def get_orchestrator() -> Optional[PipelineOrchestrator]:
    """Create PipelineOrchestrator from stored credentials"""
    api_key = st.session_state.get("api_key_stored", "")
    model = st.session_state.get("model_stored", "gpt-4o")
    if not api_key or not api_key.startswith("sk-"):
        return None
    try:
        return PipelineOrchestrator(api_key=api_key, model=model)
    except Exception as e:
        logger.error(f"Failed to create orchestrator: {e}")
        return None


# ------------------------------------------------------------------
# Header & sidebar
# ------------------------------------------------------------------

def display_header():
    """Display application header"""
    st.markdown("# 🧠 NeuroWriter")
    st.markdown(
        "**EEG/Deep Learning 의학논문 Introduction Generator**  \n"
        "뇌파와 딥러닝 기반 신경과/정신과 연구 주제를 입력하면, "
        "PubMed 논문을 인용하며 영어 Introduction을 자동 생성합니다."
    )


def setup_sidebar():
    """Setup sidebar configuration"""
    with st.sidebar:
        st.markdown("## ⚙️ 설정")

        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="https://platform.openai.com/api-keys에서 발급받으세요"
        )

        # Model selector
        model = st.selectbox(
            "LLM Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )

        # Store for orchestrator recreation
        st.session_state.api_key_stored = api_key
        st.session_state.model_stored = model

        # Reference style
        st.markdown("### 참고문헌 스타일")
        reference_style = st.selectbox(
            "Citation Style",
            ["APA", "Vancouver", "AMA"],
            index=0,
            help="APA: Author (Year). Title. Journal.\nVancouver: Number. Author. Title. Journal. Year.\nAMA: Author. Title. Journal. Year;Vol(Issue):Pages."
        )

        # Cache management
        st.markdown("### 캐시 관리")
        from utils.cache import PubmedCache
        cache = PubmedCache()
        stats = cache.get_stats()
        st.write(f"논문: {stats['article_count']}건 | 검색: {stats['search_count']}건")
        if st.button("캐시 초기화", key="clear_cache_btn"):
            cache.clear_cache()
            st.success("캐시가 초기화되었습니다")
            st.rerun()

        # Pipeline state indicator
        state = st.session_state.pipeline_state
        if state != "IDLE":
            st.markdown("### 파이프라인 상태")
            state_labels = {
                "IDLE": "대기",
                "PARSING": "주제 분석 중",
                "CONFIRM_QUERIES": "쿼리 확인 대기",
                "RESEARCHING": "리서치 수행 중",
                "CONFIRM_STRATEGY": "전략 확인 대기",
                "GENERATING": "Introduction 작성 중",
                "EVALUATING": "품질 평가 중",
                "SELF_EVOLVING": "자동 개선 중",
                "COMPLETE": "완료",
            }
            st.info(f"현재 상태: **{state_labels.get(state, state)}**")
            iteration = st.session_state.pipeline_iteration
            if iteration > 0:
                st.write(f"Self-evolution 반복: {iteration}/{MAX_EVOLUTION_ITERATIONS}")

        # History
        st.markdown("### 📋 생성 이력")
        if st.session_state.history:
            for i, item in enumerate(st.session_state.history):
                if st.button(f"{i+1}. {item['topic'][:40]}...", key=f"history_{i}"):
                    st.session_state.current_topic = item["topic"]
                    st.session_state.generation_result = item["result"]
                    st.session_state.pipeline_state = "COMPLETE"
        else:
            st.info("생성 이력이 없습니다")

        st.markdown("---")
        st.markdown("### 📖 정보")
        st.markdown(
            "[GitHub](https://github.com/anthropics/claude-code) | "
            "[문서](https://github.com/anthropics/claude-code/blob/main/README.md)"
        )

        return api_key, model, reference_style


# ------------------------------------------------------------------
# State renderers
# ------------------------------------------------------------------

def render_idle_state():
    """IDLE: Topic input and start"""
    st.markdown("## 연구 주제 입력")

    topic = st.text_area(
        "연구 주제를 한 줄로 입력하세요",
        placeholder="예: 뇌파 딥러닝 분석 기반 주요우울장애의 항우울제 치료 반응성 예측 연구",
        height=100,
        help="질환명, 데이터 유형, 분석 방법, 예측 대상 등을 포함하면 더 좋습니다"
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        start_btn = st.button("🚀 시작", key="start_btn")

    if start_btn:
        if not topic.strip():
            st.error("연구 주제를 입력하세요")
            return
        api_key = st.session_state.api_key_stored
        if not api_key or not api_key.startswith("sk-"):
            st.error("유효한 OpenAI API 키를 입력하세요")
            return

        st.session_state.current_topic = topic
        st.session_state.pipeline_state = "PARSING"
        st.rerun()


def render_parsing_state():
    """PARSING: Parse topic (auto-advance)"""
    st.markdown("## 주제 분석 중...")

    orch = get_orchestrator()
    if not orch:
        st.error("API 키를 확인하세요")
        st.session_state.pipeline_state = "IDLE"
        return

    with st.spinner("주제를 심층 분석하고 있습니다..."):
        try:
            topic_analysis = orch.parse_topic(st.session_state.current_topic)
            st.session_state.topic_analysis = topic_analysis
            st.session_state.search_queries = topic_analysis.get("search_queries", [])
            st.session_state.pipeline_state = "CONFIRM_QUERIES"
            st.rerun()
        except Exception as e:
            st.error(f"주제 분석 실패: {str(e)}")
            logger.error(f"Topic parsing error: {e}", exc_info=True)
            st.session_state.pipeline_state = "IDLE"


def render_confirm_queries_state():
    """CONFIRM_QUERIES: Checkpoint 1 - user reviews/edits queries"""
    st.markdown("## 검색 쿼리 확인 (체크포인트 1)")

    topic_analysis = st.session_state.topic_analysis
    if not topic_analysis:
        st.session_state.pipeline_state = "IDLE"
        st.rerun()
        return

    # Display parsed topic info
    st.markdown("### 파싱된 주제 정보")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**질환:** {topic_analysis.get('disease', 'N/A')}")
        st.write(f"**데이터 유형:** {topic_analysis.get('data_type', 'N/A')}")
        st.write(f"**핵심 초점:** {topic_analysis.get('key_intervention_or_focus', 'N/A')}")
    with col2:
        st.write(f"**방법론:** {topic_analysis.get('methodology', 'N/A')}")
        st.write(f"**예측 대상:** {topic_analysis.get('outcome', 'N/A')}")

    # Concept hierarchy
    concepts = topic_analysis.get("concept_hierarchy", [])
    if concepts:
        with st.expander("개념 계층구조", expanded=False):
            for concept in concepts[:8]:
                st.write(f"  - {concept}")

    # Editable search queries
    st.markdown("### 검색 쿼리 (편집 가능)")
    st.caption("한 줄에 하나의 쿼리를 입력하세요. 쿼리를 추가/삭제/수정할 수 있습니다.")

    queries_text = st.text_area(
        "검색 쿼리",
        value="\n".join(st.session_state.search_queries),
        height=300,
        key="edit_queries"
    )

    # Additional feedback
    additional_feedback = st.text_area(
        "추가 피드백 (선택사항)",
        placeholder="특별히 포함하고 싶은 검색 전략이나 강조점이 있으면 작성하세요",
        height=80,
        key="query_feedback"
    )

    col1, col2 = st.columns(2)
    with col1:
        confirm_btn = st.button("확인 & 리서치 시작", key="confirm_queries_btn")
    with col2:
        back_btn = st.button("주제 수정으로 돌아가기", key="back_to_idle_btn")

    if confirm_btn:
        edited_queries = [q.strip() for q in queries_text.split("\n") if q.strip()]
        if not edited_queries:
            st.error("최소 하나의 검색 쿼리가 필요합니다")
            return

        orch = get_orchestrator()
        if not orch:
            st.error("API 키를 확인하세요")
            return

        st.session_state.topic_analysis = orch.update_queries(
            st.session_state.topic_analysis, edited_queries
        )
        st.session_state.search_queries = edited_queries
        st.session_state.pipeline_state = "RESEARCHING"
        st.rerun()

    if back_btn:
        reset_pipeline()
        st.rerun()


def render_researching_state():
    """RESEARCHING: Run research pipeline with per-query progress"""
    st.markdown("## 리서치 수행 중...")

    orch = get_orchestrator()
    if not orch:
        st.error("API 키를 확인하세요")
        st.session_state.pipeline_state = "IDLE"
        return

    status_container = st.status("PubMed 리서치 진행 중...", expanded=True)

    try:
        with status_container:
            from utils.pubmed_utils import has_valid_abstract

            topic_analysis = st.session_state.topic_analysis
            search_queries = topic_analysis.get("search_queries", [])
            st.write(f"**{len(search_queries)}개 쿼리로 논문 수집 시작...**")

            # --- Paper collection with per-query feedback ---
            researcher = orch.deep_researcher
            all_papers = {}

            for i, query in enumerate(search_queries, 1):
                try:
                    papers = researcher.pubmed_client.search_and_fetch(
                        query, f"query_{i}", max_results=30
                    )
                    added = 0
                    retracted = 0
                    for paper in papers:
                        pmid = paper.get("pmid")
                        if pmid and pmid not in all_papers and has_valid_abstract(paper):
                            if paper.get("is_retracted", False):
                                retracted += 1
                                continue
                            all_papers[pmid] = paper
                            added += 1
                    retracted_note = f" (retracted {retracted}편 제외)" if retracted else ""
                    st.write(f"  Query {i}/{len(search_queries)}: +{added}편{retracted_note} (누적 {len(all_papers)}편) — `{query[:55]}`")
                except Exception as e:
                    st.write(f"  Query {i}/{len(search_queries)}: 실패 — {str(e)[:50]}")

            # High-impact journal search
            disease = topic_analysis.get("disease", "")
            if disease:
                try:
                    hi_papers = researcher.pubmed_client.search_and_fetch(
                        f"({disease}) AND (Nature[Journal] OR NEJM[Journal] OR Lancet[Journal] OR JAMA[Journal])",
                        "high_impact", max_results=40
                    )
                    added = 0
                    for paper in hi_papers:
                        pmid = paper.get("pmid")
                        if pmid and pmid not in all_papers and has_valid_abstract(paper):
                            if paper.get("is_retracted", False):
                                continue
                            all_papers[pmid] = paper
                            added += 1
                    st.write(f"  High-impact journals: +{added}편 (누적 {len(all_papers)}편)")
                except Exception as e:
                    st.write(f"  High-impact journals: 실패 — {str(e)[:50]}")

            paper_pool = list(all_papers.values())
            st.session_state.paper_pool = paper_pool
            st.write(f"**총 {len(paper_pool)}개 논문 수집 완료**")

        # --- 0 papers guard ---
        if not paper_pool:
            st.error(
                "PubMed에서 논문을 찾지 못했습니다. 가능한 원인:\n"
                "- 검색 쿼리가 너무 구체적이거나 PubMed 문법에 맞지 않음\n"
                "- PubMed API 일시 장애 또는 네트워크 문제\n\n"
                "쿼리를 수정한 뒤 다시 시도해 주세요."
            )
            st.session_state.pipeline_state = "CONFIRM_QUERIES"
            return

        # --- Landscape analysis ---
        with st.status("문헌 분석 및 전략 수립 중...", expanded=True) as status2:
            st.write("**문헌 경관 분석 중...**")
            landscape = orch.intro_generator.step_analyze_landscape(
                paper_pool, st.session_state.current_topic
            )
            st.session_state.landscape = landscape
            st.write(f"  핵심 발견사항 {len(landscape.get('key_findings', []))}개, "
                      f"미충족 분야 {len(landscape.get('knowledge_gaps', []))}개 식별")

            # --- Reference pool selection ---
            st.write("**최적 논문 풀 선별 중...**")
            reference_pool = orch.intro_generator.step_select_references(paper_pool, landscape)
            st.session_state.reference_pool = reference_pool
            st.write(f"  {len(reference_pool)}편 선별 완료")

            # --- Writing strategy ---
            st.write("**Writing strategy 생성 중...**")
            strategy = orch.generate_writing_strategy(
                st.session_state.topic_analysis, reference_pool, landscape
            )
            st.session_state.writing_strategy = strategy
            st.write("**리서치 완료!**")

        st.session_state.pipeline_state = "CONFIRM_STRATEGY"
        st.rerun()

    except Exception as e:
        st.error(f"리서치 실패: {str(e)}")
        logger.error(f"Research error: {e}", exc_info=True)
        st.session_state.pipeline_state = "CONFIRM_QUERIES"


def render_confirm_strategy_state():
    """CONFIRM_STRATEGY: Checkpoint 2 - user reviews strategy"""
    st.markdown("## Writing Strategy 확인 (체크포인트 2)")

    # Research summary
    st.markdown("### 리서치 요약")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("수집된 논문", len(st.session_state.paper_pool))
    with col2:
        st.metric("선별된 논문", len(st.session_state.reference_pool))
    with col3:
        landscape = st.session_state.landscape
        findings_count = len(landscape.get("key_findings", []))
        st.metric("핵심 발견사항", findings_count)

    # Writing strategy display
    strategy = st.session_state.writing_strategy
    paragraphs = strategy.get("paragraphs", [])
    ref_pool = st.session_state.reference_pool

    if paragraphs:
        st.markdown("### Paragraph별 계획")
        for para in paragraphs:
            num = para.get("paragraph_number", "?")
            topic = para.get("topic", "")
            key_points = para.get("key_points", [])
            refs = para.get("supporting_papers", [])
            transition = para.get("transition_to_next", "")

            with st.expander(f"Paragraph {num}: {topic}", expanded=True):
                st.markdown("**Key Points:**")
                for pt in key_points:
                    st.write(f"  - {pt}")

                # Show reference details
                if refs and ref_pool:
                    st.markdown("**Supporting References:**")
                    for ref_num in refs:
                        idx = ref_num - 1  # 1-indexed to 0-indexed
                        if 0 <= idx < len(ref_pool):
                            paper = ref_pool[idx]
                            authors = paper.get("authors", [])
                            first_author = authors[0] if authors else "Unknown"
                            et_al = " et al." if len(authors) > 1 else ""
                            title = paper.get("title", "")[:80]
                            journal = paper.get("journal_iso", paper.get("journal", ""))
                            year = paper.get("pub_year", "")
                            st.write(f"  [{ref_num}] {first_author}{et_al} — {title}... *{journal}* ({year})")
                        else:
                            st.write(f"  [{ref_num}] (범위 밖)")
                elif refs:
                    st.write(f"**Supporting references:** [{', '.join(str(r) for r in refs)}]")

                if transition:
                    st.write(f"**Transition:** {transition}")

    narrative_arc = strategy.get("narrative_arc", "")
    if narrative_arc:
        st.markdown("### Narrative Arc")
        st.info(narrative_arc)

    # Landscape details
    with st.expander("문헌 경관 분석 상세", expanded=False):
        _display_landscape(st.session_state.landscape)

    # Feedback
    st.markdown("---")
    feedback = st.text_area(
        "전략에 대한 피드백 (선택사항)",
        placeholder="예: '방법론 파트를 더 자세하게' 또는 '특정 논문을 더 강조해줘'",
        height=80,
        key="strategy_feedback"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        generate_btn = st.button("Introduction 작성", key="generate_intro_btn")
    with col2:
        add_research_btn = st.button("추가 리서치 후 작성", key="add_research_btn")
    with col3:
        back_btn = st.button("쿼리 수정으로 돌아가기", key="back_to_queries_btn")

    if generate_btn:
        st.session_state.pipeline_state = "GENERATING"
        st.rerun()

    if add_research_btn:
        st.markdown("### 추가 리서치")
        input_mode = st.radio(
            "입력 방식",
            ["자연어 피드백 (AI가 쿼리 생성)", "직접 PubMed 쿼리 입력"],
            key="extra_research_mode"
        )

        if input_mode == "자연어 피드백 (AI가 쿼리 생성)":
            nl_feedback = st.text_area(
                "어떤 부분이 부족한지 자연어로 설명하세요",
                placeholder="예: treatment-resistant MDD 관련 논문이 부족해요 / 방법론 파트에 EEG 전처리 관련 내용이 필요해요",
                height=100,
                key="nl_feedback_input"
            )
            if st.button("AI 쿼리 생성 & 리서치 실행", key="run_nl_research_btn"):
                if not nl_feedback.strip():
                    st.warning("피드백을 입력하세요")
                else:
                    orch = get_orchestrator()
                    if not orch:
                        st.error("API 키를 확인하세요")
                    else:
                        with st.spinner("피드백을 분석하여 검색 쿼리를 생성하고 있습니다..."):
                            result = orch.generate_queries_from_feedback(
                                user_feedback=nl_feedback,
                                writing_strategy=st.session_state.writing_strategy,
                                topic_analysis=st.session_state.topic_analysis,
                                landscape=st.session_state.landscape
                            )
                        interpretation = result.get("interpretation", "")
                        queries = result.get("queries", [])
                        if interpretation:
                            st.info(f"**AI 해석:** {interpretation}")
                        if queries:
                            st.write(f"**생성된 쿼리 ({len(queries)}개):**")
                            for i, q in enumerate(queries, 1):
                                st.write(f"  {i}. `{q}`")
                            _run_supplementary_research(queries)
                        else:
                            st.warning("쿼리를 생성하지 못했습니다. 직접 입력을 시도해 주세요.")
        else:
            extra_queries = st.text_area(
                "추가 쿼리 입력 (한 줄에 하나)",
                placeholder="추가로 검색하고 싶은 쿼리를 입력하세요",
                height=120,
                key="extra_queries_input"
            )
            if st.button("추가 리서치 실행", key="run_extra_research_btn"):
                queries = [q.strip() for q in extra_queries.split("\n") if q.strip()]
                if not queries:
                    st.warning("추가 쿼리를 입력하세요")
                else:
                    _run_supplementary_research(queries)

    if back_btn:
        st.session_state.pipeline_state = "CONFIRM_QUERIES"
        st.rerun()


def _run_supplementary_research(queries: list):
    """Run supplementary research and update state"""
    orch = get_orchestrator()
    if not orch:
        st.error("API 키를 확인하세요")
        return

    with st.spinner("추가 리서치 수행 중..."):
        try:
            expanded_pool, updated_landscape, new_ref_pool = orch.run_supplementary_research(
                additional_queries=queries,
                paper_pool=st.session_state.paper_pool,
                landscape=st.session_state.landscape,
                research_topic=st.session_state.current_topic,
                topic_analysis=st.session_state.topic_analysis,
            )
            old_count = len(st.session_state.paper_pool)
            st.session_state.paper_pool = expanded_pool
            st.session_state.landscape = updated_landscape
            st.session_state.reference_pool = new_ref_pool

            # Re-generate writing strategy
            strategy = orch.generate_writing_strategy(
                st.session_state.topic_analysis, new_ref_pool, updated_landscape
            )
            st.session_state.writing_strategy = strategy

            new_count = len(expanded_pool) - old_count
            st.success(f"추가 리서치 완료! {new_count}편 새 논문 발견. Reference pool: {len(new_ref_pool)}편")
            st.rerun()
        except Exception as e:
            st.error(f"추가 리서치 실패: {str(e)}")
            logger.error(f"Supplementary research error: {e}", exc_info=True)


def render_generating_state():
    """GENERATING: Generate introduction (auto-advance)"""
    st.markdown("## Introduction 작성 중...")

    orch = get_orchestrator()
    if not orch:
        st.error("API 키를 확인하세요")
        st.session_state.pipeline_state = "IDLE"
        return

    with st.spinner("Introduction을 작성하고 있습니다..."):
        try:
            introduction = orch.generate_introduction(
                st.session_state.topic_analysis,
                st.session_state.reference_pool,
                st.session_state.landscape
            )
            st.session_state.introduction_text = introduction
            st.session_state.pipeline_state = "EVALUATING"
            st.rerun()
        except Exception as e:
            st.error(f"Introduction 작성 실패: {str(e)}")
            logger.error(f"Generation error: {e}", exc_info=True)
            st.session_state.pipeline_state = "CONFIRM_STRATEGY"


def render_evaluating_state():
    """EVALUATING: Run 8-criterion evaluation (auto-advance)"""
    st.markdown("## 품질 평가 중...")

    orch = get_orchestrator()
    if not orch:
        st.error("API 키를 확인하세요")
        st.session_state.pipeline_state = "IDLE"
        return

    with st.spinner("8개 기준으로 품질을 평가하고 있습니다..."):
        try:
            evaluation = orch.evaluate_introduction(
                st.session_state.introduction_text,
                st.session_state.reference_pool,
                st.session_state.topic_analysis,
                st.session_state.landscape
            )
            st.session_state.evaluation_result = evaluation

            # Auto fact-check
            try:
                fact_result = orch.run_fact_check(
                    st.session_state.introduction_text,
                    st.session_state.reference_pool
                )
                st.session_state.fact_check_result = fact_result

                # Adjust factual_accuracy based on fact-check
                fc_accuracy = fact_result.get("overall_accuracy", "HIGH")
                current_fa = evaluation.get("scores", {}).get("factual_accuracy", 10)
                if fc_accuracy == "LOW":
                    evaluation["scores"]["factual_accuracy"] = min(current_fa, 5)
                elif fc_accuracy == "MEDIUM":
                    evaluation["scores"]["factual_accuracy"] = min(current_fa, 6)

                # Recalculate overall score
                scores = evaluation.get("scores", {})
                if scores:
                    evaluation["overall_score"] = round(
                        sum(scores.values()) / len(scores), 1
                    )
            except Exception as fc_err:
                logger.warning(f"Auto fact-check failed (non-blocking): {fc_err}")

            # Record iteration history
            iteration = st.session_state.pipeline_iteration
            st.session_state.iteration_history.append({
                "iteration": iteration,
                "label": "Draft" if iteration == 0 else f"Rev{iteration}",
                "introduction": st.session_state.introduction_text,
                "evaluation": evaluation,
            })

            # Check if self-evolution is needed
            if (
                orch.needs_self_evolution(evaluation)
                and st.session_state.pipeline_iteration < MAX_EVOLUTION_ITERATIONS
            ):
                st.session_state.pipeline_state = "SELF_EVOLVING"
            else:
                st.session_state.pipeline_state = "COMPLETE"

            st.rerun()

        except Exception as e:
            st.error(f"품질 평가 실패: {str(e)}")
            logger.error(f"Evaluation error: {e}", exc_info=True)
            # Still go to COMPLETE with what we have
            st.session_state.pipeline_state = "COMPLETE"
            st.rerun()


def render_self_evolving_state():
    """SELF_EVOLVING: Auto-improve introduction"""
    iteration = st.session_state.pipeline_iteration + 1
    st.markdown(f"## 자동 개선 중 (Iteration {iteration}/{MAX_EVOLUTION_ITERATIONS})")

    orch = get_orchestrator()
    if not orch:
        st.error("API 키를 확인하세요")
        st.session_state.pipeline_state = "COMPLETE"
        st.rerun()
        return

    status_container = st.status("Self-evolution 진행 중...", expanded=True)

    try:
        with status_container:
            # Step 1: Extract unsupported claims
            st.write("미지지 클레임 추출 중...")
            claims = orch.extract_unsupported_claims(
                st.session_state.evaluation_result,
                st.session_state.introduction_text
            )
            st.write(f"  {len(claims)}개 미지지 클레임 발견")

            if not claims:
                st.write("미지지 클레임이 없습니다. 완료로 이동합니다.")
                st.session_state.pipeline_state = "COMPLETE"
                st.rerun()
                return

            # Step 2: Generate supplementary queries
            st.write("보충 쿼리 생성 중...")
            queries = orch.generate_supplementary_queries(
                claims, st.session_state.topic_analysis
            )
            st.write(f"  {len(queries)}개 보충 쿼리 생성")

            # Step 3: Search PubMed
            st.write("PubMed 보충 검색 중...")
            new_papers = orch.run_supplementary_search(
                queries, st.session_state.paper_pool
            )
            st.write(f"  {len(new_papers)}편 새 논문 발견")

            # Step 4: Expand reference pool
            if new_papers:
                st.write("Reference pool 확장 중...")
                st.session_state.paper_pool = st.session_state.paper_pool + new_papers
                new_ref_pool = orch.expand_reference_pool(
                    st.session_state.reference_pool,
                    new_papers,
                    st.session_state.landscape
                )
                old_size = len(st.session_state.reference_pool)
                st.session_state.reference_pool = new_ref_pool
                st.write(f"  Reference pool: {old_size} -> {len(new_ref_pool)}편")

            # Step 5: Regenerate introduction
            st.write("Introduction 재작성 중...")
            introduction = orch.generate_introduction(
                st.session_state.topic_analysis,
                st.session_state.reference_pool,
                st.session_state.landscape
            )
            st.session_state.introduction_text = introduction

        st.session_state.pipeline_iteration = iteration
        st.session_state.pipeline_state = "EVALUATING"
        st.rerun()

    except Exception as e:
        st.error(f"자동 개선 실패: {str(e)}")
        logger.error(f"Self-evolution error: {e}", exc_info=True)
        st.session_state.pipeline_state = "COMPLETE"
        st.rerun()


def render_complete_state(reference_style: str = "APA"):
    """COMPLETE: Show final results"""
    st.markdown("## 📝 최종 결과")

    # Build final result for display & history
    result = _build_final_result()
    st.session_state.generation_result = result

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("수집된 논문", len(st.session_state.paper_pool))
    with col2:
        st.metric("선별된 논문", len(st.session_state.reference_pool))
    with col3:
        iterations = st.session_state.pipeline_iteration
        label = "Draft" if iterations == 0 else f"Draft + {iterations} Rev"
        st.metric("Iterations", label)

    # Iteration history tabs
    history = st.session_state.iteration_history
    if len(history) > 1:
        st.markdown("### 버전 비교")
        tab_names = [h["label"] for h in history]
        tabs = st.tabs(tab_names)
        for tab, h in zip(tabs, history):
            with tab:
                eval_data = h.get("evaluation", {})
                overall = eval_data.get("overall_score", "?")
                factual = eval_data.get("scores", {}).get("factual_accuracy", "?")
                st.write(f"**종합 점수:** {overall}/10 | **Factual accuracy:** {factual}/10")
                st.markdown(h["introduction"])
    elif history:
        # Single version — show it directly
        pass

    # Display final introduction
    st.markdown("---")
    st.markdown("### Introduction")
    st.markdown(st.session_state.introduction_text)

    # References
    st.markdown("### 📖 References")
    ref_pool = st.session_state.reference_pool
    if ref_pool:
        formatted_refs = _format_references_by_style(ref_pool, reference_style)
        for ref in formatted_refs:
            st.markdown(ref)

    # Evaluation results
    evaluation = st.session_state.evaluation_result
    if evaluation:
        display_self_evaluation_results(evaluation)

    # Auto fact-check results
    fact_result = st.session_state.get("fact_check_result")
    if fact_result:
        st.markdown("---")
        st.markdown("## 자동 팩트체크 결과")
        fc_accuracy = fact_result.get("overall_accuracy", "UNKNOWN")
        fc_issues = fact_result.get("issues", [])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("정확도", fc_accuracy)
        with col2:
            st.metric("발견된 이슈", len(fc_issues))

        # Claim-citation mapping details
        claim_mapping = fact_result.get("claim_mapping", {})
        claim_mappings = claim_mapping.get("claim_mappings", [])
        if claim_mappings:
            with st.expander(f"Claim-Citation 매핑 검증 ({len(claim_mappings)}개 claim)", expanded=False):
                for cm in claim_mappings:
                    supported = cm.get("is_supported", True)
                    icon = "+" if supported else "X"
                    claim_text = cm.get("claim", "")[:120]
                    st.write(f"{icon} **{claim_text}**{'...' if len(cm.get('claim', '')) > 120 else ''}")
                    if not supported and cm.get("issue"):
                        st.write(f"   Issue: {cm['issue']}")

        numerical_mismatches = claim_mapping.get("numerical_mismatches", [])
        if numerical_mismatches:
            with st.expander(f"수치 불일치 ({len(numerical_mismatches)}건)", expanded=False):
                for nm in numerical_mismatches:
                    severity = nm.get("severity", "minor").upper()
                    st.write(f"[{severity}] Claimed: {nm.get('claimed_value', '?')} vs Actual: {nm.get('actual_value', '?')}")
                    st.write(f"  {nm.get('claim', '')[:100]}")

        if fc_issues:
            with st.expander(f"팩트체크 이슈 상세 ({len(fc_issues)}건)", expanded=False):
                for issue in fc_issues:
                    st.write(f"- **{issue.get('type', '')}**: {issue.get('description', '')}")

    # Landscape details
    with st.expander("🌍 문헌 경관 분석", expanded=False):
        _display_landscape(st.session_state.landscape)

    # Topic details
    topic_analysis = st.session_state.topic_analysis
    if topic_analysis:
        with st.expander("📋 파싱된 주제 정보", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**질환:** {topic_analysis.get('disease', 'N/A')}")
                st.write(f"**데이터 유형:** {topic_analysis.get('data_type', 'N/A')}")
            with col2:
                st.write(f"**방법론:** {topic_analysis.get('methodology', 'N/A')}")
                st.write(f"**예측 대상:** {topic_analysis.get('outcome', 'N/A')}")

    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 Introduction 복사"):
            st.write("복사되었습니다 (브라우저 복사 기능 사용)")

    with col2:
        fc_label = "✅ 팩트체크 재실행" if fact_result else "✅ 팩트체크 실행"
        if st.button(fc_label, key="factcheck_btn"):
            run_fact_check(result)

    with col3:
        if st.button("🔄 새 주제 시작", key="new_topic_btn"):
            # Save to history before reset
            if st.session_state.current_topic and result:
                st.session_state.history.append({
                    "topic": st.session_state.current_topic,
                    "result": result
                })
            reset_pipeline()
            st.rerun()

    # Manual revision section
    st.markdown("---")
    st.markdown("### 수정하고 싶은 부분이 있으신가요?")

    revision_request = st.text_area(
        "수정 요청 입력",
        placeholder="예: '두 번째 문단을 더 보강해줘' 또는 '전체적으로 톤을 더 formal하게 바꿔줄래'",
        height=80,
        key="revision_request"
    )

    if st.button("수정 명령 실행", key="run_revision_btn"):
        if not revision_request:
            st.warning("수정 요청을 입력하세요")
        else:
            run_revision(
                st.session_state.introduction_text,
                revision_request,
                st.session_state.api_key_stored,
                st.session_state.model_stored,
                st.session_state.reference_pool
            )


def _build_final_result() -> dict:
    """Build a result dictionary from current pipeline state"""
    from utils.pubmed_utils import format_citation_vancouver
    references = []
    for i, article in enumerate(st.session_state.reference_pool, 1):
        citation = format_citation_vancouver(article, i)
        references.append(citation)

    return {
        "introduction": st.session_state.introduction_text,
        "references": references,
        "articles_used": st.session_state.reference_pool,
        "parsing_result": st.session_state.topic_analysis or {},
        "landscape": st.session_state.landscape,
        "paper_pool_size": len(st.session_state.paper_pool),
        "reference_pool_size": len(st.session_state.reference_pool),
        "evaluation": st.session_state.evaluation_result,
    }


# ------------------------------------------------------------------
# Shared display helpers
# ------------------------------------------------------------------

def _display_landscape(landscape: dict):
    """Display landscape analysis details"""
    field_overview = landscape.get("field_overview", "")
    if field_overview:
        st.markdown("**분야 개요:**")
        st.markdown(field_overview)

    key_findings = landscape.get("key_findings", [])
    if key_findings:
        st.markdown(f"\n**핵심 발견사항 ({len(key_findings)}개):**")
        for i, finding in enumerate(key_findings, 1):
            st.markdown(f"{i}. {finding}")

    knowledge_gaps = landscape.get("knowledge_gaps", [])
    if knowledge_gaps:
        st.markdown(f"\n**미충족 연구 필요 분야 ({len(knowledge_gaps)}개):**")
        for i, gap in enumerate(knowledge_gaps, 1):
            st.markdown(f"{i}. {gap}")

    trends = landscape.get("methodological_trends", [])
    if trends:
        st.markdown(f"\n**방법론적 동향 ({len(trends)}개):**")
        for i, trend in enumerate(trends, 1):
            st.markdown(f"{i}. {trend}")

    controversies = landscape.get("controversies_or_debates", [])
    if controversies:
        st.markdown(f"\n**논란/미해결 쟁점 ({len(controversies)}개):**")
        for i, c in enumerate(controversies, 1):
            st.markdown(f"{i}. {c}")

    underexplored = landscape.get("underexplored_areas", [])
    if underexplored:
        st.markdown(f"\n**미탐구 영역 ({len(underexplored)}개):**")
        for i, area in enumerate(underexplored, 1):
            st.markdown(f"{i}. {area}")


def display_self_evaluation_results(evaluation: dict):
    """Display self-evaluation results"""
    st.markdown("---")
    st.markdown("## 🎯 자동 품질 평가")

    overall_score = evaluation.get("overall_score", 0)
    passed = evaluation.get("passed", False)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("종합 점수", f"{overall_score}/10")
    with col2:
        status = "합격" if passed else "검토 필요"
        st.markdown(f"**상태:** {status}")
    with col3:
        improvement_count = len(evaluation.get("improvements", []))
        st.metric("개선 항목", improvement_count)

    # Detailed scores
    st.markdown("### 세부 점수")
    scores = evaluation.get("scores", {})
    cols = st.columns(4)
    criteria = list(scores.items())
    for i, (criterion, score) in enumerate(criteria):
        col_idx = i % 4
        with cols[col_idx]:
            if score >= 8:
                emoji = "✅"
            elif score >= 7:
                emoji = "O"
            elif score >= 5:
                emoji = "△"
            else:
                emoji = "X"
            st.write(f"{emoji} **{criterion.replace('_', ' ').title()}**")
            st.write(f"점수: {score}/10")

    # Feedback
    feedback = evaluation.get("feedback", {})
    improvements = evaluation.get("improvements", [])

    st.markdown("### 평가 피드백")
    for criterion_key, fb_text in feedback.items():
        criterion_label = criterion_key.replace("_", " ").title()
        criterion_score = scores.get(criterion_key, "?")
        with st.expander(f"{criterion_label} (점수: {criterion_score}/10)", expanded=False):
            st.markdown(f"**피드백:** {fb_text}")
            for imp in improvements:
                if imp["criterion"] == criterion_key:
                    st.markdown(f"\n**개선 제안:** {imp.get('improvement', '')}")
                    break

    if not improvements:
        st.success("✅ 모든 기준에서 7점 이상을 획득했습니다!")


def run_fact_check(generation_result: dict):
    """Run fact-checking on generated introduction"""
    st.markdown("### 🔍 팩트체크")

    with st.spinner("팩트체크 진행 중..."):
        try:
            fact_checker = FactChecker()
            check_result = fact_checker.verify_introduction(
                generation_result.get("introduction", ""),
                generation_result.get("articles_used", [])
            )

            st.session_state.fact_check_result = check_result

            accuracy = check_result.get("overall_accuracy", "UNKNOWN")
            st.metric("전체 정확도", accuracy, delta="Verified")

            issues = check_result.get("issues", [])
            if issues:
                st.warning(f"⚠️ {len(issues)}개의 잠재적 문제 발견")
                for issue in issues:
                    st.write(f"- **{issue.get('type')}**: {issue.get('description', '')}")
            else:
                st.success("✅ 모든 인용이 검증되었습니다")

            st.info(check_result.get("summary", ""))

        except Exception as e:
            st.error(f"팩트체크 실패: {str(e)}")
            logger.error(f"Fact check error: {e}", exc_info=True)


def run_revision(
    current_intro: str,
    revision_request: str,
    api_key: str,
    model: str,
    articles_used: list
):
    """Run revision on introduction"""
    if not revision_request.strip():
        st.warning("수정 요청을 입력하세요")
        return

    try:
        from prompts.revision import get_revision_prompt

        st.markdown("### 수정 진행 중...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.write("Revision 프롬프트 생성 중...")
        progress_bar.progress(0.3)

        llm_client = get_llm_client(api_key=api_key, model=model)
        system_prompt, user_prompt = get_revision_prompt(
            current_intro, revision_request, articles_used
        )

        status_text.write("LLM으로 수정 생성 중...")
        progress_bar.progress(0.7)

        revised_intro = llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=2000
        )

        progress_bar.progress(1.0)

        st.session_state.introduction_text = revised_intro
        st.session_state.current_intro = revised_intro
        st.success("✅ 수정 완료!")

        st.markdown("### 수정된 Introduction")
        st.markdown(revised_intro)

        if st.button("수정 버전 팩트체크", key="revised_factcheck"):
            fact_checker = FactChecker()
            check_result = fact_checker.verify_introduction(revised_intro, articles_used)
            accuracy = check_result.get("overall_accuracy", "UNKNOWN")
            st.metric("정확도", accuracy)
            issues = check_result.get("issues", [])
            if issues:
                st.warning(f"⚠️ {len(issues)}개 항목 확인 필요")
            else:
                st.success("✅ 팩트체크 완료")

    except Exception as e:
        st.error(f"수정 실패: {str(e)}")
        logger.error(f"Revision error: {e}", exc_info=True)


def _format_references_by_style(articles: list, style: str = "APA") -> list:
    """Format article list into citation strings"""
    refs = []
    for i, article in enumerate(articles, 1):
        authors = article.get("authors", [])
        title = article.get("title", "Untitled")
        journal = article.get("journal", "")
        journal_iso = article.get("journal_iso", journal)
        year = article.get("pub_year", "n.d.")
        pmid = article.get("pmid", "")
        doi = article.get("doi", "")

        if style == "APA":
            apa_authors = []
            for a in authors[:6]:
                parts = a.split()
                if len(parts) >= 2:
                    last = parts[0]
                    initials = ". ".join(p[0] + "." for p in parts[1:] if p)
                    apa_authors.append(f"{last}, {initials}")
                else:
                    apa_authors.append(a)
            if len(authors) > 6:
                author_str = ", ".join(apa_authors) + ", ... et al."
            elif len(apa_authors) > 1:
                author_str = ", ".join(apa_authors[:-1]) + ", & " + apa_authors[-1]
            elif apa_authors:
                author_str = apa_authors[0]
            else:
                author_str = "Unknown"
            doi_part = f" https://doi.org/{doi}" if doi else ""
            ref = f"{i}. {author_str} ({year}). {title}. *{journal_iso}*.{doi_part} PMID: {pmid}"

        elif style == "Vancouver":
            van_authors = []
            for a in authors[:3]:
                van_authors.append(a)
            if len(authors) > 3:
                author_str = ", ".join(van_authors) + " et al."
            else:
                author_str = ", ".join(van_authors)
            ref = f"{i}. {author_str}. {title}. {journal_iso}. {year}. PMID: {pmid}."

        else:  # AMA
            ama_authors = []
            for a in authors[:3]:
                ama_authors.append(a)
            if len(authors) > 3:
                author_str = ", ".join(ama_authors) + ", et al."
            else:
                author_str = ", ".join(ama_authors)
            ref = f"{i}. {author_str}. {title}. *{journal_iso}*. {year}. PMID: {pmid}."

        refs.append(ref)
    return refs


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    """Main application flow — state machine dispatcher"""
    initialize_session()

    api_key, model, reference_style = setup_sidebar()

    display_header()
    st.markdown("---")

    state = st.session_state.pipeline_state

    if state == "IDLE":
        render_idle_state()
    elif state == "PARSING":
        render_parsing_state()
    elif state == "CONFIRM_QUERIES":
        render_confirm_queries_state()
    elif state == "RESEARCHING":
        render_researching_state()
    elif state == "CONFIRM_STRATEGY":
        render_confirm_strategy_state()
    elif state == "GENERATING":
        render_generating_state()
    elif state == "EVALUATING":
        render_evaluating_state()
    elif state == "SELF_EVOLVING":
        render_self_evolving_state()
    elif state == "COMPLETE":
        render_complete_state(reference_style=reference_style)
    else:
        st.error(f"Unknown pipeline state: {state}")
        reset_pipeline()
        st.rerun()


if __name__ == "__main__":
    main()
