from typing import TypedDict, List, Optional, Literal, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from datetime import date
import logging

from .generate_story import StoryBookGenerator
from .generate_eval import StoryEvaluator
from .generate_summary import Summarizer

logger = logging.getLogger(__name__)


class StoryState(TypedDict):
    """동화 생성 파이프라인의 상태를 정의하는 클래스"""
    # 입력 파라미터
    name: str
    age: int
    genre: str
    uid: int
    type: int

    # 중간 생성 결과
    story_title: Optional[str]
    story_content: Optional[List[str]]
    prompt: Optional[str]

    # 평가 관련
    evaluation_scores: Optional[List[int]]
    retry_count: int

    # 최종 결과
    summary: Optional[str]
    fid: Optional[int]

    # 에러 처리
    error: Optional[str]
    success: bool


class StoryPipeline:
    """LangGraph 기반 동화 생성 파이프라인"""

    MAX_RETRIES = 10
    MIN_SCORE_THRESHOLD = 1

    def __init__(
        self,
        story_generator: StoryBookGenerator,
        story_evaluator: StoryEvaluator,
        summarizer: Summarizer,
    ):
        """
        Args:
            story_generator: 동화 생성 모델
            story_evaluator: 동화 평가 모델
            summarizer: 요약 생성 모델
        """
        self.story_generator = story_generator
        self.story_evaluator = story_evaluator
        self.summarizer = summarizer

        # StateGraph 구성
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """LangGraph StateGraph 구성"""
        workflow = StateGraph(StoryState)

        # 노드 추가
        workflow.add_node("generate_story", self._generate_story_node)
        workflow.add_node("evaluate_story", self._evaluate_story_node)
        workflow.add_node("summarize_story", self._summarize_story_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # 시작점 설정
        workflow.set_entry_point("generate_story")

        # 엣지 정의
        workflow.add_edge("generate_story", "evaluate_story")

        # 조건부 엣지: 평가 후 재생성 여부 결정
        workflow.add_conditional_edges(
            "evaluate_story",
            self._should_regenerate,
            {
                "regenerate": "generate_story",
                "summarize": "summarize_story",
                "error": "handle_error"
            }
        )

        # 요약 후 종료
        workflow.add_edge("summarize_story", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    def _generate_story_node(self, state: StoryState) -> StoryState:
        """동화 생성 노드"""
        try:
            logger.info(f"동화 생성 시작 (시도 {state['retry_count'] + 1}/{self.MAX_RETRIES})")

            result = self.story_generator.generate_story(
                name=state["name"],
                age=state["age"],
                genre=state["genre"]
            )

            state["story_title"] = result["title"]
            state["story_content"] = result["content"]
            state["prompt"] = result["prompt"]
            state["retry_count"] = state.get("retry_count", 0) + 1

            logger.info(f"동화 생성 완료: {result['title']}")

        except Exception as e:
            logger.error(f"동화 생성 실패: {e}", exc_info=True)
            state["error"] = f"동화 생성 실패: {str(e)}"
            state["success"] = False

        return state

    def _evaluate_story_node(self, state: StoryState) -> StoryState:
        """동화 평가 노드"""
        try:
            logger.info("동화 평가 시작")

            story_text = " ".join(state["story_content"])
            eval_result = self.story_evaluator.evaluate_single_story_fast(
                story_text=story_text,
                prompt=state["prompt"]
            )

            state["evaluation_scores"] = eval_result["scores"]
            logger.info(f"평가 점수: {eval_result['scores']}")

        except Exception as e:
            logger.error(f"동화 평가 실패: {e}", exc_info=True)
            state["error"] = f"동화 평가 실패: {str(e)}"
            state["success"] = False

        return state

    def _summarize_story_node(self, state: StoryState) -> StoryState:
        """동화 요약 노드"""
        try:
            logger.info("동화 요약 시작")

            summary_result = self.summarizer.generate_summary(
                uid=state["uid"],
                type=state["type"],
                title=state["story_title"],
                contents=" ".join(state["story_content"]),
                max_new_tokens=200
            )

            state["summary"] = summary_result["summary"]
            state["success"] = True

            logger.info("동화 요약 완료")

        except Exception as e:
            logger.error(f"동화 요약 실패: {e}", exc_info=True)
            state["error"] = f"동화 요약 실패: {str(e)}"
            state["success"] = False

        return state

    def _handle_error_node(self, state: StoryState) -> StoryState:
        """에러 처리 노드"""
        logger.error(f"파이프라인 실패: {state.get('error', 'Unknown error')}")
        state["success"] = False
        return state

    def _should_regenerate(self, state: StoryState) -> Literal["regenerate", "summarize", "error"]:
        """
        평가 점수를 기반으로 재생성 여부를 결정하는 조건부 라우터
        """
        # 에러가 있으면 바로 에러 처리
        if state.get("error"):
            return "error"

        scores = state.get("evaluation_scores", [])
        retry_count = state.get("retry_count", 0)

        # 모든 점수가 임계값보다 높은지 확인
        all_scores_good = all(score > self.MIN_SCORE_THRESHOLD for score in scores)

        if all_scores_good:
            logger.info("평가 통과 → 요약 단계로 진행")
            return "summarize"
        elif retry_count >= self.MAX_RETRIES:
            logger.warning(f"최대 재시도 횟수({self.MAX_RETRIES}) 초과")
            state["error"] = f"10번 시도하였으나 유효한 동화를 생성하지 못했습니다."
            return "error"
        else:
            logger.info(f"평가 미달 → 재생성 (시도 {retry_count}/{self.MAX_RETRIES})")
            return "regenerate"

    def run(
        self,
        name: str,
        age: int,
        genre: str,
        uid: int,
        type: int = 2
    ) -> StoryState:
        """
        동화 생성 파이프라인 실행

        Args:
            name: 주인공 이름
            age: 나이
            genre: 장르
            uid: 사용자 ID
            type: 동화 타입 (기본값: 2)

        Returns:
            최종 상태 딕셔너리
        """
        initial_state: StoryState = {
            "name": name,
            "age": age,
            "genre": genre,
            "uid": uid,
            "type": type,
            "story_title": None,
            "story_content": None,
            "prompt": None,
            "evaluation_scores": None,
            "retry_count": 0,
            "summary": None,
            "fid": None,
            "error": None,
            "success": False,
        }

        logger.info(f"파이프라인 시작: name={name}, age={age}, genre={genre}")

        # 그래프 실행
        final_state = self.graph.invoke(initial_state)

        logger.info(f"파이프라인 완료: success={final_state.get('success')}")

        return final_state

    def stream(
        self,
        name: str,
        age: int,
        genre: str,
        uid: int,
        type: int = 2
    ):
        """
        동화 생성 파이프라인을 스트리밍 방식으로 실행

        각 노드의 실행 결과를 실시간으로 yield합니다.
        """
        initial_state: StoryState = {
            "name": name,
            "age": age,
            "genre": genre,
            "uid": uid,
            "type": type,
            "story_title": None,
            "story_content": None,
            "prompt": None,
            "evaluation_scores": None,
            "retry_count": 0,
            "summary": None,
            "fid": None,
            "error": None,
            "success": False,
        }

        logger.info(f"파이프라인 스트리밍 시작: name={name}, age={age}, genre={genre}")

        # 그래프를 스트리밍 방식으로 실행
        for output in self.graph.stream(initial_state):
            yield output
