"""Day 1 — 운영 알람 트리아지 체인.

알람 하나를 넣으면 구조화된 분류 결과와 담당팀 안내 초안을 동시에 만듭니다.

[중요] 이 파일의 규칙
  1. 함수 이름과 인자를 바꾸지 마세요. 채점기가 이 이름으로 찾습니다.
  2. 모델은 반드시 함수 **안에서** 만듭니다. 모듈 최상단에서 만들면
     채점기가 가짜 모델을 넣을 수 없어 채점이 실패합니다.
  3. `llm` 인자가 들어오면 그것을 쓰고, None이면 직접 만듭니다.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field

# 과정 전체에서 고정으로 쓰는 값입니다. 바꾸지 마세요.
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
REGION = "us-east-1"


def _default_llm():
    """기본 모델을 만듭니다. 채점기는 이 함수를 부르지 않습니다.

    import를 함수 안에 둔 이유: 모듈을 불러오는 것만으로 AWS 자격증명을
    요구하지 않게 하기 위해서입니다.
    """
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


def get_text(message: Any) -> str:
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content if hasattr(message, "content") else message
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════


class TriageResult(BaseModel):
    """클라우드 운영 알람 분류 결과입니다."""

    severity: Literal["P1", "P2", "P3", "P4"] = Field(
        ...,
        description=(
            "장애 등급. "
            "P1: 서비스 전면 중단 또는 다수 고객 영향으로 즉시 대응 필요. "
            "P2: 핵심 기능 일부 장애 또는 성능 저하로 30분 내 대응 필요. "
            "P3: 영향 범위가 제한적이거나 우회가 가능해 업무 시간 내 처리. "
            "P4: 사용자 영향 없는 경고성 알람으로 정기 점검 때 확인."
        ),
    )
    affected_service: str = Field(
        ...,
        description="알람에서 영향을 받는 서비스 이름. 알람에 적힌 서비스명을 그대로 쓴다.",
    )
    category: Literal["availability", "performance", "capacity", "security", "other"] = Field(
        ...,
        description=(
            "증상 분류. availability: 접속 불가나 5xx 같은 가용성 문제, "
            "performance: 지연이나 응답 시간 증가, "
            "capacity: CPU, 메모리, 디스크 같은 자원 부족, "
            "security: 인증 실패나 비정상 접근, other: 위에 해당하지 않음."
        ),
    )
    first_check: str = Field(
        ...,
        description="담당자가 가장 먼저 확인해야 할 항목 한 가지를 한 문장으로 쓴다.",
    )
    owner_team: str = Field(
        ...,
        description="1차 대응을 맡을 담당팀 이름. 예: 클라우드운영팀, 물류플랫폼팀, 보안관제팀.",
    )


TRIAGE_SCHEMA = TriageResult


def build_triage_chain(llm=None):
    """알람 dict를 받아 TRIAGE_SCHEMA 형태의 dict를 반환하는 체인을 만듭니다.

    반드시 지킬 것:
      - 프롬프트에 파서의 format_instructions를 실제로 주입해야 합니다.
        스키마만 정의하고 프롬프트에 넣지 않으면 모델은 형식을 모릅니다.
    """
    llm = llm or _default_llm()

    parser = JsonOutputParser(pydantic_object=TRIAGE_SCHEMA)
    prompt = PromptTemplate(
        template=(
            "너는 삼성SDS 클라우드 운영센터의 알람 트리아지 담당자야. "
            "다음 운영 알람을 분석해서 심각도, 영향 서비스, 증상 분류, "
            "첫 확인 항목, 담당팀을 판단해.\n\n"
            "알람: {alert}\n\n"
            "{format_instructions}\n"
        ),
        input_variables=["alert"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return {"alert": lambda a: str(a)} | prompt | llm | parser


def build_notice_chain(llm=None):
    """알람 dict를 받아 담당팀 안내 초안 문자열을 반환하는 체인을 만듭니다.

    반드시 지킬 것:
      - system 메시지로 역할을 지정합니다 (30자 이상).
      - 기본 예시 문구를 그대로 두지 말고 본인 조직의 맥락으로 바꾸세요.
    """
    llm = llm or _default_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 삼성SDS 클라우드운영센터 상황실 담당자야. 운영 알람을 받으면 "
                "담당팀에 보낼 안내 초안을 정중한 존댓말로 작성해. "
                "알람 내용, 영향 서비스, 먼저 확인할 항목을 담고 세 문장 이내로 써.",
            ),
            ("human", "{alert}"),
        ]
    )
    return {"alert": lambda a: str(a)} | prompt | llm | StrOutputParser()


def build_full_chain(llm=None):
    """두 체인을 병렬로 묶어 {"triage": dict, "notice": str} 를 반환합니다.

    심화 파트를 구현했다면, triage 결과는 apply_severity_rules를 거친 값이어야 합니다.
    """
    llm = llm or _default_llm()

    triage_chain = build_triage_chain(llm)
    notice_chain = build_notice_chain(llm)

    # 모델의 분류 결과에 결정적 규칙을 얹어 최종 triage로 만듭니다.
    triage_with_rules = RunnableLambda(
        lambda alert: apply_severity_rules(triage_chain.invoke(alert), alert)
    )
    return RunnableParallel(triage=triage_with_rules, notice=notice_chain)


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# 장애 시 최우선으로 봐야 하는 서비스입니다.
CRITICAL_SERVICES: set[str] = {
    "payment-api",
    "auth-service",
    "order-gateway",
    "logistics-tracking",
}

# severity 등급의 높낮이 비교용입니다. 숫자가 작을수록 급합니다.
SEVERITY_ORDER = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}


def apply_severity_rules(triage: dict, alert: dict) -> dict:
    """모델이 매긴 severity 위에 결정적 규칙을 얹습니다.

    이 함수는 LLM을 호출하지 않습니다. 같은 입력에 항상 같은 출력을 내야 합니다.

    규칙:
      - alert["service"]가 CRITICAL_SERVICES에 있으면 최소 P2
      - alert["message"]에 "5xx" 또는 "outage"가 있으면 최소 P1
      - 둘 다면 더 높은 쪽(숫자가 작은 쪽)을 적용
      - 아무것도 해당 없으면 모델 판단을 그대로 둠
      - 규칙이 적용됐으면 그 사실을 결과에 남길 것 (예: rule_applied 키)

    Returns:
        규칙이 반영된 새 dict. 원본을 그대로 바꾸지 말고 복사해서 쓰는 편이 안전합니다.
    """
    result = dict(triage)
    model_severity = result.get("severity")
    # 모델이 등급을 비웠거나 이상한 값을 주면 가장 낮은 등급으로 보고 규칙을 얹습니다.
    current = SEVERITY_ORDER.get(model_severity, max(SEVERITY_ORDER.values()))

    message = str(alert.get("message", "")).lower()
    applied: list[str] = []
    floor = current

    if alert.get("service") in CRITICAL_SERVICES:
        applied.append("critical_service")
        floor = min(floor, SEVERITY_ORDER["P2"])
    if "5xx" in message or "outage" in message:
        applied.append("5xx_or_outage")
        floor = min(floor, SEVERITY_ORDER["P1"])

    # 규칙 때문에 등급이 실제로 올라갔을 때만 흔적을 남깁니다.
    if floor < current:
        by_rank = {rank: name for name, rank in SEVERITY_ORDER.items()}
        result["severity"] = by_rank[floor]
        result["model_severity"] = model_severity
        result["rule_applied"] = applied
    return result
