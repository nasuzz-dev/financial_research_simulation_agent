"""
Agent G (시뮬레이션 에이전트) - 3단계: 리스크 요인 분류

F로부터 받은 macro_agenda, risk_agenda(Bull 주장 + Bear 반박 텍스트)를 LLM에 넣어서,
G가 What-if 시뮬레이션을 돌릴 수 있는 매크로 변수(금리/CPI/국채금리/환율) 중
실제로 리스크로 언급된 게 있는지 분류한다.

- 매칭되는 게 있으면: [{"variable": "BASE_RATE_KR", "direction": "up"}, ...]
- 없으면: []  → G는 일반 예측만 수행 (4단계에서 폴백)

LLM 호출: Upstage의 OpenAI 호환 엔드포인트 사용 (.env의 UPSTAGE_API_KEY 재사용).
"""

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from simulation_agent.preprocessor import MACRO_INDICATORS

load_dotenv()
logger = logging.getLogger(__name__)

UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"
# 분류처럼 가벼운 작업이라 가벼운 모델 사용. 실제 사용 가능한 모델명은 Upstage 문서에서 확인 필요.
CLASSIFIER_MODEL = os.environ.get("RISK_CLASSIFIER_MODEL", "solar-pro")

_SYSTEM_PROMPT = f"""당신은 금융 토론 텍스트에서 정량적으로 시뮬레이션 가능한 \
매크로 리스크 요인을 추출하는 분류기입니다.

다룰 수 있는 변수는 정확히 이 5개뿐입니다: {", ".join(MACRO_INDICATORS)}
- BASE_RATE_KR: 한국 기준금리
- CPI_KR: 한국 소비자물가지수
- KTB_3Y_KR: 한국 국고채 3년물 금리
- KTB_10Y_KR: 한국 국고채 10년물 금리
- USD_KRW: 원/달러 환율

입력으로 주어지는 토론 텍스트(Bull 주장 + Bear 반박)를 읽고, 위 5개 변수 중 \
"리스크 요인"으로 명시적으로 언급된 것이 있으면 추출하세요. \
유가, 경쟁사 동향, 신제품, 규제 등 위 5개에 해당하지 않는 내용은 무시하세요.

반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트는 절대 포함하지 마세요.

{{
  "risk_factors": [
    {{"variable": "BASE_RATE_KR", "direction": "up"}}
  ]
}}

언급된 게 전혀 없으면:
{{"risk_factors": []}}

direction은 "up" 또는 "down"만 가능합니다. 방향이 불명확하면 그 항목은 제외하세요.
"""


class RiskClassificationError(Exception):
    pass


def _build_user_prompt(macro_agenda: Optional[dict], risk_agenda: Optional[dict]) -> str:
    parts = []
    if macro_agenda:
        parts.append(
            "[아젠다 2: 산업 및 매크로 환경]\n"
            f"Bull 주장: {macro_agenda.get('bull_claim', '')}\n"
            f"Bear 반박: {macro_agenda.get('bear_rebuttal', '')}"
        )
    if risk_agenda:
        parts.append(
            "[아젠다 3: 리스크 요인]\n"
            f"Bull 주장: {risk_agenda.get('bull_claim', '')}\n"
            f"Bear 반박: {risk_agenda.get('bear_rebuttal', '')}"
        )
    return "\n\n".join(parts) if parts else "(토론 텍스트 없음)"


def classify_risk_factors(
    macro_agenda: Optional[dict] = None,
    risk_agenda: Optional[dict] = None,
    client: Optional[OpenAI] = None,
) -> list:
    """
    Returns:
        [{"variable": "BASE_RATE_KR", "direction": "up"}, ...] 또는 []

    macro_agenda, risk_agenda가 둘 다 없으면 LLM 호출 없이 바로 [] 반환.
    """
    if not macro_agenda and not risk_agenda:
        logger.info("macro_agenda/risk_agenda가 없어 리스크 분류를 건너뜁니다.")
        return []

    if client is None:
        api_key = os.environ.get("UPSTAGE_API_KEY")
        if not api_key:
            raise RiskClassificationError(
                "UPSTAGE_API_KEY가 설정되어 있지 않습니다 (.env 확인 필요)."
            )
        client = OpenAI(api_key=api_key, base_url=UPSTAGE_BASE_URL)

    user_prompt = _build_user_prompt(macro_agenda, risk_agenda)

    try:
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("리스크 분류 LLM 호출 실패: %s", e)
        # LLM 호출 자체가 실패해도 전체 파이프라인은 안 죽고 일반 예측으로 폴백
        return []

    # 모델이 ```json ... ``` 형태로 감싸서 줄 수 있어 방어적으로 처리
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("리스크 분류 결과를 JSON으로 파싱하지 못했습니다: %s", raw_text)
        return []

    risk_factors = parsed.get("risk_factors", [])

    # 허용된 변수/방향 외의 값은 안전을 위해 필터링
    validated = [
        rf for rf in risk_factors
        if rf.get("variable") in MACRO_INDICATORS and rf.get("direction") in ("up", "down")
    ]

    if len(validated) != len(risk_factors):
        logger.warning("일부 리스크 요인이 허용 범위를 벗어나 제외되었습니다: %s", risk_factors)

    return validated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent G 리스크 분류 레이어 테스트")
    parser.add_argument(
        "--macro-bull", default="환율 상승이 수출 채산성에 긍정적입니다."
    )
    parser.add_argument(
        "--macro-bear", default="다만 기준금리가 추가 인상될 가능성이 있어 투자심리가 위축될 수 있습니다."
    )
    parser.add_argument("--risk-bull", default="실적 모멘텀이 견고합니다.")
    parser.add_argument("--risk-bear", default="경쟁사 점유율 확대가 우려됩니다.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result = classify_risk_factors(
        macro_agenda={"bull_claim": args.macro_bull, "bear_rebuttal": args.macro_bear},
        risk_agenda={"bull_claim": args.risk_bull, "bear_rebuttal": args.risk_bear},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
