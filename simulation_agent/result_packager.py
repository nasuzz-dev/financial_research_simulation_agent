"""
Agent G (시뮬레이션 에이전트) - 6단계: 결과 패키징 + risk_profile 톤 조정

5단계(monte_carlo.py)가 만든 Monte Carlo 결과를 받아서,
유저의 risk_profile에 맞게 해석 문구/리스크 카드 톤을 조정하고
프론트엔드로 전송할 최종 JSON을 패키징한다.

risk_profile 톤 매핑 (Judge와 동일한 기준):
  conservative / moderate_conservative → 리스크 중심, 보수적 톤
  moderate                             → 균형 잡힌 중립 톤
  aggressive / very_aggressive         → 기회 중심, 적극적 톤
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# risk_profile → 톤 매핑
RISK_PROFILE_TONE = {
    "conservative": "conservative",
    "moderate_conservative": "conservative",
    "moderate": "moderate",
    "aggressive": "aggressive",
    "very_aggressive": "aggressive",
}

# 톤별 해석 문구 템플릿
TONE_TEMPLATES = {
    "conservative": {
        "outlook_positive": "일부 상승 가능성이 있으나, 변동성 리스크에 주의가 필요합니다.",
        "outlook_neutral": "수익률이 제한적일 수 있으며, 원금 보전에 유의하시기 바랍니다.",
        "outlook_negative": "하락 리스크가 크게 나타나고 있어 신중한 접근이 필요합니다.",
        "what_if_amplified": "해당 리스크 요인이 실현될 경우 손실 폭이 더 커질 수 있습니다.",
        "what_if_dampened": "해당 리스크 요인의 영향은 제한적으로 나타났습니다.",
    },
    "moderate": {
        "outlook_positive": "상승 가능성이 있으나, 변동성을 감안한 균형 잡힌 접근이 필요합니다.",
        "outlook_neutral": "중립적인 수익률이 예상되며, 시장 상황을 지켜볼 필요가 있습니다.",
        "outlook_negative": "하락 가능성이 있어 포트폴리오 내 비중 조정을 고려해볼 수 있습니다.",
        "what_if_amplified": "해당 리스크 요인 발생 시 수익률에 부정적 영향이 예상됩니다.",
        "what_if_dampened": "해당 리스크 요인의 영향은 크지 않은 것으로 나타났습니다.",
    },
    "aggressive": {
        "outlook_positive": "상승 모멘텀이 나타나고 있어 적극적인 투자 기회로 볼 수 있습니다.",
        "outlook_neutral": "뚜렷한 방향성은 없으나, 변동성을 기회로 활용할 수 있는 구간입니다.",
        "outlook_negative": "단기 조정 가능성이 있으나, 장기 관점에서 저점 매수 기회가 될 수 있습니다.",
        "what_if_amplified": "해당 리스크 요인이 실현되더라도 장기 관점에서 회복 가능성이 있습니다.",
        "what_if_dampened": "해당 리스크 요인의 영향이 제한적이어서 투자 매력도가 유지됩니다.",
    },
}


def _get_outlook_key(expected_return_pct: float, upside_probability: float) -> str:
    """예측 결과를 기반으로 outlook 키 결정"""
    if expected_return_pct > 2.0 and upside_probability > 0.55:
        return "outlook_positive"
    elif expected_return_pct < -2.0 or upside_probability < 0.45:
        return "outlook_negative"
    else:
        return "outlook_neutral"


def _build_interpretation(
    mc_result: dict,
    tone: str,
    ticker: str,
) -> str:
    """Monte Carlo 결과 기반 해석 문구 생성"""
    base = mc_result["base"]
    templates = TONE_TEMPLATES[tone]
    outlook_key = _get_outlook_key(
        base["expected_return_pct"],
        base["upside_probability"],
    )
    lines = [
        f"향후 30일 예상 평균 수익률은 {base['expected_return_pct']:.1f}%이며, "
        f"상승 확률은 {base['upside_probability'] * 100:.0f}%입니다.",
        templates[outlook_key],
    ]

    # What-if 결과가 있으면 추가
    for what_if in mc_result.get("what_if", []):
        impact = what_if["impact_pct"]
        variable_name = {
            "BASE_RATE_KR_up": "기준금리 인상",
            "BASE_RATE_KR_down": "기준금리 인하",
            "CPI_KR_up": "물가 상승",
            "CPI_KR_down": "물가 하락",
            "KTB_3Y_KR_up": "국고채 3년물 금리 상승",
            "KTB_3Y_KR_down": "국고채 3년물 금리 하락",
            "KTB_10Y_KR_up": "국고채 10년물 금리 상승",
            "KTB_10Y_KR_down": "국고채 10년물 금리 하락",
            "USD_KRW_up": "환율 상승",
            "USD_KRW_down": "환율 하락",
        }.get(f"{what_if['variable']}_{what_if['direction']}", what_if["variable"])

        if impact < -1.0:
            lines.append(
                f"[{variable_name} 시나리오] {templates['what_if_amplified']} "
                f"(예상 수익률 변화: {impact:+.1f}%)"
            )
        else:
            lines.append(
                f"[{variable_name} 시나리오] {templates['what_if_dampened']} "
                f"(예상 수익률 변화: {impact:+.1f}%)"
            )

    return " ".join(lines)


def _build_risk_card(
    mc_result: dict,
    tone: str,
    risk_factors: list,
) -> dict:
    """리스크 카드 생성"""
    base = mc_result["base"]
    pessimistic = base["scenarios"]["pessimistic"]

    # 정량화 가능한 리스크 (What-if 돌린 것들)
    quantified_risks = [
        {
            "variable": w["variable"],
            "direction": w["direction"],
            "impact_pct": w["impact_pct"],
        }
        for w in mc_result.get("what_if", [])
    ]

    # 정성적 리스크 (분류됐지만 정량화 못 한 것들 — 현재는 없음, 향후 확장 가능)
    qualitative_risks = []

    return {
        "volatility": round(base["volatility"] * 100, 2),
        "pessimistic_return_pct": round(pessimistic["expected_return_pct"], 2),
        "quantified_risks": quantified_risks,
        "qualitative_risks": qualitative_risks,
        "tone": tone,
    }


def package_result(
    ticker: str,
    mc_result: dict,
    user_context: dict,
    risk_factors: list,
    current_price: float,
) -> dict:
    """
    6단계 메인 함수: Monte Carlo 결과를 프론트 전송용 JSON으로 패키징.

    Args:
        ticker: 종목코드
        mc_result: 5단계 run_monte_carlo() 결과
        user_context: get_user_context() 결과 (risk_profile 등)
        risk_factors: 3단계 classify_risk_factors() 결과
        current_price: 현재 종가

    Returns:
        프론트엔드로 전송할 최종 JSON
    """
    risk_profile = user_context.get("risk_profile", "moderate")
    tone = RISK_PROFILE_TONE.get(risk_profile, "moderate")

    base = mc_result["base"]

    return {
        "ticker": ticker,
        "current_price": current_price,
        "simulation_type": "what_if" if mc_result.get("what_if") else "general",
        "summary": {
            "expected_return_pct": round(base["expected_return_pct"], 2),
            "upside_probability": round(base["upside_probability"] * 100, 1),
            "volatility": round(base["volatility"] * 100, 2),
        },
        "scenarios": {
            "optimistic": base["scenarios"]["optimistic"],
            "neutral": base["scenarios"]["neutral"],
            "pessimistic": base["scenarios"]["pessimistic"],
        },
        "chart_data": {
            "base": base["percentile_paths"],
            "what_if": [
                {
                    "variable": w["variable"],
                    "direction": w["direction"],
                    "paths": w["result"]["percentile_paths"],
                }
                for w in mc_result.get("what_if", [])
            ],
        },
        "interpretation": _build_interpretation(mc_result, tone, ticker),
        "risk_card": _build_risk_card(mc_result, tone, risk_factors),
        "user_profile": {
            "risk_profile": risk_profile,
            "investment_goal": user_context.get("investment_goal"),
        },
    }


if __name__ == "__main__":
    import argparse
    import json

    from simulation_agent.data_collector import collect_simulation_inputs, DEFAULT_B_DB_PATH
    from simulation_agent.preprocessor import build_feature_table
    from simulation_agent.model import get_or_train_model, predict_distribution, apply_shock
    from simulation_agent.risk_classifier import classify_risk_factors
    from simulation_agent.monte_carlo import run_monte_carlo, N_PATHS

    parser = argparse.ArgumentParser(description="Agent G 결과 패키징 테스트")
    parser.add_argument("--ticker", default="005930")
    parser.add_argument("--user-id", default="u1")
    parser.add_argument("--db-path", default=DEFAULT_B_DB_PATH)
    parser.add_argument("--sequence-length", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    raw = collect_simulation_inputs(ticker=args.ticker, user_id=args.user_id, db_path=args.db_path)
    table = build_feature_table(raw["price_data"], raw["macro_data"])
    current_price = float(raw["price_data"]["latest"]["current_price"])

    macro_agenda = {
        "bull_summary": "환율 상승이 수출에 긍정적",
        "bull_arguments": "원달러 환율 상승으로 수출 채산성 개선",
        "bear_summary": "금리 인상 우려",
        "bear_arguments": "기준금리 추가 인상 시 투자심리 위축 우려",
    }
    risk_factors = classify_risk_factors(macro_agenda=macro_agenda)

    model = get_or_train_model(
        ticker=args.ticker, feature_table=table,
        sequence_length=args.sequence_length, horizon=args.horizon,
        epochs=args.epochs, force_retrain=True,
    )
    base_pred = predict_distribution(model, table, sequence_length=args.sequence_length)

    shock_preds = []
    for rf in risk_factors:
        shocked = apply_shock(table, rf["variable"], rf["direction"])
        shock_pred = predict_distribution(model, shocked, args.sequence_length)
        shock_preds.append({"variable": rf["variable"], "direction": rf["direction"], "prediction": shock_pred})

    mc_result = run_monte_carlo(
        base_prediction=base_pred,
        current_price=current_price,
        shock_predictions=shock_preds if shock_preds else None,
        horizon=args.horizon,
    )

    final = package_result(
        ticker=args.ticker,
        mc_result=mc_result,
        user_context=raw["user_context"],
        risk_factors=risk_factors,
        current_price=current_price,
    )

    # chart_data는 길어서 제외하고 출력
    output = {k: v for k, v in final.items() if k != "chart_data"}
    print(json.dumps(output, ensure_ascii=False, indent=2))
