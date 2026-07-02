"""
Agent G (시뮬레이션 에이전트) - 1단계: 데이터 수집 레이어

- 함수: C(financial_research_data_agent) 리포의 functions.get_agent_context 사용
  → _external_deps에서 sys.path에 C 리포 경로를 추가해두므로 그대로 import 가능
- 데이터: B(financial-research-agent) 리포를 클론해서 생성된 db/reports.db를
  .env의 B_DB_PATH 또는 --db-path 인자로 가리켜서 사용
- get_user_context(user_id): 아직 다른 담당자가 구현 전이라 목업으로 처리.
  실제 함수가 공유되면 이 파일의 import만 교체하면 됨.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

from simulation_agent import _external_deps  # noqa: F401  (sys.path 설정용 — import 순서 중요)

from interfaces import BaseRelationalDB
from storage.sqlite_db import SQLiteDB
from functions.get_agent_context import get_agent_context

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_B_DB_PATH = os.environ.get(
    "B_DB_PATH", "../financial-research-agent/db/reports.db"
)


# ──────────────────────────────────────────────────────────────
# get_user_context 목업 (실제 함수 공유되면 이 함수만 교체)
# ──────────────────────────────────────────────────────────────
def _mock_get_user_context(user_id: str) -> dict:
    logger.warning(
        "get_user_context()가 아직 실제 구현과 연결되지 않았습니다. "
        "목업 데이터를 반환합니다. (user_id=%s)", user_id
    )
    return {
        "risk_profile": "moderate",
        "investment_goal": "mid_term",
        "investment_amount_range": "500_2000",
        "investment_experience": "intermediate",
        "interest_sectors": [],
        "onboarding_done": True,
    }


# 실제 get_user_context가 준비되면 이 줄만 교체:
# from <어딘가> import get_user_context
get_user_context = _mock_get_user_context


# ──────────────────────────────────────────────────────────────
# G의 데이터 수집 진입점
# ──────────────────────────────────────────────────────────────
def collect_simulation_inputs(
    ticker: str,
    user_id: str,
    relational_db: Optional[BaseRelationalDB] = None,
    db_path: str = DEFAULT_B_DB_PATH,
) -> dict:
    """
    G 파이프라인 1단계: LSTM/Monte Carlo에 필요한 원본 데이터를 모두 수집.

    Args:
        ticker: 종목코드 (예: "005930")
        user_id: 유저 ID
        relational_db: 이미 연결된 DB 객체가 있으면 직접 전달 (선택)
        db_path: relational_db를 안 넘기면, 이 경로의 SQLite DB(B 리포에서
                 생성된 reports.db)에 새로 연결함. .env의 B_DB_PATH 또는
                 --db-path로 덮어쓰기 가능.

    Returns:
        {
            "ticker": str,
            "price_data": dict,
            "macro_data": dict,
            "target_prices": dict,
            "user_context": dict,
        }
    """
    if relational_db is None:
        relational_db = SQLiteDB(db_path=db_path)

    agent_context = get_agent_context(
        ticker=ticker,
        agent_type="simulation",
        relational_db=relational_db,
    )

    if "error" in agent_context:
        raise ValueError(f"get_agent_context 실패: {agent_context['error']}")

    user_context = get_user_context(user_id)

    return {
        "ticker": ticker,
        "price_data": agent_context.get("price_data"),
        "macro_data": agent_context.get("macro_data"),
        "target_prices": agent_context.get("target_prices"),
        "user_context": user_context,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Agent G 데이터 수집 레이어 테스트")
    parser.add_argument("--ticker", default="005930")
    parser.add_argument("--user-id", default="u1")
    parser.add_argument("--db-path", default=DEFAULT_B_DB_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = collect_simulation_inputs(
        ticker=args.ticker, user_id=args.user_id, db_path=args.db_path
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
