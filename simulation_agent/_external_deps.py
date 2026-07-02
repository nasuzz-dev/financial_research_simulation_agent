"""
C(financial_research_data_agent) 리포를 외부 의존성으로 연결.

이 리포는 C 리포와 합쳐지지 않고 독립적으로 존재한다.
.env의 DATA_AGENT_REPO_PATH가 가리키는 경로를 sys.path에 추가해서,
`functions.get_agent_context` 같은 C의 모듈을 그대로 import해서 쓴다.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_PATH = "../financial_research_data_agent"
_repo_path = Path(os.environ.get("DATA_AGENT_REPO_PATH", _DEFAULT_PATH)).resolve()

if not _repo_path.exists():
    raise FileNotFoundError(
        f"C(데이터 에이전트) 리포를 찾을 수 없습니다: {_repo_path}\n"
        f".env의 DATA_AGENT_REPO_PATH를 확인하거나, 아래 명령으로 클론하세요:\n"
        f"  git clone https://github.com/nasuzz-dev/financial_research_data_agent.git "
        f"{_repo_path}"
    )

if str(_repo_path) not in sys.path:
    sys.path.insert(0, str(_repo_path))

DATA_AGENT_REPO_PATH = _repo_path
