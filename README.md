# financial_research_simulation_agent

개미비서단 프로젝트의 **Agent G (시뮬레이션 에이전트)** 리포입니다.

## 역할
F(토론형 에이전트)로부터 `ticker`, `user_id`, `macro_agenda`, `risk_agenda`를 받아서
LSTM 기반 일반 예측 + (토론에서 리스크 언급 시) What-if 비교를 수행하고,
결과를 프론트엔드로 직접 전송합니다.

## 다른 리포와의 관계

이 리포는 단독으로 실행되지 않고, 아래 두 리포에 **경로로만 의존**합니다.
코드를 합치거나 git submodule로 묶지 않고, 로컬에 클론해둔 경로를 환경변수/인자로
가리키는 방식입니다.

| 리포 | 역할 | 이 리포가 가져다 쓰는 것 |
|---|---|---|
| `nasuzz-dev/financial_research_data_agent` (C, 본인) | 전처리/Vector DB/공통 함수 | `functions.get_agent_context`, `interfaces`, `storage.sqlite_db` (코드) |
| `boogiewooki02/financial-research-agent` (B) | 데이터 수집/크롤러 | `db/reports.db` (실제 데이터 파일, 코드는 안 씀) |

## 초기 설정

1. 이 리포와 같은 위치에 두 리포를 클론합니다 (아무 위치나 가능, 경로만 맞으면 됨).

   ```bash
   git clone https://github.com/nasuzz-dev/financial_research_data_agent.git ../financial_research_data_agent
   git clone https://github.com/boogiewooki02/financial-research-agent.git ../financial-research-agent
   ```

2. `.env` 파일을 만들고 두 경로를 지정합니다 (`.env.example` 참고).

   ```
   DATA_AGENT_REPO_PATH=../financial_research_data_agent
   B_DB_PATH=../financial-research-agent/db/reports.db
   ```

3. 의존성 설치

   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

4. 동작 확인

   ```bash
   python -m simulation_agent.preprocessor --ticker 005930 --user-id u1
   ```

## 폴더 구조

```
simulation_agent/
  data_collector.py   # 1단계: B의 DB + C의 함수로 원본 데이터 수집
  preprocessor.py      # 2단계: 로그수익률 계산, LSTM 입력 텐서 구성
  (이후 model.py, monte_carlo.py, risk_classifier.py 등이 추가될 예정)
```
