"""
Agent G (시뮬레이션 에이전트) - 2단계: 전처리

1단계(data_collector.collect_simulation_inputs)가 반환한 원본 데이터를 받아서,
LSTM 입력으로 쓸 수 있는 시계열 피처 행렬로 변환한다.

처리 내용:
  - price_data: 날짜순 정렬, 로그수익률 계산 (close 기반)
  - macro_data: indicator별 시계열을 날짜 기준으로 정렬 + 가격 데이터 날짜에 맞춰
                forward-fill(직전 값 유지)로 결측치 처리
  - 최종적으로 [date, log_return, volume, volatility_30d, BASE_RATE_KR, CPI_KR,
                KTB_3Y_KR, KTB_10Y_KR, USD_KRW] 형태의 피처 테이블(DataFrame) 생성
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# get_agent_context(agent_type="simulation")로 받을 수 있는 매크로 지표 5개.
# (KOSPI/KOSDAQ 지수, 국제유가는 B 스키마에 없어 제외 — 추후 추가되면 여기에 더함)
MACRO_INDICATORS = ["BASE_RATE_KR", "CPI_KR", "KTB_3Y_KR", "KTB_10Y_KR", "USD_KRW"]


class InsufficientDataError(Exception):
    """LSTM 학습/예측에 쓸 데이터가 너무 적을 때 발생"""
    pass


def _price_data_to_df(price_data: dict) -> pd.DataFrame:
    """get_price_data()/get_agent_context()의 price_data를 DataFrame으로 변환 + 로그수익률 계산"""
    prices = price_data.get("prices") or []
    if not prices:
        raise InsufficientDataError("price_data가 비어 있습니다 (해당 ticker의 가격 데이터 없음).")

    df = pd.DataFrame(prices)
    df["price_date"] = pd.to_datetime(df["price_date"])
    df = df.sort_values("price_date").reset_index(drop=True)

    # 숫자형 컬럼 강제 변환 (DB에서 None/문자열로 들어올 수 있음)
    for col in ["close", "volume", "volatility_30d"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 로그수익률: ln(close_t / close_t-1)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # volatility_30d가 비어 있으면 직접 계산한 30일 롤링 변동성으로 대체
    if "volatility_30d" not in df.columns or df["volatility_30d"].isna().all():
        df["volatility_30d"] = df["log_return"].rolling(window=30, min_periods=5).std()

    return df[["price_date", "close", "volume", "volatility_30d", "log_return"]]


def _macro_data_to_df(macro_data: dict) -> pd.DataFrame:
    """get_macro_data()/get_agent_context()의 macro_data(지표별 records)를
    날짜를 인덱스로 하는 wide-format DataFrame으로 변환"""
    indicators = macro_data.get("indicators") or []

    if not indicators:
        logger.warning("macro_data가 비어 있습니다. 매크로 피처는 전부 NaN으로 채워집니다.")
        return pd.DataFrame(columns=["date"] + MACRO_INDICATORS)

    frames = []
    for ind in indicators:
        ind_id = ind.get("indicator_id")
        if ind_id not in MACRO_INDICATORS:
            continue
        records = ind.get("records") or []
        if not records:
            continue
        s = pd.DataFrame(records)[["date", "value"]].rename(columns={"value": ind_id})
        s["date"] = pd.to_datetime(s["date"])
        frames.append(s.set_index("date"))

    if not frames:
        return pd.DataFrame(columns=["date"] + MACRO_INDICATORS)

    wide = pd.concat(frames, axis=1).sort_index()
    wide = wide.reset_index().rename(columns={"index": "date"})
    return wide


def build_feature_table(
    price_data: dict,
    macro_data: dict,
    min_rows: int = 20,
) -> pd.DataFrame:
    """
    1단계 원본 데이터를 받아 LSTM 입력용 피처 테이블(DataFrame)을 만든다.

    Returns:
        DataFrame, columns = [
            "date", "close", "volume", "volatility_30d", "log_return",
            "BASE_RATE_KR", "CPI_KR", "KTB_3Y_KR", "KTB_10Y_KR", "USD_KRW"
        ]
        결측 매크로 값은 forward-fill 후, 남은 결측은 0으로 채움.
    """
    price_df = _price_data_to_df(price_data)
    macro_df = _macro_data_to_df(macro_data)

    merged = price_df.rename(columns={"price_date": "date"})

    if not macro_df.empty:
        merged = pd.merge_asof(
            merged.sort_values("date"),
            macro_df.sort_values("date"),
            on="date",
            direction="backward",  # 가격 시점 기준, 그 직전(또는 같은날) 매크로 값 사용
        )
    else:
        for col in MACRO_INDICATORS:
            merged[col] = np.nan

    # forward-fill 후 남은 결측치는 0으로 (모델 입력에 NaN이 들어가지 않도록)
    merged[MACRO_INDICATORS] = merged[MACRO_INDICATORS].ffill().fillna(0)

    # 첫 행은 log_return이 NaN(직전 종가 없음)이므로 제거
    merged = merged.dropna(subset=["log_return"]).reset_index(drop=True)

    if len(merged) < min_rows:
        raise InsufficientDataError(
            f"전처리 후 데이터가 {len(merged)}행밖에 없습니다 (최소 {min_rows}행 필요). "
            f"해당 ticker의 가격 데이터가 더 필요합니다."
        )

    return merged


def to_lstm_input(
    feature_table: pd.DataFrame,
    sequence_length: int = 60,
    feature_cols: Optional[list] = None,
) -> np.ndarray:
    """
    피처 테이블을 LSTM 입력 형태인 (sequence_length, n_features) 윈도우로 변환.
    가장 최근 sequence_length일 만큼만 사용 (예측 시점 기준 입력 시퀀스 1개).

    Returns:
        np.ndarray, shape = (sequence_length, n_features)
    """
    if feature_cols is None:
        feature_cols = ["log_return", "volume", "volatility_30d"] + MACRO_INDICATORS

    if len(feature_table) < sequence_length:
        raise InsufficientDataError(
            f"시퀀스 길이({sequence_length}) 확보를 위한 데이터가 부족합니다 "
            f"(현재 {len(feature_table)}행)."
        )

    window = feature_table.tail(sequence_length)
    arr = window[feature_cols].to_numpy(dtype=np.float32)

    # 표준화 (이번 윈도우 기준 z-score). 모델 학습 시에는 학습 데이터 전체 기준
    # scaler를 따로 저장해서 재사용해야 함 — 여기서는 1차 버전이라 윈도우 단위로 처리.
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # 분산 0인 컬럼(상수) 나눗셈 에러 방지
    arr = (arr - mean) / std

    return arr


if __name__ == "__main__":
    import argparse
    import logging as _logging

    from simulation_agent.data_collector import collect_simulation_inputs, DEFAULT_B_DB_PATH

    parser = argparse.ArgumentParser(description="Agent G 전처리 레이어 테스트")
    parser.add_argument("--ticker", default="005930")
    parser.add_argument("--user-id", default="u1")
    parser.add_argument("--db-path", default=DEFAULT_B_DB_PATH)
    parser.add_argument("--sequence-length", type=int, default=60)
    args = parser.parse_args()

    _logging.basicConfig(level=_logging.INFO)

    raw = collect_simulation_inputs(
        ticker=args.ticker, user_id=args.user_id, db_path=args.db_path
    )
    table = build_feature_table(raw["price_data"], raw["macro_data"])
    print("피처 테이블 shape:", table.shape)
    print(table.tail())

    lstm_input = to_lstm_input(table, sequence_length=args.sequence_length)
    print("LSTM 입력 shape:", lstm_input.shape)