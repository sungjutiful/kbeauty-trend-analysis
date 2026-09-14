"""
K-뷰티 해외(미국) 검색 트렌드 분석
Codyssey AI 데이터 분석 미션 - 시계열 데이터 기반 트렌드 분석

데이터 출처: Google Trends (trends.google.com)
- 키워드: PDRN, Ectoin, Exosome
- 지역: 미국 (United States)
- 기간: 2021-09-12 ~ 2026-09-13 (5년, 주간 데이터)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = "data"
IMG_DIR = "images"


def load_trend_csv(path: str, keyword: str) -> pd.DataFrame:
    """Google Trends 'multiTimeline.csv' 형식을 읽어 date/value 데이터프레임으로 변환한다."""
    df = pd.read_csv(path, skiprows=2)
    df.columns = ["date", keyword]
    df["date"] = pd.to_datetime(df["date"])
    # Google Trends는 검색량이 1 미만인 주를 '<1' 문자열로 표기한다.
    # 결측이 아니라 '0에 가까운 실제 값'이므로 0으로 치환한다(진짜 결측과 구분).
    df[keyword] = df[keyword].astype(str).str.replace("<1", "0", regex=False)
    df[keyword] = pd.to_numeric(df[keyword], errors="coerce")
    return df


def load_and_merge() -> pd.DataFrame:
    pdrn = load_trend_csv(f"{DATA_DIR}/pdrn_raw.csv", "PDRN")
    ectoin = load_trend_csv(f"{DATA_DIR}/ectoin_raw.csv", "Ectoin")
    exosome = load_trend_csv(f"{DATA_DIR}/exosome_raw.csv", "Exosome")

    merged = pdrn.merge(ectoin, on="date", how="outer").merge(exosome, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 확인 및 처리.
    - '<1' 표기는 load_trend_csv 단계에서 이미 0으로 치환했으므로 결측이 아니다.
    - 여기서 남는 결측은 Google Trends가 데이터를 아예 제공하지 않은 진짜 결측치이며,
      선형보간으로 처리한다."""
    missing_before = df.isna().sum().sum()
    if missing_before > 0:
        df = df.set_index("date").interpolate(method="linear").reset_index()
    return df, missing_before


def add_time_series_features(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """시계열 기법 적용: (1) 4주 이동평균, (2) 전주 대비 변화율."""
    for kw in keywords:
        df[f"{kw}_MA4"] = df[kw].rolling(window=4, min_periods=1).mean()
        df[f"{kw}_WoW_pct"] = df[kw].pct_change().replace([float("inf"), float("-inf")], None) * 100
    return df


def find_surge_start(df: pd.DataFrame, keyword: str, sustain_weeks: int = 8) -> tuple:
    """질문 A: 급상승 시작 시점 탐지.
    베이스라인(첫 52주 평균)의 3배를 넘고, 이후 sustain_weeks 연속으로
    베이스라인 이상을 유지하는 첫 시점을 '상승 시작'으로 정의한다.
    """
    ma_col = f"{keyword}_MA4"
    baseline = df[keyword].iloc[:52].mean()
    threshold = max(baseline * 3, baseline + 10)

    above = df[ma_col] >= threshold
    for i in range(len(df) - sustain_weeks):
        if above.iloc[i:i + sustain_weeks].all():
            return df["date"].iloc[i], baseline, threshold
    return None, baseline, threshold


def find_decay(df: pd.DataFrame, keyword: str) -> dict:
    """질문 D: 피크 이후 관심도가 절반으로 꺾이기까지 걸린 기간(바이럴 감쇠)."""
    peak_idx = df[keyword].idxmax()
    peak_date = df["date"].iloc[peak_idx]
    peak_val = df[keyword].iloc[peak_idx]
    half = peak_val / 2

    after = df.iloc[peak_idx:]
    decay_row = after[after[keyword] <= half]
    if decay_row.empty:
        return {"peak_date": peak_date, "peak_val": peak_val, "weeks_to_half": None}

    decay_date = decay_row["date"].iloc[0]
    weeks = (decay_date - peak_date).days // 7
    return {"peak_date": peak_date, "peak_val": peak_val,
            "decay_date": decay_date, "weeks_to_half": weeks}


def main():
    df = load_and_merge()
    df, missing_before = clean_data(df)
    keywords = ["PDRN", "Ectoin", "Exosome"]
    df = add_time_series_features(df, keywords)

    print(f"데이터 기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"데이터 포인트 수: {len(df)}개 (주간)")
    print(f"결측치(처리 전): {missing_before}개")
    print()

    # ---- 질문 A: PDRN 급상승 시작 시점 ----
    surge_date, baseline, threshold = find_surge_start(df, "PDRN")
    print("[질문 A] PDRN 급상승 시작 시점")
    print(f"  베이스라인(첫 1년 평균): {baseline:.1f}")
    print(f"  임계값: {threshold:.1f}")
    print(f"  상승 시작(8주 연속 유지 기준): {surge_date.date() if surge_date is not None else '탐지 안 됨'}")
    print()

    # ---- 질문 D: PDRN 바이럴 감쇠 ----
    decay = find_decay(df, "PDRN")
    print("[질문 D] PDRN 피크 이후 감쇠")
    print(f"  피크 시점: {decay['peak_date'].date()}, 피크 값: {decay['peak_val']:.0f}")
    if decay["weeks_to_half"] is not None:
        print(f"  피크 대비 절반으로 떨어지기까지: {decay['weeks_to_half']}주 ({decay['decay_date'].date()})")
    print()

    # ---- 질문 E: 세 키워드 간 상관관계 ----
    corr = df[keywords].corr()
    print("[질문 E] 키워드 간 상관관계 (Pearson)")
    print(corr.round(2))
    print()

    # 각 키워드 피크 시점 (상관관계 해석 보조용)
    peak_dates = {kw: df.loc[df[kw].idxmax(), "date"].date() for kw in keywords}
    print("키워드별 피크 시점:", peak_dates)

    # ---- 시각화 1: 세 키워드 원자료 추이 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for kw in keywords:
        ax.plot(df["date"], df[kw], label=kw, linewidth=1.5)
    ax.set_title("K-Beauty Ingredient Search Interest (US, Google Trends)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Search Interest (0-100, normalized per keyword)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/01_search_interest_trend.png", dpi=150)
    plt.close(fig)

    # ---- 시각화 2: PDRN 급상승 시점 (원자료 + 4주 이동평균 + 임계값 + 상승 시작점) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["PDRN"], label="PDRN (weekly)", alpha=0.4, linewidth=1)
    ax.plot(df["date"], df["PDRN_MA4"], label="4-week moving average", linewidth=2, color="tab:red")
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label=f"Surge threshold ({threshold:.0f})")
    if surge_date is not None:
        ax.axvline(surge_date, color="green", linestyle=":", linewidth=1.5, label=f"Surge start ({surge_date.date()})")
    ax.set_title("PDRN Search Interest: Surge Detection")
    ax.set_xlabel("Date")
    ax.set_ylabel("Search Interest")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/02_pdrn_surge_detection.png", dpi=150)
    plt.close(fig)

    # ---- 시각화 3: PDRN 피크 전후 확대 (바이럴 감쇠) ----
    peak_idx = df["PDRN"].idxmax()
    window = df.iloc[max(0, peak_idx - 8):peak_idx + 20]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(window["date"], window["PDRN"], marker="o", markersize=3, linewidth=1.5, color="tab:red")
    ax.axhline(decay["peak_val"], color="gray", linestyle="--", linewidth=1, label=f"Peak ({decay['peak_val']:.0f})")
    ax.axhline(decay["peak_val"] / 2, color="gray", linestyle=":", linewidth=1, label="Half of peak")
    ax.axvline(decay["peak_date"], color="green", linestyle=":", linewidth=1.5, label="Peak week")
    if decay["weeks_to_half"] is not None:
        ax.axvline(decay["decay_date"], color="orange", linestyle=":", linewidth=1.5,
                    label=f"Decayed to half (+{decay['weeks_to_half']}wk)")
    ax.set_title("PDRN: Virality Decay After Peak")
    ax.set_xlabel("Date")
    ax.set_ylabel("Search Interest")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/03_pdrn_decay.png", dpi=150)
    plt.close(fig)

    # ---- 시각화 4: 상관관계 히트맵 ----
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(keywords)))
    ax.set_yticks(range(len(keywords)))
    ax.set_xticklabels(keywords)
    ax.set_yticklabels(keywords)
    for i in range(len(keywords)):
        for j in range(len(keywords)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("Correlation Between Keywords")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/04_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    df.to_csv(f"{DATA_DIR}/merged_processed.csv", index=False)
    print("\n분석 완료. images/ 폴더에 시각화 4개 저장됨. data/merged_processed.csv에 가공 데이터 저장됨.")


if __name__ == "__main__":
    main()
