# ============================================
# HFM Report Dashboard - app.py (Integrated)
# ============================================
# 변경 포인트(요청 반영):
# 0) '타원형' UI 최소화: tag(칩)/탭/라디오 radius 축소
# 1) Preview 중복 텍스트 제거: 업로드 전/후 Preview 렌더 1회로 정리
# 2) Overall 스코어카드 월 인식 오류 + 선택시 미변경 이슈 해결:
#    - robust date parsing
#    - tabs -> radio(CTA)로 변경
# 3) 필터 칩: 색 #858fbb 유지, 텍스트 black + bold 해제
# 4) Trend Summary Table: performance comparison과 동일한 HTML table 스타일 적용
# 5) What drive / Performance Comparison: 중복 소제목 제거
# ============================================

import calendar
import re
from dataclasses import dataclass
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================
# SECTION 0) THEME / CSS
# ============================================

BG = "#e6f4f1"
POINT = "#002873"
SUB1 = "#82a1ff"
SUB2 = "#238783"
FILTER_TAG_BG = "#858fbb"  # 요청 색상

st.set_page_config(page_title="HFM Report Dashboard", layout="wide")

st.markdown(
    f"""
    <style>
      .stApp {{
        background: {BG};
      }}
      html, body, [class*="css"] {{
        color: {POINT};
      }}

      /* ---------- Card Layout ---------- */
      .section-card {{
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 16px;
        padding: 16px 16px 12px 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        margin-bottom: 14px;
      }}
      .section-head {{
        display:flex;
        align-items:center;
        justify-content:space-between;
        margin-bottom: 10px;
      }}
      .section-title {{
        font-size: 18px;
        font-weight: 900;
        color: {POINT};
        margin: 0;
        line-height: 1.1;
      }}
      .section-sub {{
        font-size: 12px;
        font-weight: 600;
        color: rgba(0,40,115,0.65);
        margin-top: 2px;
      }}

      /* ---------- Main Titles ---------- */
      .main-title {{
        font-size: 24px;
        font-weight: 950;
        color: {POINT};
        margin: 10px 0 8px 0;
      }}

      /* ---------- Inputs: dropdown white ---------- */
      div[data-baseweb="select"] > div {{
        background-color: #ffffff !important;
      }}
      div[data-baseweb="select"] {{
        background-color: #ffffff !important;
      }}

      /* ---------- Multiselect tags (칩) : 타원형 느낌 최소화 ---------- */
      span[data-baseweb="tag"] {{
        background-color: {FILTER_TAG_BG} !important;
        color: #000000 !important;            /* 요청: 텍스트 블랙 */
        border-radius: 6px !important;        /* 타원 -> 각진 형태로 */
        font-weight: 400 !important;          /* bold 해제 */
        border: 1px solid rgba(0,0,0,0.10) !important;
      }}
      span[data-baseweb="tag"] svg {{
        fill: #000000 !important;
      }}

      /* ---------- Tabs / Radio pill 느낌 줄이기 ---------- */
      .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        font-weight: 800;
      }}
      .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {POINT};
      }}

      /* Radio를 CTA 버튼처럼(각지게) */
      div[role="radiogroup"] > label {{
        border-radius: 10px !important;
      }}

      /* ---------- Metric Cards ---------- */
      [data-testid="stMetric"] {{
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(0,0,0,0.06);
        padding: 10px 12px;
        border-radius: 16px;
      }}
      [data-testid="stMetricLabel"] {{
        color: rgba(35,135,131,0.95);
        font-weight: 900;
      }}

      /* ---------- Comparison HTML Table ---------- */
      table.comp {{
        width: 100%;
        border-collapse: collapse;
        background: rgba(255,255,255,0.82);
        border-radius: 16px;
        overflow: hidden;
      }}
      table.comp th {{
        text-align: left;
        padding: 10px 10px;
        font-size: 12px;
        color: {POINT};
        background: rgba(130,161,255,0.22);
        border-bottom: 1px solid rgba(0,0,0,0.06);
        white-space: nowrap;
      }}
      table.comp td {{
        padding: 9px 10px;
        font-size: 12px;
        border-bottom: 1px solid rgba(0,0,0,0.05);
        white-space: nowrap;
      }}
      tr.delta {{
        background: rgba(35,135,131,0.09);
      }}
      span.pos {{
        color: {POINT};
        font-weight: 900;
      }}
      span.neg {{
        color: #c62828;
        font-weight: 900;
      }}
      span.neu {{
        color: #607d8b;
        font-weight: 700;
      }}

      .note {{
        font-size: 11px;
        color: rgba(0,40,115,0.62);
        margin-top: 6px;
      }}

      .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 2.0rem;
      }}
    </style>
    """,
    unsafe_allow_html=True
)


def card_start(title: str, subtitle: str = ""):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="section-head">
          <div>
            <div class="section-title">{title}</div>
            {"<div class='section-sub'>" + subtitle + "</div>" if subtitle else ""}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def main_title(text: str):
    st.markdown(f'<div class="main-title">{text}</div>', unsafe_allow_html=True)


# ============================================
# SECTION 1) HELPERS
# ============================================

@st.cache_data
def read_csv_safely(file_bytes: bytes) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949"]
    seps = [",", "\t", ";"]
    last_error = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    pd.io.common.BytesIO(file_bytes),
                    encoding=enc,
                    sep=sep,
                    engine="python"
                )
                if df.shape[1] == 1:
                    continue
                return df
            except Exception as e:
                last_error = e
    raise RuntimeError(f"CSV 로딩 실패: {last_error}")


def get_excel_sheets(file_bytes: bytes) -> list[str]:
    xls = pd.ExcelFile(pd.io.common.BytesIO(file_bytes), engine="openpyxl")
    return xls.sheet_names


@st.cache_data
def read_excel_safely(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(pd.io.common.BytesIO(file_bytes), sheet_name=sheet_name, engine="openpyxl")


def to_number(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": None, "-": None, "—": None, "nan": None, "None": None})
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("₩", "", regex=False)
    s = s.str.replace("원", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0)


def safe_div(a, b):
    return a / b if b and b != 0 else 0


def fmt_int(x: float) -> str:
    return f"{x:,.0f}"


def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def month_start_end(d: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(d.year, d.month, 1)
    last_day = calendar.monthrange(d.year, d.month)[1]
    end = pd.Timestamp(d.year, d.month, last_day)
    return start, end


def add_months(dt: pd.Timestamp, months: int) -> pd.Timestamp:
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    day = min(dt.day, calendar.monthrange(y, m)[1])
    return pd.Timestamp(y, m, day)


def window(anchor_start: pd.Timestamp, length_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = anchor_start + pd.Timedelta(days=length_days - 1)
    return anchor_start, end


# ---- robust date parsing (스코어카드 월 인식 오류 해결 핵심) ----
def parse_date_series(s: pd.Series) -> pd.Series:
    """
    다양한 날짜 형태를 robust하게 파싱:
    - Excel serial number
    - 'YYYY-MM-DD', 'YYYY/MM/DD', 'YYYY.MM.DD'
    - 'YYYY년 M월 D일'
    - 혼합일 경우도 최대한 살림
    """
    # 1) 이미 datetime이면 그대로
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    # 2) 숫자형(엑셀 시리얼) 가능성
    #    숫자로 변환 가능한 값이 많으면 excel origin으로 변환 시도
    s_num = pd.to_numeric(s, errors="coerce")
    num_ratio = s_num.notna().mean()
    if num_ratio >= 0.6:  # 과반 이상 숫자면 excel serial로 판단
        # 엑셀 날짜 시리얼 기준(Windows): origin 1899-12-30
        dt = pd.to_datetime(s_num, unit="D", origin="1899-12-30", errors="coerce")
        return dt

    # 3) 문자열 정규화 후 to_datetime
    s_str = s.astype(str).str.strip()

    # 'YYYY년 M월 D일' 같은 형태 정리
    s_str = s_str.str.replace(r"\s+", "", regex=True)
    s_str = s_str.str.replace("년", "-", regex=False)
    s_str = s_str.str.replace("월", "-", regex=False)
    s_str = s_str.str.replace("일", "", regex=False)

    # 구분자 통일
    s_str = s_str.str.replace("/", "-", regex=False)
    s_str = s_str.str.replace(".", "-", regex=False)

    # 1차 파싱
    dt1 = pd.to_datetime(s_str, errors="coerce", format=None)

    # 4) 혹시 'YYYYMMDD' 같은 8자리 연속 숫자 문자열이 섞이면 추가 처리
    if dt1.isna().mean() > 0.2:
        cand = s_str.str.extract(r"^(\d{8})$")[0]
        dt2 = pd.to_datetime(cand, errors="coerce", format="%Y%m%d")
        dt1 = dt1.fillna(dt2)

    return dt1


def guess_col(all_columns: list[str], keywords: list[str]) -> str:
    for c in all_columns:
        cc = str(c).lower()
        if any(k.lower() in cc for k in keywords):
            return c
    return "(없음)"


# ============================================
# SECTION 2) MAPPING / KPI MODEL
# ============================================

def build_defaults(columns: list[str]) -> dict:
    return {
        "date": guess_col(columns, ["일자", "date", "day"]),
        "media": guess_col(columns, ["미디어", "매체", "media", "channel"]),
        "adproduct": guess_col(columns, ["광고상품", "상품", "placement", "ad product", "adproduct"]),
        "campaign": guess_col(columns, ["캠페인", "campaign"]),
        "opt_goal": guess_col(columns, ["목표", "goal", "objective", "opt.goal", "optimization"]),
        "spent": guess_col(columns, ["광고비", "비용", "cost", "spent", "소진금액", "spend"]),
        "impression": guess_col(columns, ["노출", "노출수", "impression", "impr"]),
        "clicks": guess_col(columns, ["클릭", "click", "클릭수", "clicks", "link click", "link clicks"]),
        "purchase": guess_col(columns, ["구매", "purchase", "구매수", "orders", "conversion"]),
        "revenue": guess_col(columns, ["매출", "매출액", "revenue", "sales", "gmv"]),
        "install": guess_col(columns, ["앱설치", "install", "installs"]),
        "signup": guess_col(columns, ["회원가입", "signup", "sign up", "register"]),
        "target": guess_col(columns, ["타겟", "타겟그룹", "target", "audience"]),
        "creative": guess_col(columns, ["소재", "크리에이티브", "creative", "ad name", "asset"]),
        "etc": "(없음)",
    }


@dataclass
class Mapping:
    date: str
    media: str
    adproduct: str
    campaign: str
    opt_goal: str
    spent: str
    impression: str
    clicks: str
    purchase: str
    revenue: str
    install: str
    signup: str
    target: str
    creative: str
    etc: str


def map_ui_one_block(raw_cols: list[str], defaults: dict) -> Mapping:
    cols = ["(없음)"] + raw_cols

    r1 = st.columns(5)
    with r1[0]:
        c_date = st.selectbox("date*", cols, index=cols.index(defaults["date"]) if defaults["date"] in cols else 0)
    with r1[1]:
        c_media = st.selectbox("media", cols, index=cols.index(defaults["media"]) if defaults["media"] in cols else 0)
    with r1[2]:
        c_adproduct = st.selectbox("adproduct", cols, index=cols.index(defaults["adproduct"]) if defaults["adproduct"] in cols else 0)
    with r1[3]:
        c_campaign = st.selectbox("campaign", cols, index=cols.index(defaults["campaign"]) if defaults["campaign"] in cols else 0)
    with r1[4]:
        c_goal = st.selectbox("opt.goal", cols, index=cols.index(defaults["opt_goal"]) if defaults["opt_goal"] in cols else 0)

    r2 = st.columns(5)
    with r2[0]:
        c_spent = st.selectbox("spent", cols, index=cols.index(defaults["spent"]) if defaults["spent"] in cols else 0)
    with r2[1]:
        c_impr = st.selectbox("impression", cols, index=cols.index(defaults["impression"]) if defaults["impression"] in cols else 0)
    with r2[2]:
        c_clicks = st.selectbox("clicks", cols, index=cols.index(defaults["clicks"]) if defaults["clicks"] in cols else 0)
    with r2[3]:
        c_purchase = st.selectbox("purchase", cols, index=cols.index(defaults["purchase"]) if defaults["purchase"] in cols else 0)
    with r2[4]:
        c_revenue = st.selectbox("revenue", cols, index=cols.index(defaults["revenue"]) if defaults["revenue"] in cols else 0)

    r3 = st.columns(5)
    with r3[0]:
        c_install = st.selectbox("install", cols, index=cols.index(defaults["install"]) if defaults["install"] in cols else 0)
    with r3[1]:
        c_signup = st.selectbox("signup", cols, index=cols.index(defaults["signup"]) if defaults["signup"] in cols else 0)
    with r3[2]:
        c_target = st.selectbox("target", cols, index=cols.index(defaults["target"]) if defaults["target"] in cols else 0)
    with r3[3]:
        c_creative = st.selectbox("creative", cols, index=cols.index(defaults["creative"]) if defaults["creative"] in cols else 0)
    with r3[4]:
        c_etc = st.selectbox("etc", cols, index=0)

    return Mapping(
        date=c_date, media=c_media, adproduct=c_adproduct, campaign=c_campaign, opt_goal=c_goal,
        spent=c_spent, impression=c_impr, clicks=c_clicks, purchase=c_purchase, revenue=c_revenue,
        install=c_install, signup=c_signup, target=c_target, creative=c_creative, etc=c_etc
    )


def build_canonical_df(raw: pd.DataFrame, m: Mapping) -> pd.DataFrame:
    df = pd.DataFrame()

    # ★ 핵심: robust parse
    df["date"] = parse_date_series(raw[m.date])
    df = df.dropna(subset=["date"]).copy()

    df["media"] = raw[m.media].astype(str).str.strip() if m.media != "(없음)" else "(all)"
    df["adproduct"] = raw[m.adproduct].astype(str).str.strip() if m.adproduct != "(없음)" else "(all)"
    df["campaign"] = raw[m.campaign].astype(str).str.strip() if m.campaign != "(없음)" else "(all)"
    df["opt_goal"] = raw[m.opt_goal].astype(str).str.strip() if m.opt_goal != "(없음)" else "(all)"
    df["target"] = raw[m.target].astype(str).str.strip() if m.target != "(없음)" else ""
    df["creative"] = raw[m.creative].astype(str).str.strip() if m.creative != "(없음)" else ""
    df["etc"] = raw[m.etc].astype(str).str.strip() if m.etc != "(없음)" else ""

    df["spent"] = to_number(raw[m.spent]) if m.spent != "(없음)" else 0
    df["impression"] = to_number(raw[m.impression]) if m.impression != "(없음)" else 0
    df["clicks"] = to_number(raw[m.clicks]) if m.clicks != "(없음)" else 0
    df["purchase"] = to_number(raw[m.purchase]) if m.purchase != "(없음)" else 0
    df["revenue"] = to_number(raw[m.revenue]) if m.revenue != "(없음)" else 0
    df["install"] = to_number(raw[m.install]) if m.install != "(없음)" else 0
    df["signup"] = to_number(raw[m.signup]) if m.signup != "(없음)" else 0

    df["CTR"] = df.apply(lambda r: safe_div(r["clicks"], r["impression"]), axis=1)
    df["CVR"] = df.apply(lambda r: safe_div(r["purchase"], r["clicks"]), axis=1)
    df["ROAS"] = df.apply(lambda r: safe_div(r["revenue"], r["spent"]), axis=1)
    df["CPC"] = df.apply(lambda r: safe_div(r["spent"], r["clicks"]), axis=1)
    df["CPI"] = df.apply(lambda r: safe_div(r["spent"], r["install"]), axis=1)
    df["CPA(purchase)"] = df.apply(lambda r: safe_div(r["spent"], r["purchase"]), axis=1)
    df["CPA(signup)"] = df.apply(lambda r: safe_div(r["spent"], r["signup"]), axis=1)

    return df


# ============================================
# SECTION 3) RENDER: UPLOAD / PREVIEW (중복 렌더 제거)
# ============================================

st.title("HFM Report Dashboard")

up_l, up_r = st.columns([1, 1])

with up_l:
    card_start("Upload")
    uploaded = st.file_uploader("CSV / Excel 업로드", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    card_end()

# 업로드 전/후 Preview를 '한 번만' 렌더
with up_r:
    if uploaded is None:
        card_start("Preview")
        st.caption("파일 업로드 후 미리보기가 표시됩니다.")
        card_end()
        st.stop()

file_bytes = uploaded.getvalue()
name = uploaded.name.lower()

# load raw
try:
    if name.endswith(".csv"):
        raw = read_csv_safely(file_bytes)
        sheet_used = None
        sheets = None
    else:
        sheets = get_excel_sheets(file_bytes)
        sheet_used = sheets[0]
        raw = read_excel_safely(file_bytes, sheet_used)
except Exception as e:
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

raw = raw.dropna(how="all").copy()
raw.columns = [str(c).strip() for c in raw.columns]

# Preview (업로드 후)
with up_r:
    card_start("Preview")
    if sheets:
        sheet_used = st.selectbox("엑셀 시트 선택", sheets, index=0)
        raw = read_excel_safely(file_bytes, sheet_used)
        raw = raw.dropna(how="all").copy()
        raw.columns = [str(c).strip() for c in raw.columns]

    with st.expander("원본 미리보기 (상위 30행)", expanded=False):
        st.dataframe(raw.head(30), use_container_width=True, hide_index=True)
    card_end()

st.success(f"업로드 완료: {uploaded.name} · {raw.shape[0]:,} rows × {raw.shape[1]:,} cols")


# ============================================
# SECTION 4) COLUMN MAPPING + CLEANED PREVIEW
# ============================================

card_start("Column Mapping")
defaults = build_defaults(list(raw.columns))
mapping = map_ui_one_block(list(raw.columns), defaults)
card_end()

if mapping.date == "(없음)":
    st.error("date(일자)는 필수입니다. 컬럼을 선택해 주세요.")
    st.stop()

df = build_canonical_df(raw, mapping)

card_start("Data Clean & KPI")
st.caption("정제 및 KPI 계산 결과(하단 토글에서 확인)")
card_end()

with st.expander("정제 데이터 미리보기 (상위 50행)", expanded=False):
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)

st.divider()


# ============================================
# SECTION 5) CAMPAIGN RESULT OVERALL (월 선택 로직 수정)
# ============================================

def summarize_totals(dfx: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    d = dfx[(dfx["date"] >= start) & (dfx["date"] <= end)].copy()
    spent = d["spent"].sum()
    impr = d["impression"].sum()
    clicks = d["clicks"].sum()
    purchase = d["purchase"].sum()
    revenue = d["revenue"].sum()
    install = d["install"].sum()

    ctr = safe_div(clicks, impr)
    cpc = safe_div(spent, clicks)
    cpa_p = safe_div(spent, purchase)
    cvr = safe_div(purchase, clicks)
    roas = safe_div(revenue, spent)
    cpi = safe_div(spent, install)

    return {
        "spent": spent, "impression": impr, "clicks": clicks, "CTR": ctr, "CPC": cpc,
        "purchase": purchase, "revenue": revenue, "CPA(purchase)": cpa_p, "CVR": cvr, "ROAS": roas,
        "install": install, "CPI": cpi
    }


def pct_change(cur: float, base: float) -> float:
    return safe_div(cur - base, base)


def metric_card(label: str, value_str: str, m_pct: float | None, y_pct: float | None):
    st.metric(label, value_str)
    line = []
    if m_pct is not None:
        line.append(f"전월 {fmt_pct(m_pct)}")
    if y_pct is not None:
        line.append(f"전년동월 {fmt_pct(y_pct)}")
    st.markdown(f"<div class='note'>{' | '.join(line) if line else '비교 대상 기간 데이터 없음'}</div>", unsafe_allow_html=True)


main_title("Campaign Result Overall")

# ★ 최신 날짜가 잘못 잡히는 문제: date 파싱 개선 + max 확인용 디버그(필요하면 펼쳐 확인)
max_dt = df["date"].max()
min_dt = df["date"].min()

with st.expander("날짜 인식 확인(디버그)", expanded=False):
    st.write("min date:", min_dt)
    st.write("max date:", max_dt)

cur_m_start, cur_m_end = month_start_end(max_dt.date())

# 기존 tabs 방식은 session_state 덮어쓰기 문제 -> radio(CTA)로 변경
month_choice = st.radio(
    "기간 선택",
    ["당월(최신월)", "전월", "전년도 동월"],
    horizontal=True,
    index=0
)

if month_choice == "당월(최신월)":
    base_start, base_end = cur_m_start, cur_m_end
elif month_choice == "전월":
    prev_anchor = add_months(pd.Timestamp(cur_m_start), -1).date()
    base_start, base_end = month_start_end(prev_anchor)
else:
    yoy_anchor = date(cur_m_start.year - 1, cur_m_start.month, 1)
    base_start, base_end = month_start_end(yoy_anchor)

base_len = (base_end - base_start).days + 1
pm_start = add_months(base_start, -1)
pm_start, pm_end = window(pm_start, base_len)
py_start = add_months(base_start, -12)
py_start, py_end = window(py_start, base_len)

base_sum = summarize_totals(df, base_start, base_end)
pm_sum = summarize_totals(df, pm_start, pm_end)
py_sum = summarize_totals(df, py_start, py_end)

r1 = st.columns(5)
with r1[0]:
    metric_card("광고비", fmt_int(base_sum["spent"]), pct_change(base_sum["spent"], pm_sum["spent"]), pct_change(base_sum["spent"], py_sum["spent"]))
with r1[1]:
    metric_card("노출", fmt_int(base_sum["impression"]), pct_change(base_sum["impression"], pm_sum["impression"]), pct_change(base_sum["impression"], py_sum["impression"]))
with r1[2]:
    metric_card("클릭", fmt_int(base_sum["clicks"]), pct_change(base_sum["clicks"], pm_sum["clicks"]), pct_change(base_sum["clicks"], py_sum["clicks"]))
with r1[3]:
    metric_card("CTR", fmt_pct(base_sum["CTR"]), pct_change(base_sum["CTR"], pm_sum["CTR"]), pct_change(base_sum["CTR"], py_sum["CTR"]))
with r1[4]:
    metric_card("CPC", fmt_int(base_sum["CPC"]), pct_change(base_sum["CPC"], pm_sum["CPC"]), pct_change(base_sum["CPC"], py_sum["CPC"]))

r2 = st.columns(7)
with r2[0]:
    metric_card("구매", fmt_int(base_sum["purchase"]), pct_change(base_sum["purchase"], pm_sum["purchase"]), pct_change(base_sum["purchase"], py_sum["purchase"]))
with r2[1]:
    metric_card("매출액", fmt_int(base_sum["revenue"]), pct_change(base_sum["revenue"], pm_sum["revenue"]), pct_change(base_sum["revenue"], py_sum["revenue"]))
with r2[2]:
    metric_card("CPA(구매)", fmt_int(base_sum["CPA(purchase)"]), pct_change(base_sum["CPA(purchase)"], pm_sum["CPA(purchase)"]), pct_change(base_sum["CPA(purchase)"], py_sum["CPA(purchase)"]))
with r2[3]:
    metric_card("CVR", fmt_pct(base_sum["CVR"]), pct_change(base_sum["CVR"], pm_sum["CVR"]), pct_change(base_sum["CVR"], py_sum["CVR"]))
with r2[4]:
    metric_card("ROAS", f"{base_sum['ROAS']*100:.0f}%", pct_change(base_sum["ROAS"], pm_sum["ROAS"]), pct_change(base_sum["ROAS"], py_sum["ROAS"]))
with r2[5]:
    metric_card("앱설치", fmt_int(base_sum["install"]), pct_change(base_sum["install"], pm_sum["install"]), pct_change(base_sum["install"], py_sum["install"]))
with r2[6]:
    metric_card("CPI", fmt_int(base_sum["CPI"]), pct_change(base_sum["CPI"], pm_sum["CPI"]), pct_change(base_sum["CPI"], py_sum["CPI"]))

st.markdown(
    f"""
    <div class="note">
      * 선택 기간: {base_start.date()} ~ {base_end.date()}<br>
      * 전월 동기간: {pm_start.date()} ~ {pm_end.date()}<br>
      * 전년도 동기간: {py_start.date()} ~ {py_end.date()}
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================
# SECTION 6) DATA TREND CHART (summary table HTML 통일)
# ============================================

main_title("Data Trend Chart")

card_start("Filters")
trend_period_options = ["당월(최신월)", "전월", "전년도 동월"]
trend_period_sel = st.multiselect("비교 기간 선택(복수 선택 가능)", trend_period_options, default=trend_period_options)

# period map
cur_start, cur_end = cur_m_start, cur_m_end
prev_anchor = add_months(pd.Timestamp(cur_m_start), -1).date()
prev_start, prev_end = month_start_end(prev_anchor)
yoy_anchor = date(cur_m_start.year - 1, cur_m_start.month, 1)
yoy_start, yoy_end = month_start_end(yoy_anchor)

period_map = {
    "당월(최신월)": (cur_start, cur_end),
    "전월": (prev_start, prev_end),
    "전년도 동월": (yoy_start, yoy_end),
}

# scope
scopes = []
for p in trend_period_sel:
    s, e = period_map[p]
    scopes.append(df[(df["date"] >= s) & (df["date"] <= e)])
trend_scope = pd.concat(scopes, ignore_index=True) if scopes else df.iloc[0:0]

f1, f2, f3, f4 = st.columns(4)
with f1:
    media_opts = sorted(trend_scope["media"].unique())
    media_sel = st.multiselect("미디어(media)", media_opts, default=media_opts)
    d1 = trend_scope[trend_scope["media"].isin(media_sel)] if media_sel else trend_scope.iloc[0:0]
with f2:
    prod_opts = sorted(d1["adproduct"].unique())
    prod_sel = st.multiselect("광고상품(adproduct)", prod_opts, default=prod_opts)
    d2 = d1[d1["adproduct"].isin(prod_sel)] if prod_sel else d1.iloc[0:0]
with f3:
    camp_opts = sorted(d2["campaign"].unique())
    camp_sel = st.multiselect("캠페인(campaign)", camp_opts, default=camp_opts)
    d3 = d2[d2["campaign"].isin(camp_sel)] if camp_sel else d2.iloc[0:0]
with f4:
    goal_opts = sorted(d3["opt_goal"].unique())
    goal_sel = st.multiselect("목표(opt.goal)", goal_opts, default=goal_opts)
    trend_filtered_all = d3[d3["opt_goal"].isin(goal_sel)] if goal_sel else d3.iloc[0:0]

metric_opt = st.selectbox(
    "트렌드 지표",
    ["spent", "impression", "clicks", "purchase", "revenue", "install", "signup",
     "CTR", "CVR", "ROAS", "CPC", "CPI", "CPA(purchase)", "CPA(signup)"]
)

card_end()

def period_daily(dfx: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, label: str) -> pd.DataFrame:
    d = dfx[(dfx["date"] >= start) & (dfx["date"] <= end)].copy()
    scaffold = pd.DataFrame({"day": list(range(1, 32))})
    if d.empty:
        scaffold["period"] = label
        scaffold["value"] = None
        return scaffold

    daily = d.groupby("date", as_index=False).agg(
        spent=("spent", "sum"),
        impression=("impression", "sum"),
        clicks=("clicks", "sum"),
        purchase=("purchase", "sum"),
        revenue=("revenue", "sum"),
        install=("install", "sum"),
        signup=("signup", "sum"),
    )
    daily["CTR"] = daily.apply(lambda r: safe_div(r["clicks"], r["impression"]), axis=1)
    daily["CVR"] = daily.apply(lambda r: safe_div(r["purchase"], r["clicks"]), axis=1)
    daily["ROAS"] = daily.apply(lambda r: safe_div(r["revenue"], r["spent"]), axis=1)
    daily["CPC"] = daily.apply(lambda r: safe_div(r["spent"], r["clicks"]), axis=1)
    daily["CPI"] = daily.apply(lambda r: safe_div(r["spent"], r["install"]), axis=1)
    daily["CPA(purchase)"] = daily.apply(lambda r: safe_div(r["spent"], r["purchase"]), axis=1)
    daily["CPA(signup)"] = daily.apply(lambda r: safe_div(r["spent"], r["signup"]), axis=1)

    daily["day"] = daily["date"].dt.day
    keep = daily[["day", metric_opt]].rename(columns={metric_opt: "value"})
    keep = scaffold.merge(keep, on="day", how="left")
    keep["period"] = label
    return keep

trend_parts = []
for p in trend_period_sel:
    s, e = period_map[p]
    trend_parts.append(period_daily(trend_filtered_all, s, e, p))
trend_df = pd.concat(trend_parts, ignore_index=True) if trend_parts else pd.DataFrame({"day": [], "value": [], "period": []})

fig = px.line(trend_df, x="day", y="value", color="period", markers=True)
fig.update_layout(xaxis_title="Day (1~31)", yaxis_title=metric_opt)
st.plotly_chart(fig, use_container_width=True)

# ---- Summary Table: HTML(comp) 스타일로 통일 ----
def period_summary_row(dfx: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    s = summarize_totals(dfx, start, end)
    return {
        "광고비": s["spent"],
        "노출": s["impression"],
        "클릭": s["clicks"],
        "CTR": s["CTR"],
        "CPC": s["CPC"],
        "구매": s["purchase"],
        "매출액": s["revenue"],
        "ROAS": s["ROAS"],
        "CPI": s["CPI"],
    }

def format_summary_cell(col: str, val: float, is_pct_line: bool):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if is_pct_line:
        cls = "pos" if val > 0 else ("neg" if val < 0 else "neu")
        return f'<span class="{cls}">{fmt_pct(val)}</span>'
    # normal
    if col == "CTR":
        return fmt_pct(val)
    if col == "ROAS":
        return f"{val*100:.0f}%"
    if col in ["광고비", "매출액", "CPC", "CPI"]:
        return fmt_int(val)
    return fmt_int(val)

# rows 구성
rows = []
for p in trend_period_sel:
    s, e = period_map[p]
    rows.append((f"{p} ({s.date()}~{e.date()})", period_summary_row(trend_filtered_all, s, e), False))

def add_delta_pct(cur_key: str, base_key: str, label: str):
    cur_s, cur_e = period_map[cur_key]
    base_s, base_e = period_map[base_key]
    cur = period_summary_row(trend_filtered_all, cur_s, cur_e)
    base = period_summary_row(trend_filtered_all, base_s, base_e)
    delta = {}
    for k in cur.keys():
        delta[k] = safe_div(cur[k] - base[k], base[k])
    rows.append((label, delta, True))

if "당월(최신월)" in trend_period_sel and "전월" in trend_period_sel:
    add_delta_pct("당월(최신월)", "전월", "Δ% (당월 vs 전월)")
if "당월(최신월)" in trend_period_sel and "전년도 동월" in trend_period_sel:
    add_delta_pct("당월(최신월)", "전년도 동월", "Δ% (당월 vs 전년동월)")

# HTML table render
summary_cols = ["광고비","노출","클릭","CTR","CPC","구매","매출액","ROAS","CPI"]
thead = "".join([f"<th>{c}</th>" for c in (["구분"] + summary_cols)])
tbody = ""
for label, data, is_delta in rows:
    cls = "delta" if is_delta else ""
    tds = f"<td><b>{label}</b></td>"
    for c in summary_cols:
        tds += f"<td>{format_summary_cell(c, data.get(c), is_delta)}</td>"
    tbody += f"<tr class='{cls}'>{tds}</tr>"

st.markdown("#### Summary Table", unsafe_allow_html=True)
st.markdown(
    f"""
    <table class="comp">
      <thead><tr>{thead}</tr></thead>
      <tbody>{tbody}</tbody>
    </table>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================
# SECTION 7) PERFORMANCE COMPARISON (중복 소제목 제거)
# ============================================

main_title("Performance Comparison")

# (중복 카드 타이틀 제거) 바로 UI 렌더
cmp_tab1, cmp_tab2 = st.tabs(["최근 7일(기본)", "기간 직접 선택"])

max_ts = df["date"].max()
default_end = pd.Timestamp(max_ts.year, max_ts.month, max_ts.day)
default_start = default_end - pd.Timedelta(days=6)

with cmp_tab1:
    cmp_start, cmp_end = default_start, default_end

with cmp_tab2:
    c_start, c_end = st.date_input(
        "선택 기간",
        value=(default_start.date(), default_end.date())
    )
    cmp_start, cmp_end = pd.to_datetime(c_start), pd.to_datetime(c_end)

base_len = (cmp_end - cmp_start).days + 1
w_start, w_end = window(cmp_start - pd.Timedelta(days=7), base_len)
m_start = add_months(cmp_start, -1)
m_start, m_end = window(m_start, base_len)
y_start = add_months(cmp_start, -12)
y_start, y_end = window(y_start, base_len)

base_sum = summarize_totals(df, cmp_start, cmp_end)
week_sum = summarize_totals(df, w_start, w_end)
month_sum = summarize_totals(df, m_start, m_end)
year_sum = summarize_totals(df, y_start, y_end)

def diff_row(cur, prev):
    return {k: cur[k] - prev.get(k, 0) for k in cur.keys()}

def pct_row(cur, prev):
    return {k: safe_div((cur[k] - prev.get(k, 0)), prev.get(k, 0)) for k in cur.keys()}

order_cols = ["spent","impression","clicks","CTR","CPC","purchase","revenue","CPA(purchase)","CVR","ROAS","install","CPI"]
label_map = {
    "spent":"광고비","impression":"노출","clicks":"클릭","CTR":"CTR","CPC":"CPC",
    "purchase":"구매","revenue":"매출액","CPA(purchase)":"CPA(구매)","CVR":"CVR","ROAS":"ROAS",
    "install":"앱설치","CPI":"CPI"
}

def format_cell(row_label, col, val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    is_delta_pct = "Δ%" in row_label
    is_delta = "Δ (" in row_label

    if is_delta_pct:
        cls = "pos" if val > 0 else ("neg" if val < 0 else "neu")
        return f'<span class="{cls}">{fmt_pct(val)}</span>'
    if is_delta:
        cls = "pos" if val > 0 else ("neg" if val < 0 else "neu")
        if col in ["CTR", "CVR"]:
            return f'<span class="{cls}">{val*100:+.2f}%p</span>'
        if col == "ROAS":
            return f'<span class="{cls}">{val*100:+.0f}%p</span>'
        return f'<span class="{cls}">{val:+,.0f}</span>'

    if col in ["CTR","CVR"]:
        return fmt_pct(val)
    if col == "ROAS":
        return f"{val*100:.0f}%"
    if col in ["spent","revenue","CPC","CPA(purchase)","CPI"]:
        return fmt_int(val)
    return fmt_int(val)

rows = []
rows.append((f"선택기간 ({cmp_start.date()}~{cmp_end.date()})", base_sum, False))
rows.append((f"전주 ({w_start.date()}~{w_end.date()})", week_sum, False))
rows.append(("Δ (선택-전주)", diff_row(base_sum, week_sum), True))
rows.append(("Δ% (선택-전주)", pct_row(base_sum, week_sum), True))

rows.append((f"전월동기간 ({m_start.date()}~{m_end.date()})", month_sum, False))
rows.append(("Δ (선택-전월)", diff_row(base_sum, month_sum), True))
rows.append(("Δ% (선택-전월)", pct_row(base_sum, month_sum), True))

rows.append((f"전년도동기간 ({y_start.date()}~{y_end.date()})", year_sum, False))
rows.append(("Δ (선택-전년)", diff_row(base_sum, year_sum), True))
rows.append(("Δ% (선택-전년)", pct_row(base_sum, year_sum), True))

thead = "".join([f"<th>{c}</th>" for c in (["구분"] + [label_map[c] for c in order_cols])])
tbody = ""
for label, data_dict, is_delta in rows:
    cls = "delta" if is_delta else ""
    tds = f"<td><b>{label}</b></td>"
    for c in order_cols:
        tds += f"<td>{format_cell(label, c, data_dict.get(c))}</td>"
    tbody += f"<tr class='{cls}'>{tds}</tr>"

st.markdown(
    f"""
    <table class="comp">
      <thead><tr>{thead}</tr></thead>
      <tbody>{tbody}</tbody>
    </table>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================
# SECTION 8) WHAT DRIVE THE CHANGE? (중복 소제목 제거)
# ============================================

main_title("What drive the change?")

bench = st.radio("비교 기준", ["전주", "전월동기간", "전년도동기간"], horizontal=True)
if bench == "전주":
    b_start, b_end = w_start, w_end
elif bench == "전월동기간":
    b_start, b_end = m_start, m_end
else:
    b_start, b_end = y_start, y_end

driver_metric = st.selectbox("편차 기준 지표", ["revenue", "purchase", "spent"], index=0)
driver_label = {"revenue":"매출액", "purchase":"구매", "spent":"광고비"}[driver_metric]

st.markdown(
    f"<div class='note'>선택기간: {cmp_start.date()}~{cmp_end.date()} / 비교기간({bench}): {b_start.date()}~{b_end.date()}</div>",
    unsafe_allow_html=True
)

def group_delta(dfx: pd.DataFrame, group_cols: list[str], metric: str,
                cur_start: pd.Timestamp, cur_end: pd.Timestamp,
                base_start: pd.Timestamp, base_end: pd.Timestamp) -> tuple[pd.DataFrame, float]:
    cur = dfx[(dfx["date"] >= cur_start) & (dfx["date"] <= cur_end)]
    base = dfx[(dfx["date"] >= base_start) & (dfx["date"] <= base_end)]

    cur_g = cur.groupby(group_cols, as_index=False)[metric].sum().rename(columns={metric: "cur"})
    base_g = base.groupby(group_cols, as_index=False)[metric].sum().rename(columns={metric: "base"})

    out = cur_g.merge(base_g, on=group_cols, how="outer").fillna(0)
    out["delta"] = out["cur"] - out["base"]
    total_delta = out["delta"].sum()

    out["contrib"] = out["delta"].apply(lambda x: safe_div(x, total_delta) if total_delta != 0 else 0)
    out["abs_delta"] = out["delta"].abs()
    out = out.sort_values("abs_delta", ascending=False).drop(columns=["abs_delta"])
    return out, total_delta

def fmt_delta(val: float) -> str:
    cls = "pos" if val > 0 else ("neg" if val < 0 else "neu")
    return f'<span class="{cls}">{val:+,.0f}</span>'

def fmt_contrib(val: float) -> str:
    cls = "pos" if val > 0 else ("neg" if val < 0 else "neu")
    return f'<span class="{cls}">{fmt_pct(val)}</span>'

def render_driver_table(df_driver: pd.DataFrame, title: str, total_delta: float, top_n: int = 15):
    st.markdown(f"#### {title}", unsafe_allow_html=True)
    st.markdown(
        f"<div class='note'>전체 {driver_label} 증감(선택-비교): {fmt_delta(total_delta)}</div>",
        unsafe_allow_html=True
    )

    show = df_driver.head(top_n).copy()
    view_cols = [c for c in show.columns if c not in ["cur", "base"]]  # group + delta + contrib
    thead = "".join([f"<th>{c}</th>" for c in view_cols])

    tbody = ""
    for _, r in show.iterrows():
        tds = ""
        for c in view_cols:
            if c == "delta":
                tds += f"<td>{fmt_delta(r[c])}</td>"
            elif c == "contrib":
                tds += f"<td>{fmt_contrib(r[c])}</td>"
            else:
                tds += f"<td>{str(r[c])}</td>"
        tbody += f"<tr>{tds}</tr>"

    st.markdown(
        f"""
        <table class="comp">
          <thead><tr>{thead}</tr></thead>
          <tbody>{tbody}</tbody>
        </table>
        """,
        unsafe_allow_html=True
    )

t_media, t_prod, t_camp = st.tabs(["미디어", "광고상품", "캠페인"])

with t_media:
    d, total = group_delta(df, ["media"], driver_metric, cmp_start, cmp_end, b_start, b_end)
    d = d.rename(columns={"media": "미디어"})
    render_driver_table(d[["미디어", "delta", "contrib"]], "미디어 기여도", total)

with t_prod:
    d, total = group_delta(df, ["media", "adproduct"], driver_metric, cmp_start, cmp_end, b_start, b_end)
    d = d.rename(columns={"media": "미디어", "adproduct": "광고상품"})
    render_driver_table(d[["미디어", "광고상품", "delta", "contrib"]], "광고상품 기여도(미디어 포함)", total)

with t_camp:
    d, total = group_delta(df, ["media", "adproduct", "campaign"], driver_metric, cmp_start, cmp_end, b_start, b_end)
    d = d.rename(columns={"media": "미디어", "adproduct": "광고상품", "campaign": "캠페인"})
    render_driver_table(d[["미디어", "광고상품", "캠페인", "delta", "contrib"]], "캠페인 기여도(미디어/상품 포함)", total)

st.divider()


# ============================================
# SECTION 9) EXPORT
# ============================================

card_start("Export")
csv_bytes = trend_filtered_all.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="CSV 다운로드",
    data=csv_bytes,
    file_name="filtered_report.csv",
    mime="text/csv"
)
with st.expander("Export 대상 데이터 미리보기 (상위 50행)", expanded=False):
    st.dataframe(trend_filtered_all.head(50), use_container_width=True, hide_index=True)
card_end()