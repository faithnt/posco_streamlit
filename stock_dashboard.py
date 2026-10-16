from datetime import date, timedelta
import io
import os
from pathlib import Path
import smtplib
import zipfile
import xml.etree.ElementTree as ET
from email.message import EmailMessage

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import altair as alt
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).resolve().with_name(".env"))

st.set_page_config(page_title="국내 주식 대시보드", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f4f6f8; color: #17202a; }
    [data-testid="stHeader"] { background: transparent !important; height: 2.75rem !important; }
    [data-testid="stToolbar"] { visibility: hidden; height: 0; }
    .block-container { max-width: 1440px; width: 100%; padding: 1.25rem 2.5rem 3rem; }
    .brand { display:flex; align-items:center; gap:10px; margin-bottom:2px; }
    .brand-mark { background:#03c75a; color:white; border-radius:7px; font-weight:900; padding:5px 9px; font-size:18px; }
    .brand-name { font-weight:800; color:#17202a; font-size:23px; letter-spacing:-1px; }
    .muted { color:#7c8793; font-size:13px; }
    div[data-testid="stMetric"] { background:white; border:1px solid #e3e8ed; border-radius:10px; padding:13px 15px; box-shadow:0 1px 3px rgba(30,50,70,.04); }
    div[data-testid="stMetricLabel"] p { color:#697581; font-size:13px; }
    div[data-testid="stMetricValue"] { font-size:22px; color:#17202a; }
    .section-title { font-size:18px; font-weight:750; color:#17202a; margin:24px 0 10px; letter-spacing:-.5px; }
    .stTabs [data-baseweb="tab-list"] { gap:24px; border-bottom:1px solid #dfe5ea; overflow:visible; }
    .stTabs [data-baseweb="tab"] { padding:10px 2px; color:#667380 !important; font-weight:600; white-space:nowrap; }
    .stTabs [data-baseweb="tab"] p { color:inherit !important; }
    .stTabs [aria-selected="true"] { color:#03a94f !important; border-bottom:2px solid #03c75a; }
    .stTabs [data-baseweb="tab-panel"] { padding: 0; overflow: visible; }
    [data-testid="stLineChart"], [data-testid="stVegaLiteChart"] { width:100% !important; max-width:100%; overflow:hidden; }
    [data-testid="stDataFrame"] { border:1px solid #e1e6ea; border-radius:8px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

STOCKS = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "373220.KS": "LG에너지솔루션",
    "005380.KS": "현대차",
    "000270.KS": "기아",
    "035420.KS": "NAVER",
    "035720.KS": "카카오",
    "051910.KS": "LG화학",
    "006400.KS": "삼성SDI",
    "068270.KS": "셀트리온",
}


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    # yfinance의 end 날짜는 미포함이므로 선택한 종료일 다음 날까지 요청합니다.
    data = yf.download(
        tickers=list(tickers), start=start, end=end + timedelta(days=1),
        auto_adjust=True, progress=False, threads=True,
    )
    if data.empty or "Close" not in data:
        return pd.DataFrame()
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.rename(columns=STOCKS).dropna(how="all")


@st.cache_data(ttl=900, show_spinner=False)
def load_news(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            yf_items = yf.Ticker(ticker).news[:8]
            for item in yf_items:
                content = item.get("content", item)
                url_obj = content.get("canonicalUrl", content.get("clickThroughUrl", {}))
                provider = content.get("provider", {})
                rows.append({"종목": STOCKS.get(ticker, ticker), "제목": content.get("title", ""), "언론사": provider.get("displayName", "") if isinstance(provider, dict) else str(provider), "게시일": content.get("pubDate", ""), "링크": url_obj.get("url", "") if isinstance(url_obj, dict) else ""})
        except Exception:
            pass
        # yfinance 버전에 따라 Ticker.news가 비어 있는 경우 Yahoo 검색 API로 보완합니다.
        if not any(row["종목"] == STOCKS.get(ticker, ticker) for row in rows):
            try:
                payload = requests.get("https://query1.finance.yahoo.com/v1/finance/search", params={"q": ticker, "newsCount": 8}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).json()
                for item in payload.get("news", []):
                    rows.append({"종목": STOCKS.get(ticker, ticker), "제목": item.get("title", ""), "언론사": item.get("publisher", ""), "게시일": item.get("providerPublishTime", ""), "링크": item.get("link", "")})
            except Exception:
                pass
        # Yahoo가 국내 종목 뉴스를 반환하지 않는 경우 Google News RSS를 사용합니다.
        if not any(row["종목"] == STOCKS.get(ticker, ticker) for row in rows):
            try:
                company = STOCKS.get(ticker, ticker)
                rss = requests.get("https://news.google.com/rss/search", params={"q": f"{company} 주식", "hl": "ko", "gl": "KR", "ceid": "KR:ko"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                root = ET.fromstring(rss.content)
                for item in root.findall("./channel/item")[:8]:
                    rows.append({"종목": company, "제목": item.findtext("title", ""), "언론사": item.findtext("source", ""), "게시일": item.findtext("pubDate", ""), "링크": item.findtext("link", "")})
            except Exception:
                pass
    return pd.DataFrame(rows).drop_duplicates(subset=["제목"]) if rows else pd.DataFrame(columns=["종목", "제목", "언론사", "게시일", "링크"])


@st.cache_data(ttl=86400, show_spinner=False)
def load_corp_codes(api_key: str) -> dict[str, str]:
    response = requests.get("https://opendart.fss.or.kr/api/corpCode.xml", params={"crtfc_key": api_key}, timeout=20)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        root = pd.read_xml(archive.open("CORPCODE.xml"))
    return dict(zip(root["corp_name"].astype(str).str.strip(), root["corp_code"].astype(str).str.zfill(8)))


@st.cache_data(ttl=900, show_spinner=False)
def load_disclosures(api_key: str, corp_code: str, company: str) -> pd.DataFrame:
    end = date.today(); start = end - timedelta(days=365)
    payload = requests.get("https://opendart.fss.or.kr/api/list.json", params={"crtfc_key": api_key, "corp_code": corp_code, "bgn_de": start.strftime("%Y%m%d"), "end_de": end.strftime("%Y%m%d"), "page_count": 20}, timeout=20).json()
    if payload.get("status") != "000": return pd.DataFrame()
    return pd.DataFrame([{"종목": company, "접수일": r.get("rcept_dt", ""), "보고서명": r.get("report_nm", ""), "제출인": r.get("flr_nm", ""), "공시 링크": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={r.get('rcept_no', '')}"} for r in payload.get("list", [])])


def make_report(summary: pd.DataFrame, news: pd.DataFrame, period: str) -> str:
    lines = [f"국내 주식 시장 리포트 ({date.today().isoformat()})", "", f"조회 기간: {period}", "", "[종목 요약]"]
    for _, row in summary.iterrows():
        lines.append(f"- {row['종목']}: {row['최근 종가(원)']:,.0f}원 / {row[f'{period} 수익률']:.2f}%")
    lines.append("\n[관련 뉴스]")
    if news.empty:
        lines.append("- 수집된 뉴스가 없습니다.")
    else:
        for _, row in news.head(15).iterrows():
            lines.append(f"- [{row['종목']}] {row['제목']} ({row['언론사']})\n  {row['링크']}")
    lines += ["", "※ 본 리포트는 참고용이며 투자 권유가 아닙니다."]
    return "\n".join(lines)


def send_report_email(report: str, recipient: str) -> None:
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    if not username or not password:
        raise ValueError("SMTP_USER와 SMTP_PASSWORD를 .env에 설정해 주세요.")
    message = EmailMessage()
    message["Subject"] = f"국내 주식 리포트 - {date.today().isoformat()}"
    message["From"] = username
    message["To"] = recipient
    message.set_content(report)
    with smtplib.SMTP(host, port, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(username, password)
        smtp.send_message(message)


st.markdown('<div class="brand"><span class="brand-mark">N</span><span class="brand-name">국내 증권</span></div>', unsafe_allow_html=True)
st.markdown('<span class="muted">시세 · 종목분석 · 시장현황</span>', unsafe_allow_html=True)

with st.sidebar:
    st.header("조회 설정")
    period = st.selectbox("조회 기간", ["1개월", "3개월", "6개월", "1년", "2년"], index=2)
    days = {"1개월": 30, "3개월": 90, "6개월": 180, "1년": 365, "2년": 730}[period]
    st.caption("한국거래소 종목 코드를 사용합니다. 실제 투자 판단 전 거래소 데이터와 대조해 주세요.")

st.markdown('<div class="section-title">관심 종목</div>', unsafe_allow_html=True)
selected = st.pills(
    "관심 종목을 선택하세요",
    options=list(STOCKS.values()),
    selection_mode="multi",
    default=list(STOCKS.values()),
    key="watchlist",
)

if not selected:
    st.info("왼쪽에서 조회할 종목을 선택해 주세요.")
    st.stop()

today = date.today()
prices = load_prices(tuple(ticker for ticker, name in STOCKS.items() if name in selected), today - timedelta(days=days), today)
if prices.empty:
    st.error("주가 데이터를 가져오지 못했습니다. 네트워크 연결이나 Yahoo Finance 응답을 확인한 뒤 다시 시도해 주세요.")
    if st.button("다시 불러오기"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

prices = prices.loc[:, [name for name in selected if name in prices.columns]]
returns = prices.apply(lambda series: series.dropna().iloc[-1] / series.dropna().iloc[0] - 1 if series.notna().sum() > 1 else float("nan"))
latest = prices.ffill().iloc[-1]

st.markdown('<div class="section-title">관심 종목 시세</div>', unsafe_allow_html=True)
metrics = st.columns(min(5, len(selected)))
for i, name in enumerate(selected[:5]):
    if name in latest.index:
        delta = returns.get(name)
        metrics[i].metric(name, f"₩{latest[name]:,.0f}", f"{delta:+.2%}" if pd.notna(delta) else None)

tab_chart, tab_rank = st.tabs(["차트", "수익률 순위"])
with tab_chart:
    st.markdown(f'<div class="section-title">종목별 주가 추이 <span class="muted">최근 {period}</span></div>', unsafe_allow_html=True)
    chart_data = prices.reset_index().rename(columns={prices.index.name or "index": "날짜"}).melt("날짜", var_name="종목", value_name="수정 종가")
    chart_data["날짜"] = pd.to_datetime(chart_data["날짜"])
    available_dates = sorted(chart_data["날짜"].dt.date.unique())
    range_start, range_end = st.slider(
        "차트 조회 구간",
        min_value=available_dates[0], max_value=available_dates[-1],
        value=(available_dates[0], available_dates[-1]),
        format="YYYY-MM-DD", key="chart_date_range",
    )
    chart_data = chart_data[(chart_data["날짜"].dt.date >= range_start) & (chart_data["날짜"].dt.date <= range_end)]
    data_min = range_start.isoformat()
    data_max = range_end.isoformat()
    chart = alt.Chart(chart_data).mark_line(strokeWidth=2).encode(
        x=alt.X("날짜:T", title="날짜", scale=alt.Scale(domain=[data_min, data_max])),
        y=alt.Y("수정 종가:Q", title="수정 종가 (원)"),
        color=alt.Color("종목:N", legend=alt.Legend(orient="bottom", columns=5)),
        tooltip=[alt.Tooltip("날짜:T", title="날짜"), alt.Tooltip("종목:N"), alt.Tooltip("수정 종가:Q", format=",.0f")],
    ).properties(height=390)
    st.altair_chart(chart, width="stretch", theme=None)

summary = pd.DataFrame({
    "종목": latest.index,
    "최근 종가(원)": latest.values,
    f"{period} 수익률": [returns.get(name) * 100 for name in latest.index],
    "데이터 기준일": [prices[name].last_valid_index().date().isoformat() if prices[name].notna().any() else "-" for name in latest.index],
}).sort_values(f"{period} 수익률", ascending=False, na_position="last")
with tab_rank:
    st.markdown('<div class="section-title">기간 수익률 순위</div>', unsafe_allow_html=True)
    st.bar_chart(summary.set_index("종목")[f"{period} 수익률"], y_label="수익률 (%)", height=390)

st.markdown('<div class="section-title">종목별 현황</div>', unsafe_allow_html=True)
st.dataframe(summary, hide_index=True, width="stretch", column_config={
    "최근 종가(원)": st.column_config.NumberColumn(format="₩%.0f"),
    f"{period} 수익률": st.column_config.NumberColumn(format="%.2f%%"),
})
st.download_button("CSV 다운로드", summary.to_csv(index=False).encode("utf-8-sig"), "국내주식_요약.csv", "text/csv", width="content")

news = load_news(tuple(ticker for ticker, name in STOCKS.items() if name in selected))
st.markdown('<div class="section-title">기업 뉴스</div>', unsafe_allow_html=True)
if news.empty:
    st.info("현재 가져올 수 있는 뉴스가 없습니다.")
else:
    news_filter = st.selectbox("뉴스 종목", ["전체"] + selected, key="news_filter")
    visible_news = news if news_filter == "전체" else news[news["종목"] == news_filter]
    for _, item in visible_news.head(20).iterrows():
        link = item["링크"]
        title = f"[{item['제목']}]({link})" if link else item["제목"]
        st.markdown(f"**{item['종목']}**  {title}  \n<span class='muted'>{item['언론사']} · {item['게시일']}</span>", unsafe_allow_html=True)

dart_key = os.getenv("OPENDART_API_KEY", "")
st.markdown('<div class="section-title">OpenDART 공시</div>', unsafe_allow_html=True)
disclosures = pd.DataFrame()
if not dart_key:
    st.info("공시를 사용하려면 프로젝트 루트의 .env에 OPENDART_API_KEY를 설정해 주세요.", icon="📄")
else:
    try:
        corp_codes = load_corp_codes(dart_key)
        dart_company = st.selectbox("공시 종목", selected, key="dart_company")
        corp_code = corp_codes.get(dart_company)
        if corp_code:
            disclosures = load_disclosures(dart_key, corp_code, dart_company)
            if disclosures.empty: st.info("최근 1년 공시가 없습니다.")
            else: st.dataframe(disclosures, hide_index=True, width="stretch", column_config={"공시 링크": st.column_config.LinkColumn("공시 링크")})
        else: st.warning(f"OpenDART 기업 코드에서 {dart_company}를 찾지 못했습니다.")
    except Exception as exc:
        st.error(f"OpenDART 데이터를 불러오지 못했습니다: {exc}")

st.markdown('<div class="section-title">메일로 리포트 받기</div>', unsafe_allow_html=True)
st.caption("현재 주가와 수집된 뉴스를 텍스트 리포트로 만들어 SMTP로 발송합니다.")
default_recipient = os.getenv("REPORT_RECIPIENT", os.getenv("SMTP_USER", ""))
recipient = st.text_input("받는 사람 이메일", value=default_recipient, placeholder="name@example.com", key="report_recipient")
report_preview = make_report(summary, news, period)
with st.expander("리포트 미리보기"):
    st.text(report_preview)
if st.button("리포트 메일 보내기", type="primary", disabled=not bool(recipient.strip())):
    try:
        send_report_email(report_preview, recipient.strip())
        st.success(f"리포트를 {recipient.strip()}로 보냈습니다.")
    except Exception as exc:
        st.error(f"메일 발송에 실패했습니다: {exc}")

st.markdown('<div class="section-title">AI 종목 상담</div>', unsafe_allow_html=True)
st.caption("현재 조회한 주가 데이터만 바탕으로 답변합니다. 투자 판단의 최종 책임은 사용자에게 있습니다.")

api_key = os.getenv("OPENAI_API_KEY", "")
try:
    api_key = api_key or st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    pass
if not api_key:
    st.info("챗봇을 사용하려면 환경변수 OPENAI_API_KEY를 설정하거나 .streamlit/secrets.toml에 OPENAI_API_KEY를 입력해 주세요.", icon="🔑")
else:
    if "stock_chat_messages" not in st.session_state:
        st.session_state.stock_chat_messages = [{
            "role": "assistant",
            "content": "안녕하세요. 현재 조회된 국내 주식 데이터에 대해 물어보세요. 예: '수익률이 가장 좋은 종목은?'",
        }]
    for message in st.session_state.stock_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("주가나 수익률에 대해 질문해 보세요"):
        st.session_state.stock_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        context = summary.to_csv(index=False)
        news_context = news.head(30).to_csv(index=False) if not news.empty else "뉴스 없음"
        dart_context = disclosures.to_csv(index=False) if not disclosures.empty else "공시 없음"
        history = st.session_state.stock_chat_messages[-7:]
        instructions = (
            "당신은 국내 주식 데이터 분석 도우미입니다. 사용자가 제공한 표에 있는 숫자만 사용해 한국어로 답변하세요. "
            "표에 없는 최신 뉴스, 목표주가, 매매 추천은 추측하지 말고 데이터가 없다고 말하세요. "
            "수익률은 표의 기간 기준이며, 답변은 간결하게 핵심 수치와 근거를 포함하세요.\n\n"
            f"현재 주가 데이터(CSV):\n{context}\n\n관련 뉴스(CSV):\n{news_context}\n\nOpenDART 공시(CSV):\n{dart_context}"
        )
        try:
            client = OpenAI(api_key=api_key)
            with st.chat_message("assistant"):
                with st.spinner("데이터를 확인하고 있습니다..."):
                    response = client.responses.create(
                        model="gpt-4.1-mini",
                        instructions=instructions,
                        input=[{"role": item["role"], "content": item["content"]} for item in history],
                    )
                    answer = response.output_text
                    st.markdown(answer)
            st.session_state.stock_chat_messages.append({"role": "assistant", "content": answer})
        except Exception as exc:
            st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {exc}")
