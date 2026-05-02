import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import requests
import ta
from sklearn.preprocessing import RobustScaler
import time
import os
import warnings

# 🌟 최상단 배치 (Streamlit 설정)
st.set_page_config(page_title="AI Quant Radar V6", layout="wide")
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. V6 마스터 AI 모델 구조 ---
class SwingBinaryMasterGRU(nn.Module):
    def __init__(self, input_size=19, hidden_size=128, num_layers=2):
        super(SwingBinaryMasterGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))
    def forward(self, x):
        out, _ = self.gru(x)
        w = torch.softmax(torch.tanh(self.attention(out)), dim=1)
        c = torch.sum(w * out, dim=1)
        return self.fc(c)

# --- 2. 보안 접속 (Enter 키 대응) ---
def check_password():
    def password_entered():
        # 비밀번호가 맞으면 세션 상태 업데이트 후 입력값 삭제
        if st.session_state["password"] == "dlghdud121!":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🛡️ AI Quant Radar 접속 보안")
        # on_change를 사용하여 Enter 입력 시 바로 password_entered 함수 실행
        st.text_input("보안 코드를 입력하고 Enter를 누르세요", type="password", on_change=password_entered, key="password")
        st.info("승인된 트레이더만 접속 가능합니다.")
        st.stop()
        return False
    elif not st.session_state["password_correct"]:
        st.title("🛡️ AI Quant Radar 접속 보안")
        st.text_input("보안 코드를 입력하고 Enter를 누르세요", type="password", on_change=password_entered, key="password")
        st.error("❌ 보안 코드가 일치하지 않습니다.")
        st.stop()
        return False
    return True

# --- 3. 데이터 및 분석 엔진 ---
@st.cache_resource
def load_trained_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwingBinaryMasterGRU(input_size=19)
    try:
        if not os.path.exists(path): return None, device
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model, device
    except: return None, device

@st.cache_data(ttl=3600)
def get_macro_dashboard_data():
    indices = {"USD/KRW": "USD/KRW", "NASDAQ": "IXIC", "S&P500": "US500", "KOSPI": "KS11", "KOSDAQ": "KQ11"}
    data = {}
    for name, code in indices.items():
        try:
            df = fdr.DataReader(code).tail(2)
            curr, prev = df['Close'].iloc[-1], df['Close'].iloc[-2]
            data[name] = (curr, ((curr - prev) / prev) * 100)
        except: continue
    return data

@st.cache_data(ttl=3600*6)
def load_macro_feature_data():
    end_dt = datetime.today().strftime('%Y-%m-%d')
    start_dt = (datetime.today() - timedelta(days=500)).strftime('%Y-%m-%d')
    usdkrw = fdr.DataReader('USD/KRW', start_dt, end_dt)['Close'].rename('usd_krw')
    nasdaq = fdr.DataReader('IXIC', start_dt, end_dt)['Close'].rename('nasdaq')
    macro_df = pd.concat([usdkrw, nasdaq], axis=1).ffill().dropna()
    macro_df['usd_krw_ret'] = macro_df['usd_krw'].pct_change()
    macro_df['nasdaq_ret'] = macro_df['nasdaq'].pct_change()
    return macro_df[['usd_krw_ret', 'nasdaq_ret']]

def get_naver_supply_demand(code):
    try:
        url = f"https://finance.naver.com/item/frgn.naver?code={code}"
        res = requests.get(url, headers={'User-agent': 'Mozilla/5.0'}, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        tables = soup.find_all('table', class_='type2')
        if not tables or len(tables) < 2: return 0, 0, None
        rows = tables[1].find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 9 and cols[0].text.strip(): 
                inst_str = cols[5].text.strip().replace(',', '')
                for_str = cols[6].text.strip().replace(',', '')
                return int(inst_str), int(for_str), cols[0].text.strip()
        return 0, 0, None
    except: return 0, 0, None

def prepare_master_features(ticker, df_chart, macro_df):
    if len(df_chart) < 60: return pd.DataFrame(), None, 0, 0
    df = df_chart.copy().join(macro_df, how='left')
    df.ffill().bfill(inplace=True)
    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
    
    feats = pd.DataFrame(index=df.index)
    feats['ret'] = close.pct_change()
    feats['dist_ma'] = close / (close.rolling(20).mean() + 1e-8)
    feats['macd_hist'] = ta.trend.MACD(close).macd_diff()
    feats['adx'] = ta.trend.ADXIndicator(high, low, close).adx() / 100.0
    feats['rsi'] = ta.momentum.RSIIndicator(close).rsi() / 100.0
    feats['stoch'] = ta.momentum.StochasticOscillator(high, low, close).stoch() / 100.0
    feats['bb_pband'] = ta.volatility.BollingerBands(close).bollinger_pband()
    feats['atr_pct'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range() / (close + 1e-8)
    feats['obv_ret'] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume().pct_change()
    feats['mfi'] = ta.volume.MFIIndicator(high, low, close, vol).money_flow_index() / 100.0
    feats['bb_width'] = ta.volatility.BollingerBands(close).bollinger_wband() / 100.0
    feats['cci'] = ta.trend.CCIIndicator(high, low, close).cci() / 100.0
    feats['roc'] = ta.momentum.ROCIndicator(close).roc() / 100.0
    feats['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, vol).chaikin_money_flow()
    feats['will_r'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r() / -100.0
    
    inst_net, for_net, valid_date = get_naver_supply_demand(ticker)
    feats['inst_ratio'] = 0.0; feats['foreigner_ratio'] = 0.0 
    last_idx = feats.index[-1]; last_vol = vol.iloc[-1] + 1e-8
    feats.loc[last_idx, 'inst_ratio'] = inst_net / last_vol
    feats.loc[last_idx, 'foreigner_ratio'] = for_net / last_vol
    feats['usd_krw_ret'] = df['usd_krw_ret']; feats['nasdaq_ret'] = df['nasdaq_ret']
    
    return feats.dropna(), valid_date, inst_net/last_vol, for_net/last_vol

def generate_ai_judgment_brief(name, prob, f_r, i_r):
    line1 = f"🚀 **{name} AI 판단:** 상승 확률 **{prob:.1f}%**로 " + ("강력 매수 저격 구간" if prob >= 75 else "매수 우위 구간")
    if f_r > 0 and i_r > 0: line2 = "✅ **수급 상황:** 외인/기관 쌍끌이 매수 포착. 신뢰도 높음."
    elif f_r > 0: line2 = "✅ **수급 상황:** 외국인 매수세가 지지 중. 수급 긍정적."
    else: line2 = "⚠️ **수급 상황:** 메이저 수급이 약함. 차트 중심 대응 필요."
    return f"{line1}\n\n{line2}"

@st.cache_data(ttl=3600)
def get_v6_market_rankings(market_type="KOSPI", top_n=50):
    try:
        df_list = fdr.StockListing(market_type)
        if market_type == "ETF/KR":
             tickers = df_list.sort_values('Volume', ascending=False).head(top_n)['Symbol'].tolist()
             names = df_list.sort_values('Volume', ascending=False).head(top_n)['Name'].tolist()
        else:
             tickers = df_list.sort_values('Marcap', ascending=False).head(top_n)['Code'].tolist()
             names = df_list.sort_values('Marcap', ascending=False).head(top_n)['Name'].tolist()
        ticker_name_map = dict(zip(tickers, names))
    except: return pd.DataFrame()
    results = []
    prog = st.progress(0)
    macro_df = load_macro_feature_data()
    for i, ticker in enumerate(tickers):
        try:
            df_chart = fdr.DataReader(ticker, (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d'))
            feats_df, v_date, i_r, f_r = prepare_master_features(ticker, df_chart, macro_df)
            if feats_df.empty: continue
            inp = torch.FloatTensor(RobustScaler().fit_transform(feats_df.tail(60).values)).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(inp), dim=1).cpu().numpy()[0][1] * 100
            results.append({"종목명": ticker_name_map[ticker], "코드": ticker, "상승확률": prob, "현재가": int(df_chart['Close'].iloc[-1])})
        except: continue
        prog.progress((i + 1) / len(tickers))
    prog.empty()
    return pd.DataFrame(results)

@st.cache_data
def get_top10_news(name):
    news_list = []
    try:
        url = f"https://news.google.com/rss/search?q={name} 주식&hl=ko&gl=KR&ceid=KR:ko"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, 'xml')
        for item in soup.find_all('item')[:5]:
            news_list.append({"title": item.title.get_text(), "link": item.link.get_text()})
    except: pass
    return news_list

def generate_ai_briefing(name, buy_prob, for_ratio, inst_ratio, rsi, stoch, valid_date):
    date_str = f"({valid_date} 장마감 기준)" if valid_date else "(수급 정보 없음)"
    briefing = f"**[{name} AI 트레이딩 브리핑]** {date_str}\n\n"
    
    if buy_prob >= 75: briefing += f"🎯 **초고도 확신 구간 ({buy_prob:.1f}%):** AI가 강력한 단기 상승을 예고합니다. "
    elif buy_prob >= 60: briefing += f"👍 **매수 우위 구간 ({buy_prob:.1f}%):** 상승 에너지가 긍정적입니다. "
    elif buy_prob >= 40: briefing += f"☁️ **관망 구간 ({buy_prob:.1f}%):** 뚜렷한 방향성이 없습니다. "
    else: briefing += f"🛑 **하락 위험 구간 ({buy_prob:.1f}%):** 하방 압력이 강합니다. "

    # 수치 기반 정밀 멘트 (0.001 -> 0.1% 이상 유입 시)
    if for_ratio > 0.001 and inst_ratio > 0.001: briefing += "특히 외국인과 기관의 쌍끌이 매수가 유입되며 상승 추세를 뒷받침하고 있습니다.\n\n"
    elif for_ratio > 0.001: briefing += "외국인 자금이 유입되며 하방을 방어하고 있습니다.\n\n"
    elif inst_ratio > 0.001: briefing += "기관의 저가 매수세가 들어오고 있습니다.\n\n"
    elif for_ratio < -0.001 and inst_ratio < -0.001: briefing += "현재 메이저(외인/기관) 양매도가 출회 중이므로 접근에 주의가 필요합니다.\n\n"
    else: briefing += "현재 메이저 수급의 뚜렷한 유입은 포착되지 않았습니다.\n\n"
    
    briefing += "💡 **기술적 위치:** "
    if rsi > 0.7 or stoch > 0.8: briefing += "현재 단기 과열권에 진입했습니다. 추격 매수보다는 눌림목을 기다리거나 분할 익절을 고려하세요."
    elif rsi < 0.3 or stoch < 0.2: briefing += "현재 단기 과매도(침체) 구간입니다. 저점 반등을 노린 단기 매수 접근이 유효합니다."
    else: briefing += "현재 적정 밸류에이션 구간에서 방향을 탐색 중입니다."
    
    return briefing

# --- 4. 메인 실행부 ---
MODEL_PATH = "weather_advisor_v6_master_D.pt"
model, device = load_trained_model(MODEL_PATH)

if check_password():
    # 🌟 사이드바: 글로벌 매크로 상시 노출
    idx_data = get_macro_dashboard_data()
    st.sidebar.title("🌍 Global Macro")
    for k, v in idx_data.items():
        st.sidebar.metric(k, f"{v[0]:,.2f}", f"{v[1]:+.2f}%")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.radio("모드 선택", ["🔍 단일 종목 X-Ray", "🎯 V6 스윙 타점 스캐너", "🌐 글로벌 매크로 & ETF"], index=0)

if menu == "🔍 단일 종목 X-Ray":
    st.title("🔍 단일 종목 X-Ray")
    target_input = st.sidebar.text_input("종목명/코드", value="삼성전자")
    
    st.info(f"📅 실시간 네이버 금융 수급 연동 중")
    
    with st.spinner("종목 정보를 검색 중입니다..."):
        listing = pd.concat([fdr.StockListing('KRX'), fdr.StockListing('ETF/KR').rename(columns={'Symbol':'Code'})])
        match = listing[listing['Name'].str.contains(target_input, na=False) | (listing['Code'] == target_input)].drop_duplicates('Code')

    if match.empty:
        st.error(f"❌ '{target_input}' 종목을 찾을 수 없습니다. (정확한 상장 종목명이나 코드를 입력해 주세요)")
    elif model is None:
        st.error(f"❌ AI 모델 파일을 불러오지 못했습니다.")
    else:
        ticker, name = match['Code'].iloc[0], match['Name'].iloc[0]
        
        with st.spinner(f"[{name}] 글로벌 매크로 및 차트/수급 데이터 수집 중..."):
            try:
                df_chart = fdr.DataReader(ticker, (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d'))
                if df_chart.empty or len(df_chart) < 60:
                    st.error("❌ 차트 데이터가 부족하여 분석할 수 없습니다. (최소 60일치 캔들 필요)")
                    st.stop()
                    
                macro_df = load_macro_feature_data()
                feats_df, valid_date, inst_ratio, for_ratio = prepare_master_features(ticker, df_chart, macro_df)
                
                if feats_df.empty:
                    st.error("❌ 지표 추출 중 오류가 발생했습니다. (데이터 결측치 발생)")
                    st.stop()
                    
            except Exception as e:
                st.error(f"❌ 데이터 수집 중 오류가 발생했습니다: {e}")
                st.stop()

        curr_p = df_chart['Close'].iloc[-1]
        recent = feats_df.tail(60).values
        scaled = RobustScaler().fit_transform(recent)
        input_t = torch.FloatTensor(scaled).unsqueeze(0).to(device)
        
        with torch.no_grad():
            probs = torch.softmax(model(input_t), dim=1).cpu().numpy()[0]
        
        buy_prob = probs[1] * 100
        if buy_prob >= 75: ai_status, color = "🚀 강력 매수 (Sniper)", "green"
        elif buy_prob >= 60: ai_status, color = "👍 매수 우위", "blue"
        elif buy_prob >= 40: ai_status, color = "☁️ 관망 (방향성 탐색)", "gray"
        else: ai_status, color = "🛑 하락 위험 (접근 금지)", "red"
        
        last_rsi = feats_df['rsi'].iloc[-1]
        last_stoch = feats_df['stoch'].iloc[-1]

        # 🌟 종합 요약 브리핑
        st.markdown("---")
        st.subheader("💡 종합 AI 트레이딩 판단")
        briefing_text = generate_ai_briefing(name, buy_prob, for_ratio, inst_ratio, last_rsi, last_stoch, valid_date)
        st.info(briefing_text)
        st.markdown("---")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"📈 {name} 일봉 차트")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_chart.index[-100:], open=df_chart['Open'][-100:], high=df_chart['High'][-100:], 
                                         low=df_chart['Low'][-100:], close=df_chart['Close'][-100:], name='Price'))
            fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🤖 AI 타점 분석")
            st.metric("AI 포지션", ai_status, f"상승확률 {buy_prob:.1f}%", delta_color="normal" if buy_prob >=50 else "inverse")
            
            st.write("📊 **직전 장 수급 동향 (총 거래량 대비 비중)**")
            if valid_date:
                st.write(f"📅 기준일: {valid_date}")
            st.write(f"- 외국인: {for_ratio * 100:+.2f}%")
            st.write(f"- 기  관: {inst_ratio * 100:+.2f}%")
            
            st.markdown("---")
            st.write("🎯 **트레이딩 가이드**")
            st.write(f"단기 익절(+5%): {int(curr_p * 1.05):,}원")
            st.write(f"단기 손절(-4%): {int(curr_p * 0.96):,}원")

        st.markdown("---")
        st.subheader(f"📰 {name} 최신 뉴스")
        news = get_top10_news(name)
        for n in news: st.markdown(f"• [{n['title']}]({n['link']})")
        
elif menu == "🌐 글로벌 매크로 & ETF":
    st.title("🌐 글로벌 매크로 & ETF 레이더")
    st.write("시장 전체의 자금 흐름과 분위기를 파악합니다.")
    
    idx_data = get_macro_dashboard_data()
    idx_cols = st.columns(len(idx_data))
    for i, (k, v) in enumerate(idx_data.items()):
        color = "normal" if k == "USD/KRW" else ("inverse" if v[1] < 0 else "normal")
        idx_cols[i].metric(k, f"{v[0]:,.2f}", f"{v[1]:+.2f}%", delta_color=color)
    
    st.markdown("---")
    st.subheader("🔥 AI 거래대금 상위 ETF 방향성 스캔")
    if st.button("ETF 스캔 시작 (Top 50)"):
        with st.spinner("거래대금 상위 ETF 타점 분석 중..."):
            etf_df = get_v6_market_rankings("ETF/KR", top_n=50)
            if not etf_df.empty:
                display_df = etf_df.copy()
                display_df['상승확률'] = display_df['상승확률'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df.sort_values("상승확률", ascending=False).reset_index(drop=True), use_container_width=True)

elif menu == "🎯 V6 스윙 타점 스캐너":
    st.title("🎯 V6 저격수 스캐너 (Daily)")
    st.info(f"📅 실시간 네이버 금융 수급 연동 중 (주말/휴일 완벽 대응)")
    
    if model is None:
        st.error(f"❌ AI 모델 파일을 불러오지 못했습니다. 경로를 확인해주세요: \n`{MODEL_PATH}`")
        st.stop()

    m_type = st.radio("타겟 시장 선택", ["KOSDAQ", "KOSPI"], horizontal=True)
    if st.button(f"🚀 {m_type} 시총 Top 100 타점 스캔"):
        with st.spinner("네이버 금융에서 외국인/기관 수급 데이터를 긁어오고 있습니다. 약 1~2분 소요됩니다..."):
            rank_df = get_v6_market_rankings(m_type, top_n=100)
    
        if rank_df.empty: st.error("⚠️ 데이터를 불러오지 못했습니다.")
        else:
            display_df = rank_df.copy()
            sniper_df = display_df[display_df['상승확률'] >= 75.0].sort_values("상승확률", ascending=False).reset_index(drop=True)
            display_df['상승확률'] = display_df['상승확률'].apply(lambda x: f"{x:.1f}%")
            
            st.markdown("---")
            if not sniper_df.empty:
                st.success(f"🎯 **[Sniper] 75% 이상 초고도 확신 매수 타점 포착 ({len(sniper_df)}건)**")
                sniper_df['상승확률'] = sniper_df['상승확률'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(sniper_df, use_container_width=True)
            else:
                st.warning("🎯 오늘 장은 75% 이상 확신할 만한 완벽한 매수 타점이 없습니다. (관망 권장)")

            st.markdown("---")
            st.write("📋 전체 스캔 결과 (참고용)")
            st.dataframe(display_df.sort_values("상승확률", ascending=False).reset_index(drop=True), use_container_width=True)
