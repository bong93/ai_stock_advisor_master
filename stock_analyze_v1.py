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
from pykrx import stock
import joblib
import lightgbm as lgb
import time
import os
import warnings

# 🌟 최상단 배치 (Streamlit 설정)
st.set_page_config(page_title="AI Quant Radar V6 Ensemble", layout="wide")
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# --- 1. V6 앙상블 마스터 AI 모델 구조 (24 Features) ---
class SwingBinaryMasterGRU(nn.Module):
    def __init__(self, input_size=24, hidden_size=128, num_layers=2):
        super(SwingBinaryMasterGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))
    def forward(self, x):
        out, _ = self.gru(x)
        w = torch.softmax(torch.tanh(self.attention(out)), dim=1)
        c = torch.sum(w * out, dim=1)
        return self.fc(c)

# --- 2. 보안 접속 ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "dlghdud121!":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🛡️ AI Quant Radar 접속 보안")
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
@st.cache_data(ttl=3600*24) # 하루 동안 리스트 저장 (속도 최적화)
def get_all_stock_list():
    try:
        # KRX 종목과 ETF 종목을 한 번에 가져와서 "종목명 (코드)" 형태로 만듭니다.
        df_krx = fdr.StockListing('KRX')
        df_etf = fdr.StockListing('ETF/KR').rename(columns={'Symbol':'Code'})
        
        krx_list = df_krx['Name'] + " (" + df_krx['Code'] + ")"
        etf_list = df_etf['Name'] + " (" + df_etf['Code'] + ")"
        
        # 두 리스트를 합친 후 리스트 형태로 반환
        return pd.concat([krx_list, etf_list]).dropna().tolist()
    except:
        return ["삼성전자 (005930)"] # 에러 방지용 기본값
    
@st.cache_resource
def load_ensemble_models(gru_path, lgb_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_gru = SwingBinaryMasterGRU(input_size=24)
    try:
        if not os.path.exists(gru_path) or not os.path.exists(lgb_path): 
            return None, None, device
        # GRU 로드
        state_dict = torch.load(gru_path, map_location=device, weights_only=True)
        model_gru.load_state_dict(state_dict)
        model_gru.to(device).eval()
        # LGBM 로드
        model_lgb = joblib.load(lgb_path)
        return model_gru, model_lgb, device
    except: return None, None, device

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
    kospi = fdr.DataReader('KS11', start_dt, end_dt)['Close'].rename('kospi')
    kosdaq = fdr.DataReader('KQ11', start_dt, end_dt)['Close'].rename('kosdaq')
    vix = fdr.DataReader('VIX', start_dt, end_dt)['Close'].rename('vix')
    
    macro_df = pd.concat([usdkrw, nasdaq, kospi, kosdaq, vix], axis=1).ffill().dropna()
    macro_df['usd_krw_ret'] = macro_df['usd_krw'].pct_change()
    macro_df['nasdaq_ret'] = macro_df['nasdaq'].pct_change()
    macro_df['kospi_ret'] = macro_df['kospi'].pct_change()
    macro_df['kosdaq_ret'] = macro_df['kosdaq'].pct_change()
    macro_df['vix_ret'] = macro_df['vix'].pct_change()
    return macro_df[['usd_krw_ret', 'nasdaq_ret', 'kospi_ret', 'kosdaq_ret', 'vix_ret']]

def get_naver_supply_demand_history(code, pages=4):
    records = []
    for p in range(1, pages + 1):
        try:
            url = f"https://finance.naver.com/item/frgn.naver?code={code}&page={p}"
            res = requests.get(url, headers={'User-agent': 'Mozilla/5.0'}, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            tables = soup.find_all('table', class_='type2')
            if len(tables) < 2: continue
            
            rows = tables[1].find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 9 and cols[0].text.strip():
                    date_str = cols[0].text.strip().replace('.', '-')
                    inst_str = cols[5].text.strip().replace(',', '').replace('+', '')
                    for_str = cols[6].text.strip().replace(',', '').replace('+', '')
                    try:
                        records.append({
                            'Date': pd.to_datetime(date_str),
                            'inst_net': int(inst_str),
                            'foreigner_net': int(for_str)
                        })
                    except: pass
        except: continue
        
    if records:
        return pd.DataFrame(records).set_index('Date').sort_index()
    return pd.DataFrame()

def prepare_master_features(ticker, df_chart, macro_df):
    if len(df_chart) < 60: return pd.DataFrame(), None, 0, 0, None
    
    # 🔥 Pykrx 대신 네이버 80일치 데이터를 가져와 실시간 수급까지 한 번에 해결!
    sd_df = get_naver_supply_demand_history(ticker, pages=4)
    
    df = df_chart.copy().join(macro_df, how='left')
    if not sd_df.empty: 
        df = df.join(sd_df, how='left')
    else: 
        df['inst_net'] = 0; df['foreigner_net'] = 0
        
    df.ffill(inplace=True); df.bfill(inplace=True)

    # [일목균형표 기초 지표 계산]
    high_9 = df['High'].rolling(9).max()
    low_9 = df['Low'].rolling(9).min()
    tenkan = (high_9 + low_9) / 2
    
    high_26 = df['High'].rolling(26).max()
    low_26 = df['Low'].rolling(26).min()
    kijun = (high_26 + low_26) / 2
    
    span_a_raw = (tenkan + kijun) / 2
    high_52 = df['High'].rolling(52).max()
    low_52 = df['Low'].rolling(52).min()
    span_b_raw = (high_52 + low_52) / 2

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=26, freq='B')
    df_plot = pd.concat([df, pd.DataFrame(index=future_dates)])

    tmp_span_a = pd.Series(index=df_plot.index, dtype=float)
    tmp_span_a.loc[df.index] = span_a_raw
    df_plot['senkou_span_a'] = tmp_span_a.shift(26)

    tmp_span_b = pd.Series(index=df_plot.index, dtype=float)
    tmp_span_b.loc[df.index] = span_b_raw
    df_plot['senkou_span_b'] = tmp_span_b.shift(26)

    df_plot['tenkan_sen'] = tenkan
    df_plot['kijun_sen'] = kijun

    # [24개 피처 추출]
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
    
    # 🔥 실시간 데이터가 반영된 '완벽한 수급 비율 & 5일 누적' 계산
    feats['inst_ratio'] = df['inst_net'] / (vol + 1e-8)
    feats['foreigner_ratio'] = df['foreigner_net'] / (vol + 1e-8)
    feats['inst_ratio_5d'] = df['inst_net'].rolling(window=5).sum() / (vol.rolling(window=5).sum() + 1e-8)
    feats['foreigner_ratio_5d'] = df['foreigner_net'].rolling(window=5).sum() / (vol.rolling(window=5).sum() + 1e-8)
    
    feats['usd_krw_ret'] = df['usd_krw_ret']
    feats['nasdaq_ret'] = df['nasdaq_ret']
    feats['kospi_ret'] = df['kospi_ret']
    feats['kosdaq_ret'] = df['kosdaq_ret']
    feats['vix_ret'] = df['vix_ret']
    
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats.dropna(inplace=True)
    
    # V6 앙상블 모델 전용 24개 피처 순서
    feature_cols = [
        'ret', 'dist_ma', 'macd_hist', 'adx', 'rsi', 'stoch', 'bb_pband', 'atr_pct', 'obv_ret', 'mfi', 
        'bb_width', 'cci', 'roc', 'cmf', 'will_r', 'inst_ratio', 'foreigner_ratio', 'inst_ratio_5d', 'foreigner_ratio_5d',
        'usd_krw_ret', 'nasdaq_ret', 'kospi_ret', 'kosdaq_ret', 'vix_ret'
    ]
    
    if len(feats) == 0:
        return feats, None, 0, 0, df_plot
        
    valid_date = df.index[-1].strftime('%Y-%m-%d')
    last_inst_ratio = feats['inst_ratio'].iloc[-1]
    last_for_ratio = feats['foreigner_ratio'].iloc[-1]
    
    return feats[feature_cols], valid_date, last_inst_ratio, last_for_ratio, df_plot

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
            feats_df, v_date, i_r, f_r, _ = prepare_master_features(ticker, df_chart, macro_df)
            if feats_df.empty or len(feats_df) < 60: continue
            
            scaled_feat = RobustScaler().fit_transform(feats_df.tail(60).values)
            
            # 🌟 앙상블 로직
            inp = torch.FloatTensor(scaled_feat).unsqueeze(0).to(device)
            with torch.no_grad():
                gru_prob = torch.softmax(model_gru(inp), dim=1).cpu().numpy()[0][1]
            lgb_prob = model_lgb.predict_proba(scaled_feat[-1].reshape(1, -1))[0][1]
            
            final_prob = (gru_prob * 0.5 + lgb_prob * 0.5) * 100
            results.append({"종목명": ticker_name_map[ticker], "코드": ticker, "상승확률": final_prob, "현재가": int(df_chart['Close'].iloc[-1])})
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
    
    if buy_prob >= 70: briefing += f"🔥 **[S급] 초고도 확신 구간 ({buy_prob:.1f}%):** 승률 85% 이상의 완벽한 상승 추세입니다. "
    elif buy_prob >= 60: briefing += f"🚀 **[A급] 강한 확신 구간 ({buy_prob:.1f}%):** 상승 에너지가 긍정적인 매수 추천 자리입니다. "
    elif buy_prob >= 50: briefing += f"✅ **[B급] 관망 권장 구간 ({buy_prob:.1f}%):** 반반의 확률을 가진 애매한 구간입니다. "
    else: briefing += f"🛑 **[패스] 하락 위험 구간 ({buy_prob:.1f}%):** 하방 압력이 강합니다. "

    if for_ratio > 0.001 and inst_ratio > 0.001: briefing += "특히 외국인과 기관의 쌍끌이 매수가 유입되며 상승 추세를 뒷받침하고 있습니다.\n\n"
    elif for_ratio > 0.001: briefing += "외국인 자금이 유입되며 하방을 방어하고 있습니다.\n\n"
    elif inst_ratio > 0.001: briefing += "기관의 저가 매수세가 들어오고 있습니다.\n\n"
    elif for_ratio < -0.001 and inst_ratio < -0.001: briefing += "현재 메이저(외인/기관) 양매도가 출회 중이므로 접근에 주의가 필요합니다.\n\n"
    else: briefing += "현재 메이저 수급의 뚜렷한 유입은 포착되지 않았습니다.\n\n"
    
    briefing += "💡 **기술적 위치:** "
    if rsi > 0.7 or stoch > 0.8: briefing += "현재 단기 과열권에 진입했습니다. 보유자의 경우 분할 익절을 고려하세요."
    elif rsi < 0.3 or stoch < 0.2: briefing += "현재 단기 과매도 구간입니다. 기술적 반등을 노린 접근이 유효합니다."
    else: briefing += "현재 적정 구간에서 방향을 탐색 중입니다."
    
    return briefing

def draw_ichimoku_chart(df_plot):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['senkou_span_a'], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['senkou_span_b'], line=dict(width=0), 
                             fill='tonexty', fillcolor='rgba(150, 150, 150, 0.2)', name='Kumo Cloud'))
    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], 
                                 low=df_plot['Low'], close=df_plot['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['tenkan_sen'], line=dict(color='orange', width=1), name='전환선'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['kijun_sen'], line=dict(color='dodgerblue', width=1), name='기준선'))

    fig.update_layout(
        height=500, template="plotly_dark", xaxis_rangeslider_visible=False,
        xaxis=dict(type='date', range=[df_plot.index[-146], df_plot.index[-1]]),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

# --- 4. 메인 실행부 ---
# 🚨 모델 파일 두 개가 같은 폴더에 있어야 합니다.
GRU_PATH = r"weather_advisor_v6_master_D.pt"
LGB_PATH = r"weather_advisor_v6_master_D_lgb.pkl"

model_gru, model_lgb, device = load_ensemble_models(GRU_PATH, LGB_PATH)

if check_password():
    idx_data = get_macro_dashboard_data()
    st.sidebar.title("🌍 Global Macro")
    for k, v in idx_data.items():
        st.sidebar.metric(k, f"{v[0]:,.2f}", f"{v[1]:+.2f}%")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.radio("모드 선택", ["🔍 단일 종목 X-Ray", "🎯 V6 스윙 타점 스캐너", "🌐 글로벌 매크로 & ETF"], index=0)

if menu == "🔍 단일 종목 X-Ray":
    st.title("🔍 단일 종목 X-Ray")
    
    # 🌟 1. 전체 종목 리스트 불러오기 (자동완성용)
    all_stocks = get_all_stock_list()
    default_idx = all_stocks.index("삼성전자 (005930)") if "삼성전자 (005930)" in all_stocks else 0
    
    # 🌟 2. 텍스트 입력창 대신 selectbox 사용 (여기에 타이핑하면 자동완성 필터링이 됩니다!)
    selected_item = st.sidebar.selectbox("종목 검색 (초성/이름/코드 입력)", options=all_stocks, index=default_idx)
    
    if model_gru is None or model_lgb is None:
        st.error(f"❌ AI 모델 로드 실패. 파일 경로를 확인하세요.")
    else:
        import re
        # 🌟 3. 선택된 문자열 "삼성전자 (005930)" 에서 이름과 코드를 분리
        match = re.match(r"(.*) \((.*)\)", selected_item)
        if match:
            name = match.group(1)
            ticker = match.group(2)
        else:
            name, ticker = "삼성전자", "005930"
        
        with st.spinner(f"[{name}] 데이터 분석 중..."):
            try:
                df_chart = fdr.DataReader(ticker, (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d'))
                macro_df = load_macro_feature_data()
                feats_df, v_date, i_r, f_r, df_plot = prepare_master_features(ticker, df_chart, macro_df)
                
                if not feats_df.empty and len(feats_df) >= 60:
                    curr_p = df_chart['Close'].iloc[-1]
                    prev_p = df_chart['Close'].iloc[-2]
                    st.metric(label=f"{name} ({ticker})", value=f"{int(curr_p):,}원", delta=f"{int(curr_p-prev_p):+}원 ({(curr_p-prev_p)/prev_p*100:+.2f}%)")

                    # 🌟 앙상블 AI 판단
                    scaled_feat = RobustScaler().fit_transform(feats_df.tail(60).values)
                    inp = torch.FloatTensor(scaled_feat).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        gru_prob = torch.softmax(model_gru(inp), dim=1).cpu().numpy()[0][1]
                    lgb_prob = model_lgb.predict_proba(scaled_feat[-1].reshape(1, -1))[0][1]
                    
                    final_prob = (gru_prob * 0.5) + (lgb_prob * 0.5)
                    
                    st.info(generate_ai_briefing(name, final_prob * 100, f_r, i_r, feats_df['rsi'].iloc[-1], feats_df['stoch'].iloc[-1], v_date))
                    st.markdown("---")

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.plotly_chart(draw_ichimoku_chart(df_plot), use_container_width=True)
                    
                    with col2:
                        # 🌟 앙상블 3구도 UI 패널
                        st.subheader("🤖 V6 앙상블 분석")
                        cA, cB, cC = st.columns(3)
                        cA.metric("GRU", f"{gru_prob*100:.1f}%")
                        cB.metric("LGBM", f"{lgb_prob*100:.1f}%")
                        cC.metric("최종", f"{final_prob*100:.1f}%")
                        
                        st.markdown("---")
                        st.write(f"📊 **실시간 수급 (비중)**")
                        st.write(f"- 외국인: {f_r * 100:+.2f}%")
                        st.write(f"- 기  관: {i_r * 100:+.2f}%")
                        st.markdown("---")
                        st.write(f"🎯 **트레이딩 가이드**\n- 익절(+4%): {int(curr_p * 1.04):,}원\n- 손절(-3%): {int(curr_p * 0.97):,}원")

                    st.markdown("---")
                    st.subheader(f"📰 {name} 최신 뉴스")
                    for n in get_top10_news(name): st.markdown(f"• [{n['title']}]({n['link']})")
                else:
                    st.error("데이터가 부족하여 분석할 수 없습니다. (신규 상장 등)")
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")

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
    st.info(f"📅 pykrx 히스토리컬 수급 데이터 연동 중 (24피처 앙상블 분석)")
    if model_gru is None or model_lgb is None:
        st.error(f"❌ AI 모델 파일을 불러오지 못했습니다. 경로를 확인해주세요: \n`{GRU_PATH}`\n`{LGB_PATH}`")
        st.stop()
        
    m_type = st.radio("타겟 시장 선택", ["KOSDAQ", "KOSPI"], horizontal=True)
    if st.button(f"🚀 {m_type} 시총 Top 100 타점 스캔"):
        with st.spinner("과거 수급 데이터와 매크로 지표를 분석 중입니다. 약 1~2분 소요됩니다..."):
            rank_df = get_v6_market_rankings(m_type, top_n=100)
            
        if rank_df.empty: st.error("⚠️ 데이터를 불러오지 못했습니다.")
        else:
            display_df = rank_df.copy()
            # 🌟 백테스트 기준에 맞춰 S급, A급 분리
            sniper_s_df = display_df[display_df['상승확률'] >= 70.0].sort_values("상승확률", ascending=False).reset_index(drop=True)
            sniper_a_df = display_df[(display_df['상승확률'] >= 60.0) & (display_df['상승확률'] < 70.0)].sort_values("상승확률", ascending=False).reset_index(drop=True)
            
            st.markdown("---")
            if not sniper_s_df.empty:
                st.success(f"🔥 **[S급] 70% 이상 초고도 확신 타점 ({len(sniper_s_df)}건)** - 승률 85% 구간")
                sniper_s_df['상승확률'] = sniper_s_df['상승확률'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(sniper_s_df, use_container_width=True)
            else:
                st.warning("🔥 오늘 장은 70% 이상 확신할 만한 S급 매수 타점이 없습니다.")
                
            if not sniper_a_df.empty:
                st.info(f"🚀 **[A급] 60% 이상 매수 우위 타점 ({len(sniper_a_df)}건)** - 승률 60% 구간")
                sniper_a_df['상승확률'] = sniper_a_df['상승확률'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(sniper_a_df, use_container_width=True)
                
            st.markdown("---")
            st.write("📋 전체 스캔 결과 (B급 이하 포함)")
            display_df['상승확률'] = display_df['상승확률'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_df.sort_values("상승확률", ascending=False).reset_index(drop=True), use_container_width=True)
