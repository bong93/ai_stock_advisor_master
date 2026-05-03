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
import urllib.parse
import ta
from sklearn.preprocessing import RobustScaler
from pykrx import stock
import joblib
import lightgbm as lgb
import time
import os
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import networkx as nx

# 🌟 최상단 배치 (Streamlit 설정)
st.set_page_config(page_title="AI Quant Radar V7.0", layout="wide")
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# 🌟 주린이(초보자)용 용어 해설 사전
JURIN_DICT = {
    "쌍끌이 매수": "외국인과 기관 투자자가 동시에 주식을 대량으로 사들이는 아주 강력한 호재 신호입니다.",
    "양매도": "외국인과 기관이 동시에 주식을 대량으로 팔고 있는 악재 신호입니다.",
    "과열권": "주가가 단기간에 너무 많이 올라 조만간 떨어질(조정받을) 가능성이 높은 위험한 상태입니다.",
    "낙폭 과대": "주가가 단기간에 너무 많이 떨어져서 조만간 다시 오를(반등할) 가능성이 높은 상태입니다.",
    "눌림목": "주가가 상승 추세에서 잠시 숨을 고르며 살짝 떨어지는 시점(좋은 매수 타이밍)입니다.",
    "수급": "시장에서 주식을 사려는 사람과 팔려는 사람 간의 힘 겨루기(자금 흐름)를 말합니다.",
    "센티먼트": "뉴스를 통해 느껴지는 사람들의 투자 심리나 분위기(호재/악재)를 뜻합니다."
}

def apply_jurin_help(text):
    for term, desc in JURIN_DICT.items():
        html_tag = f'<span title="{desc}" style="cursor: help; border-bottom: 2px dotted #00E676; color: #00E676; font-weight: bold;">{term}</span>'
        text = text.replace(term, html_tag)
    return text.replace('\n', '<br>')

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

# --- 2. 뉴스 감성 분석 (FinBERT) 엔진 ---
@st.cache_resource
def load_finbert_model():
    MODEL_NAME = "snunlp/KR-FinBERT-SC"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        return tokenizer, model
    except Exception: return None, None

fin_tokenizer, fin_model = load_finbert_model()

def get_google_news_titles(keyword, display=100, days=30):
    search_query = f"{keyword} 주식 when:{days}d"
    enc_keyword = urllib.parse.quote(search_query)
    url = f"https://news.google.com/rss/search?q={enc_keyword}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        soup = BeautifulSoup(res.content, 'xml')
        items = soup.find_all('item')
        results = []
        for item in items[:display]:
            title = item.title.get_text() if item.title else "제목 없음"
            link = item.link.get_text() if item.link else "#"
            pub_date_str = item.pubDate.get_text() if item.pubDate else ""
            if pub_date_str:
                try:
                    dt_utc = pd.to_datetime(pub_date_str, utc=True).tz_convert('Asia/Seoul')
                    date_str = dt_utc.strftime('%Y-%m-%d %H:%M')
                except: date_str = pub_date_str
            else: date_str = "날짜 미상"
            results.append({"title": title, "link": link, "date": date_str})
        return results
    except Exception: return []

def analyze_sentiment(text):
    if fin_model is None or fin_tokenizer is None: return 0.0
    inputs = fin_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = fin_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    return probs[2].item() - probs[0].item()

def get_news_sentiment_details(ticker_name, display=100):
    news_items = get_google_news_titles(ticker_name, display=display)
    if not news_items: return 0.0, []
    for item in news_items: item['score'] = analyze_sentiment(item['title'])
    return sum(item['score'] for item in news_items) / len(news_items), news_items

# --- 3. 보안 접속 ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "dlghdud121!":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else: st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🛡️ AI Quant Radar 접속 보안")
        st.text_input("보안 코드를 입력하고 Enter를 누르세요", type="password", on_change=password_entered, key="password")
        st.info("승인된 트레이더만 접속 가능합니다.")
        st.stop()
        return False
    elif not st.session_state["password_correct"]:
        st.error("❌ 보안 코드가 일치하지 않습니다.")
        st.stop()
        return False
    return True

# --- 4. 데이터 및 분석 엔진 ---
@st.cache_data(ttl=3600*24)
def get_all_stock_list():
    try:
        df_krx = fdr.StockListing('KRX')
        df_etf = fdr.StockListing('ETF/KR').rename(columns={'Symbol':'Code'})
        return pd.concat([df_krx['Name'] + " (" + df_krx['Code'] + ")", df_etf['Name'] + " (" + df_etf['Code'] + ")"]).dropna().tolist()
    except: return ["삼성전자 (005930)"]
    
@st.cache_resource
def load_ensemble_models(gru_path, lgb_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_gru = SwingBinaryMasterGRU(input_size=24)
    try:
        if not os.path.exists(gru_path) or not os.path.exists(lgb_path): return None, None, device
        model_gru.load_state_dict(torch.load(gru_path, map_location=device, weights_only=True))
        model_gru.to(device).eval()
        return model_gru, joblib.load(lgb_path), device
    except: return None, None, device

@st.cache_data(ttl=3600)
def get_macro_dashboard_data():
    indices = {"USD/KRW": "USD/KRW", "NASDAQ": "IXIC", "S&P500": "US500", "KOSPI": "KS11", "KOSDAQ": "KQ11", "VIX": "VIX"}
    data = {}
    for name, code in indices.items():
        try:
            df = fdr.DataReader(code).tail(2)
            data[name] = (df['Close'].iloc[-1], ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100)
        except: continue
    return data

@st.cache_data(ttl=3600*6)
def load_macro_feature_data():
    end_dt = datetime.today().strftime('%Y-%m-%d')
    start_dt = (datetime.today() - timedelta(days=500)).strftime('%Y-%m-%d')
    macro_df = pd.concat([
        fdr.DataReader('USD/KRW', start_dt, end_dt)['Close'].rename('usd_krw'),
        fdr.DataReader('IXIC', start_dt, end_dt)['Close'].rename('nasdaq'),
        fdr.DataReader('KS11', start_dt, end_dt)['Close'].rename('kospi'),
        fdr.DataReader('KQ11', start_dt, end_dt)['Close'].rename('kosdaq'),
        fdr.DataReader('VIX', start_dt, end_dt)['Close'].rename('vix')
    ], axis=1).ffill().dropna()
    for col in ['usd_krw', 'nasdaq', 'kospi', 'kosdaq', 'vix']: macro_df[f'{col}_ret'] = macro_df[col].pct_change()
    return macro_df[['usd_krw_ret', 'nasdaq_ret', 'kospi_ret', 'kosdaq_ret', 'vix_ret']]

def get_naver_supply_demand_history(code, pages=4):
    records = []
    for p in range(1, pages + 1):
        try:
            res = requests.get(f"https://finance.naver.com/item/frgn.naver?code={code}&page={p}", headers={'User-agent': 'Mozilla/5.0'}, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            tables = soup.find_all('table', class_='type2')
            if len(tables) < 2: continue
            for row in tables[1].find_all('tr'):
                cols = row.find_all('td')
                if len(cols) >= 9 and cols[0].text.strip():
                    try:
                        records.append({
                            'Date': pd.to_datetime(cols[0].text.strip().replace('.', '-')),
                            'inst_net': int(cols[5].text.strip().replace(',', '').replace('+', '')),
                            'foreigner_net': int(cols[6].text.strip().replace(',', '').replace('+', ''))
                        })
                    except: pass
        except: continue
    return pd.DataFrame(records).set_index('Date').sort_index() if records else pd.DataFrame()

def prepare_master_features(ticker, df_chart, macro_df):
    if len(df_chart) < 60: return pd.DataFrame(), None, 0, 0, None
    sd_df = get_naver_supply_demand_history(ticker, pages=4)
    df = df_chart.copy().join(macro_df, how='left')
    df = df.join(sd_df, how='left') if not sd_df.empty else df.assign(inst_net=0, foreigner_net=0)
    df.ffill(inplace=True); df.bfill(inplace=True)

    high_9, low_9 = df['High'].rolling(9).max(), df['Low'].rolling(9).min()
    high_26, low_26 = df['High'].rolling(26).max(), df['Low'].rolling(26).min()
    high_52, low_52 = df['High'].rolling(52).max(), df['Low'].rolling(52).min()
    
    tenkan, kijun = (high_9 + low_9) / 2, (high_26 + low_26) / 2
    span_a_raw, span_b_raw = (tenkan + kijun) / 2, (high_52 + low_52) / 2

    df_plot = pd.concat([df, pd.DataFrame(index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=26, freq='B'))])
    tmp_span_a = pd.Series(index=df_plot.index, dtype=float); tmp_span_a.loc[df.index] = span_a_raw
    tmp_span_b = pd.Series(index=df_plot.index, dtype=float); tmp_span_b.loc[df.index] = span_b_raw
    df_plot['senkou_span_a'], df_plot['senkou_span_b'] = tmp_span_a.shift(26), tmp_span_b.shift(26)
    df_plot['tenkan_sen'], df_plot['kijun_sen'] = tenkan, kijun

    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
    feats = pd.DataFrame(index=df.index)
    feats['ret'], feats['dist_ma'] = close.pct_change(), close / (close.rolling(20).mean() + 1e-8)
    feats['macd_hist'] = ta.trend.MACD(close).macd_diff()
    feats['adx'], feats['rsi'] = ta.trend.ADXIndicator(high, low, close).adx() / 100.0, ta.momentum.RSIIndicator(close).rsi() / 100.0
    feats['stoch'] = ta.momentum.StochasticOscillator(high, low, close).stoch() / 100.0
    feats['bb_pband'], feats['atr_pct'] = ta.volatility.BollingerBands(close).bollinger_pband(), ta.volatility.AverageTrueRange(high, low, close).average_true_range() / (close + 1e-8)
    feats['obv_ret'], feats['mfi'] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume().pct_change(), ta.volume.MFIIndicator(high, low, close, vol).money_flow_index() / 100.0
    feats['bb_width'], feats['cci'] = ta.volatility.BollingerBands(close).bollinger_wband() / 100.0, ta.trend.CCIIndicator(high, low, close).cci() / 100.0
    feats['roc'], feats['cmf'] = ta.momentum.ROCIndicator(close).roc() / 100.0, ta.volume.ChaikinMoneyFlowIndicator(high, low, close, vol).chaikin_money_flow()
    feats['will_r'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r() / -100.0
    
    feats['inst_ratio'], feats['foreigner_ratio'] = df['inst_net'] / (vol + 1e-8), df['foreigner_net'] / (vol + 1e-8)
    feats['inst_ratio_5d'] = df['inst_net'].rolling(5).sum() / (vol.rolling(5).sum() + 1e-8)
    feats['foreigner_ratio_5d'] = df['foreigner_net'].rolling(5).sum() / (vol.rolling(5).sum() + 1e-8)
    for col in ['usd_krw_ret', 'nasdaq_ret', 'kospi_ret', 'kosdaq_ret', 'vix_ret']: feats[col] = df[col]
    
    feats.replace([np.inf, -np.inf], np.nan, inplace=True); feats.dropna(inplace=True)
    
    feature_cols = ['ret', 'dist_ma', 'macd_hist', 'adx', 'rsi', 'stoch', 'bb_pband', 'atr_pct', 'obv_ret', 'mfi', 'bb_width', 'cci', 'roc', 'cmf', 'will_r', 'inst_ratio', 'foreigner_ratio', 'inst_ratio_5d', 'foreigner_ratio_5d', 'usd_krw_ret', 'nasdaq_ret', 'kospi_ret', 'kosdaq_ret', 'vix_ret']
    
    if len(feats) == 0: return feats, None, 0, 0, df_plot
    return feats[feature_cols], df.index[-1].strftime('%Y-%m-%d'), feats['inst_ratio'].iloc[-1], feats['foreigner_ratio'].iloc[-1], df_plot

@st.cache_data(ttl=3600)
def get_v6_market_rankings(market_type="KOSPI", top_n=50):
    try:
        df_list = fdr.StockListing(market_type)
        if market_type == "ETF/KR":
             tickers, names = df_list.sort_values('Volume', ascending=False).head(top_n)['Symbol'].tolist(), df_list.sort_values('Volume', ascending=False).head(top_n)['Name'].tolist()
        else:
             tickers, names = df_list.sort_values('Marcap', ascending=False).head(top_n)['Code'].tolist(), df_list.sort_values('Marcap', ascending=False).head(top_n)['Name'].tolist()
        ticker_name_map = dict(zip(tickers, names))
    except: return pd.DataFrame()
    
    results = []
    prog = st.progress(0)
    macro_df = load_macro_feature_data()
    for i, ticker in enumerate(tickers):
        try:
            df_chart = fdr.DataReader(ticker, (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d'))
            feats_df, _, _, _, _ = prepare_master_features(ticker, df_chart, macro_df)
            if feats_df.empty or len(feats_df) < 60: continue
            
            scaled_feat = RobustScaler().fit_transform(feats_df.tail(60).values)
            inp = torch.FloatTensor(scaled_feat).unsqueeze(0).to(device)
            with torch.no_grad(): gru_prob = torch.softmax(model_gru(inp), dim=1).cpu().numpy()[0][1]
            lgb_prob = model_lgb.predict_proba(scaled_feat[-1].reshape(1, -1))[0][1]
            base_prob = (gru_prob * 0.5 + lgb_prob * 0.5) * 100
            
            t_name = ticker_name_map[ticker]
            news_score, _ = (0.0, []) if market_type == "ETF/KR" else get_news_sentiment_details(t_name, display=15)
            final_prob = max(0.0, min(100.0, base_prob + (news_score * 5.0)))
            
            results.append({"종목명": t_name, "코드": ticker, "현재가": int(df_chart['Close'].iloc[-1]), "기본확률(AI)": base_prob, "뉴스점수": news_score, "최종확률": final_prob})
        except: continue
        prog.progress((i + 1) / len(tickers))
    prog.empty()
    return pd.DataFrame(results)

def generate_ai_briefing(name, base_prob, news_score, final_prob, for_ratio, inst_ratio, rsi, stoch, valid_date, news_items=None):
    date_str = f"({valid_date} 장마감 기준)" if valid_date else "(수급 정보 없음)"
    briefing = f"[{name} 트레이딩 브리핑] {date_str}\n\n"
    briefing += f"AI 기본 판단: 차트와 수급을 분석한 기술적 상승 확률은 {base_prob:.1f}% 입니다. "
    
    news_impact = news_score * 5.0
    if news_score > 0.3: briefing += f"여기에 최근 시장의 호재 뉴스가 반영되어 +{news_impact:.1f}%의 강력한 확률 가산점을 받았습니다.\n\n"
    elif news_score < -0.3: briefing += f"다만, 최근 발생한 악재 뉴스로 인해 시장 센티먼트가 위축되어 {news_impact:.1f}%의 확률 감점이 발생했습니다.\n\n"
    else: briefing += f"최근 두드러진 호재나 악재 뉴스가 없어 감성 점수는 중립({news_score:+.2f}점)을 유지 중입니다.\n\n"

    if news_items:
        pos_news = max(news_items, key=lambda x: x['score'])
        neg_news = min(news_items, key=lambda x: x['score'])
        if pos_news['score'] >= 0.25 or neg_news['score'] <= -0.25:
            briefing += f"🗣️ AI 이슈 요약:**\n"
            if pos_news['score'] >= 0.25: briefing += f"- 🔥 강력한 호재: [{pos_news['title']}]\n"
            if neg_news['score'] <= -0.25: briefing += f"- 🛑 주의할 악재: [{neg_news['title']}]\n"
            briefing += "\n"

    briefing += f"🎯 종합 타점 등급: "
    if final_prob >= 70: briefing += f"🔥 [S급] 초고도 확신 구간 ({final_prob:.1f}%):** 차트와 재료(뉴스)가 완벽히 일치하는 강력한 매수 타이밍입니다.\n\n"
    elif final_prob >= 60: briefing += f"🚀 [A급] 강한 확신 구간 ({final_prob:.1f}%):** 상승 에너지가 긍정적인 매수 우위 자리입니다.\n\n"
    elif final_prob >= 50: briefing += f"✅ [B급] 관망 권장 구간 ({final_prob:.1f}%):** 반반의 확률을 가진 애매한 구간입니다.\n\n"
    else: briefing += f"🛑 [패스] 매수 금지 구간 ({final_prob:.1f}%):** 하방 압력이 강해 관망을 권장합니다.\n\n"

    if for_ratio > 0.001 and inst_ratio > 0.001: briefing += "💡 수급/기술적 코멘트: 현재 외국인과 기관의 쌍끌이 매수가 유입 중이며, "
    elif for_ratio > 0.001: briefing += "💡 수급/기술적 코멘트: 외국인 자금이 유입되며 하방을 방어 중이며, "
    elif inst_ratio > 0.001: briefing += "💡 수급/기술적 코멘트: 기관의 저가 매수세가 들어오는 가운데, "
    elif for_ratio < -0.001 and inst_ratio < -0.001: briefing += "💡 수급/기술적 코멘트: 현재 메이저 양매도가 출회 중이므로 접근에 주의해야 하며, "
    else: briefing += "💡 수급/기술적 코멘트: 메이저 수급의 뚜렷한 특징은 없으며, "
    
    if rsi > 0.7 or stoch > 0.8: briefing += "차트가 단기 과열권에 진입했습니다. 추격 매수보다는 눌림목을 대기하세요."
    elif rsi < 0.3 or stoch < 0.2: briefing += "단기 낙폭 과대 구간입니다. 기술적 반등을 노린 분할 매수가 유효합니다."
    else: briefing += "기술적 지표는 안정적인 적정 구간에 위치해 있습니다."
    return briefing

def draw_ichimoku_chart(df_plot):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['senkou_span_a'], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['senkou_span_b'], line=dict(width=0), fill='tonexty', fillcolor='rgba(150, 150, 150, 0.2)', name='Kumo Cloud'))
    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['tenkan_sen'], line=dict(color='orange', width=1), name='전환선'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['kijun_sen'], line=dict(color='dodgerblue', width=1), name='기준선'))
    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, xaxis=dict(type='date', range=[df_plot.index[-146], df_plot.index[-1]]), margin=dict(l=10, r=10, t=30, b=10))
    return fig

# 🌟 [신규 기능 1] 자금 흐름 네트워크 맵 그리기
def draw_correlation_network(market="KOSPI", top_n=30):
    try:
        df_list = fdr.StockListing(market)
        tickers = df_list.sort_values('Marcap', ascending=False).head(top_n)['Code'].tolist()
        names = df_list.sort_values('Marcap', ascending=False).head(top_n)['Name'].tolist()
        t_map = dict(zip(tickers, names))
        
        # 최근 60일 데이터 수집 및 일일 수익률 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        df_prices = pd.DataFrame()
        
        prog = st.progress(0)
        for i, t in enumerate(tickers):
            try:
                p = fdr.DataReader(t, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))['Close']
                df_prices[t_map[t]] = p.pct_change()
            except: pass
            prog.progress((i+1)/len(tickers))
        prog.empty()
        
        df_prices.dropna(inplace=True)
        corr_matrix = df_prices.corr()
        
        # 네트워크 그래프 생성 (상관계수 0.5 이상 또는 -0.5 이하만 연결)
        G = nx.Graph()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= 0.5:
                    G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr)
                    
        # 노드 레이아웃 설정
        pos = nx.spring_layout(G, k=0.5, seed=42)
        
        edge_x, edge_y, edge_colors = [], [], []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            # 양의 상관관계는 빨간색(동조화), 음의 상관관계는 파란색(디커플링)
            edge_colors.append('rgba(255,50,50,0.5)' if edge[2]['weight'] > 0 else 'rgba(50,50,255,0.5)')
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
        
        node_x, node_y, text_labels = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x); node_y.append(y); text_labels.append(node)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=text_labels, textposition="top center",
            hoverinfo='text', marker=dict(showscale=True, colorscale='YlGnBu', size=20,
            color=[G.degree(n) for n in G.nodes()], line_width=2))
            
        fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(text=f'🕸️ {market} 시총 상위 {top_n} 자금 흐름 동조화 네트워크 (최근 60일)', font=dict(size=16)),
                showlegend=False, hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template='plotly_dark'
             ))
        return fig
    except Exception as e:
        st.error(f"네트워크 맵 생성 중 오류: {e}")
        return go.Figure()

# --- 5. 메인 실행부 ---
GRU_PATH = r"D:\KOSPI_KOSDAK_DAYTRAIDER_AI_PRJ\model_output\weather_advisor_v6_master_D.pt"
LGB_PATH = r"D:\KOSPI_KOSDAK_DAYTRAIDER_AI_PRJ\model_output\weather_advisor_v6_master_D_lgb.pkl"
model_gru, model_lgb, device = load_ensemble_models(GRU_PATH, LGB_PATH)

if check_password():
    idx_data = get_macro_dashboard_data()
    st.subheader("한국/미국 주요 지표")
    
    macro_helps = {
        "USD/KRW": "원/달러 환율입니다. 오르면 외국인 자금이 빠져나갈 우려가 있습니다.",
        "NASDAQ": "미국 기술주 중심의 나스닥 지수입니다. 한국 반도체/IT 주가에 큰 영향을 줍니다.",
        "S&P500": "미국 500곳의 대기업이 포함된 실질적인 간판 지수로써 대표 주가 지수로 불립니다.",
        "KOSPI": "대한민국 유가증권시장의 종합 주가 지수이며, 제 1시장입니다.",
        "KOSDAQ": "대한민국의 벤처기업이 몰려있는 종합 주가 지수이며, 제 2시장입니다.",
        "VIX": "미국 공포 지수입니다. 지수가 급등하면 미국 시장의 불안정성을 의미하며, 이는 한국 시장의 하락 및 변동성 확대로 해석됩니다."
    }
    idx_cols = st.columns(len(idx_data))
    for i, (k, v) in enumerate(idx_data.items()):
        color = "inverse" if k in ["USD/KRW", "VIX"] else "normal"
        idx_cols[i].metric(k, f"{v[0]:,.2f}", f"{v[1]:+.2f}%", delta_color=color, help=macro_helps.get(k, "글로벌 매크로 지표입니다."))
    st.markdown("---") # 🌟 매크로와 메인 콘텐츠를 구분하는 선
    
    # 🌟 메뉴에 네트워크 맵 추가
    menu = st.sidebar.radio("모드 선택", ["단일 종목 스캐너", "스윙 타점 스캐너", "자금 흐름 네트워크 맵", "ETF 스캐너"], index=0)

if menu == "단일 종목 스캐너":
    st.title("단일 종목 스캐너 (AI + News + 시뮬레이터)")
    
    all_stocks = get_all_stock_list()
    default_idx = all_stocks.index("삼성전자 (005930)") if "삼성전자 (005930)" in all_stocks else 0
    selected_item = st.sidebar.selectbox("종목 검색 (초성/이름/코드 입력)", options=all_stocks, index=default_idx)
    
    if model_gru is None or model_lgb is None: st.error(f"❌ AI 모델 로드 실패. 파일 경로를 확인하세요.")
    else:
        import re
        match = re.match(r"(.*) \((.*)\)", selected_item)
        if match: name, ticker = match.group(1), match.group(2)
        else: name, ticker = "삼성전자", "005930"
        
        with st.spinner(f"[{name}] 데이터 분석 중..."):
            try:
                df_chart = fdr.DataReader(ticker, (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d'))
                macro_df = load_macro_feature_data()
                feats_df, v_date, i_r, f_r, df_plot = prepare_master_features(ticker, df_chart, macro_df)
                
                if not feats_df.empty and len(feats_df) >= 60:
                    curr_p, prev_p = df_chart['Close'].iloc[-1], df_chart['Close'].iloc[-2]
                    st.metric(label=f"{name} ({ticker})", value=f"{int(curr_p):,}원", delta=f"{int(curr_p-prev_p):+}원 ({(curr_p-prev_p)/prev_p*100:+.2f}%)")
                    
                    briefing_container = st.container()
                    
                    # 🌟 [신규 기능 2] 매크로 스트레스 테스트 시뮬레이터 UI
                    with st.expander("🎛️ 매크로 스트레스 테스트 (What-If 시뮬레이터)", expanded=False):
                        st.info("만약 오늘 밤 나스닥이 폭락하거나 환율이 치솟는다면, 이 종목의 내일 상승 확률은 어떻게 변할지 테스트해보세요.")
                        col_s1, col_s2, col_s3 = st.columns(3)
                        sim_nasdaq = col_s1.slider("🇺🇸 나스닥 변동 (%)", -5.0, 5.0, 0.0, 0.5,
                                                   help="간밤에 미국 기술주(나스닥)가 폭락하거나 폭등했을 때, 다음 날 해당 종목에 미칠 충격을 시뮬레이션합니다.")
                        sim_usdkrw = col_s2.slider("💵 환율 변동 (%)", -3.0, 3.0, 0.0, 0.1,
                                                   help="원/달러 환율이 급등(원화 가치 하락)하면 외국인 자금 이탈 우려가 커져 증시에 악재로 작용하는 경향이 있습니다.")
                        sim_vix = col_s3.slider("😨 VIX 공포지수 변동 (%)", -20.0, 20.0, 0.0, 1.0,
                                                help="시장의 공포지수입니다. VIX가 치솟으면 전 세계적인 투자 심리가 얼어붙어 증시에 강한 하방 압력을 줍니다.")
                    
                    # 시뮬레이션 데이터 복사 및 변동치 적용 (마지막 날짜 데이터 강제 수정)
                    sim_feats_df = feats_df.copy()
                    sim_feats_df.loc[sim_feats_df.index[-1], 'nasdaq_ret'] += (sim_nasdaq / 100.0)
                    sim_feats_df.loc[sim_feats_df.index[-1], 'usd_krw_ret'] += (sim_usdkrw / 100.0)
                    sim_feats_df.loc[sim_feats_df.index[-1], 'vix_ret'] += (sim_vix / 100.0)

                    # 시뮬레이션 적용된 데이터로 스케일링 및 AI 예측
                    scaled_feat = RobustScaler().fit_transform(sim_feats_df.tail(60).values)
                    inp = torch.FloatTensor(scaled_feat).unsqueeze(0).to(device)
                    
                    with torch.no_grad(): gru_prob = torch.softmax(model_gru(inp), dim=1).cpu().numpy()[0][1]
                    lgb_prob = model_lgb.predict_proba(scaled_feat[-1].reshape(1, -1))[0][1]
                    base_prob_pct = ((gru_prob * 0.5) + (lgb_prob * 0.5)) * 100
                    
                    sentiment_score, news_items = get_news_sentiment_details(name, display=100)
                    news_impact = sentiment_score * 5.0
                    final_prob_pct = max(0.0, min(100.0, base_prob_pct + news_impact))
                    
                    raw_briefing = generate_ai_briefing(name, base_prob_pct, sentiment_score, final_prob_pct, f_r, i_r, feats_df['rsi'].iloc[-1], feats_df['stoch'].iloc[-1], v_date, news_items)
                    briefing_html = apply_jurin_help(raw_briefing)
                    
                    with briefing_container:
                            st.markdown(f"""
                            <div style="background-color: rgba(0, 230, 118, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #00E676; margin-bottom: 20px; line-height: 1.6;">
                                {briefing_html}
                            </div>
                            """, unsafe_allow_html=True)

                    col1, col2 = st.columns([2, 1])
                    with col1: st.plotly_chart(draw_ichimoku_chart(df_plot), use_container_width=True)
                    
                    with col2:
                        # 시뮬레이터 가동 여부에 따라 제목 변경
                        is_simulated = sim_nasdaq != 0 or sim_usdkrw != 0 or sim_vix != 0
                        title_prefix = "🔬 [시뮬레이션 적용됨]" if is_simulated else "1차: 2 AI 앙상블 기본 판단"
                        
                        st.subheader(title_prefix, help="수만 개의 과거 차트 패턴과 메이저 수급(외국인/기관) 데이터를 바탕으로 산출된 순수 기술적 상승 확률입니다. 시뮬레이터를 켜면 매크로 변동성이 반영됩니다.")
                        cA, cB, cC = st.columns(3)
                        cA.metric("GRU", f"{gru_prob*100:.1f}%", help="과거 차트 패턴을 기억하고 미래를 예측하는 딥러닝 AI입니다.")
                        cB.metric("LGBM", f"{lgb_prob*100:.1f}%", help="수급, 거래량 데이터를 분석하는 머신러닝 AI입니다.")
                        cC.metric("기본 확률", f"{base_prob_pct:.1f}%", help="매크로 변동이 반영된 순수 상승 확률입니다.")
                        
                        st.markdown("---")
                        st.subheader("📰 2차: 뉴스 센티먼트 융합", help="최근 한 달간 보도된 관련 뉴스 100개를 AI가 읽고 문맥의 긍정/부정을 판별하여 최종 상승 확률에 가중치를 부여합니다.")
                        cD, cE, cF = st.columns(3)
                        news_emoji = "🔥" if sentiment_score > 0 else ("🛑" if sentiment_score < 0 else "➖")
                        cD.metric(f"뉴스 ({news_emoji})", f"{sentiment_score:+.2f}점", help="-1(극단적 악재)부터 +1(극단적 호재)까지의 수치입니다.")
                        cE.metric("가산점", f"{news_impact:+.1f}%p", help="뉴스 점수에 따라 최종 확률에 더해지는 가중치입니다.")
                        
                        # 시뮬레이션 시 색상 변화로 강조
                        delta_str = "시뮬레이션!" if is_simulated else None
                        cF.metric("최종 확신도", f"{final_prob_pct:.1f}%", delta=delta_str, delta_color="inverse")
                        
                        st.markdown("---")
                        st.write(f"📊 실시간 수급 (비중)")
                        st.write(f"- 외국인: {f_r * 100:+.2f}%")
                        st.write(f"- 기  관: {i_r * 100:+.2f}%")

                    st.markdown("---")
                    st.subheader(f"📰 {name} 주요 최신 뉴스 (표본 100개 중 최신 10개)")
                    if news_items:
                        for n in news_items[:10]: 
                            score = n['score']
                            emoji = "🔥" if score > 0.3 else ("🛑" if score < -0.3 else "➖")
                            st.markdown(f"• `{n['date']}` | {emoji} **[{score:+.2f}점]** [{n['title']}]({n['link']})")
                    else: st.write("최근 30일간 검색된 뉴스가 없습니다.")
                else: st.error("데이터가 부족하여 분석할 수 없습니다.")
            except Exception as e: st.error(f"분석 중 오류 발생: {e}")

# 🌟 [신규 메뉴 추가] 자금 흐름 네트워크 맵
elif menu == "자금 흐름 네트워크 맵":
    st.title("시총 상위 자금 흐름 네트워크")
    st.info("KOSPI/KOSDAQ 시장의 대형주들이 어떻게 묶여서 같이 오르고 내리는지 상관관계를 시각화합니다. (연결선이 굵고 많을수록 시장의 주도 테마입니다.)")
    
    m_type = st.radio("타겟 시장 선택", ["KOSPI", "KOSDAQ"], horizontal=True)
    if st.button("네트워크 맵 분석 시작 (약 10초 소요)"):
        with st.spinner(f"{m_type} 시총 상위 50개 종목의 최근 60일 상관관계를 분석 중입니다..."):
            fig = draw_correlation_network(market=m_type, top_n=50)
            st.plotly_chart(fig, use_container_width=True)

elif menu == "ETF 스캐너":
    st.title("ETF 레이더")
    st.write("시장 전체의 자금 흐름과 분위기를 파악합니다.")
    st.markdown("---")
    st.subheader("🔥 AI 거래대금 상위 ETF 방향성 스캔")
    if st.button("ETF 스캔 시작 (Top 20)"):
        with st.spinner("거래대금 상위 ETF 타점 분석 중..."):
            etf_df = get_v6_market_rankings("ETF/KR", top_n=20)
            if not etf_df.empty:
                display_df = etf_df.copy()
                display_df['최종확률'] = display_df['최종확률'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df[['종목명', '코드', '현재가', '최종확률']].sort_values("최종확률", ascending=False).reset_index(drop=True), use_container_width=True)

elif menu == "스윙 타점 스캐너":
    st.title("저격수 스캐너 (AI + News)")
    st.info(f"📅 30개 종목의 기술적 타점과 뉴스 센티먼트(호재/악재)를 동시에 스캔합니다.")
    if model_gru is None or model_lgb is None:
        st.error(f"❌ AI 모델 파일을 불러오지 못했습니다.")
        st.stop()
        
    m_type = st.radio("타겟 시장 선택", ["KOSDAQ", "KOSPI"], horizontal=True)
    if st.button(f"🚀 {m_type} 시총 Top 30 타점 스캔 (약 2~3분 소요)"):
        with st.spinner("과거 수급 데이터와 실시간 구글 뉴스를 융합 분석 중입니다..."):
            rank_df = get_v6_market_rankings(m_type, top_n=30)
            
        if rank_df.empty: st.error("⚠️ 데이터를 불러오지 못했습니다.")
        else:
            display_df = rank_df.copy()
            format_df = display_df.copy()
            format_df['기본확률(AI)'] = format_df['기본확률(AI)'].apply(lambda x: f"{x:.1f}%")
            format_df['뉴스점수'] = format_df['뉴스점수'].apply(lambda x: f"{x:+.2f}점")
            format_df['최종확률'] = format_df['최종확률'].apply(lambda x: f"{x:.1f}%")
            
            sniper_s_df = format_df[display_df['최종확률'] >= 70.0].sort_values(by=display_df['최종확률'].name, ascending=False).reset_index(drop=True)
            sniper_a_df = format_df[(display_df['최종확률'] >= 60.0) & (display_df['최종확률'] < 70.0)].sort_values(by=display_df['최종확률'].name, ascending=False).reset_index(drop=True)
            
            st.markdown("---")
            if not sniper_s_df.empty:
                st.success(f"🔥 **[S급] 70% 이상 초고도 확신 타점 ({len(sniper_s_df)}건)**")
                st.dataframe(sniper_s_df, use_container_width=True)
            else: st.warning("🔥 오늘 장은 70% 이상 확신할 만한 S급 매수 타점이 없습니다.")
                
            if not sniper_a_df.empty:
                st.info(f"🚀 **[A급] 60% 이상 매수 우위 타점 ({len(sniper_a_df)}건)**")
                st.dataframe(sniper_a_df, use_container_width=True)
                
            st.markdown("---")
            st.write("📋 전체 스캔 결과 (B급 이하 포함)")
            st.dataframe(format_df.sort_values(by=display_df['최종확률'].name, ascending=False).reset_index(drop=True), use_container_width=True)
