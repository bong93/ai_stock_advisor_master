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
st.set_page_config(page_title="AI Quant Master", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# 🌟 [신규] 모바일 반응형 UI 강제 최적화 (CSS Injection)
st.markdown("""
<style>
    /* 1. 모바일 환경에서 탭(Tab) 버튼이 한 줄로 예쁘게 나오도록 수정 */
    div[data-baseweb="tab-list"] {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }
    
    /* 2. 스마트폰 세로 모드일 때 메트릭(숫자) 크기 자동 조절 */
    @media (max-width: 600px) {
        h1 { font-size: 24px !important; }
        h2 { font-size: 20px !important; }
        h3 { font-size: 18px !important; }
        div[data-testid="stMetricValue"] { font-size: 22px !important; }
        
        /* 3. 모바일에서 데이터프레임 폰트 및 여백 최소화 (렌더링 속도 개선) */
        .dataframe { font-size: 12px !important; }
        .stDataFrame { padding: 0 !important; }
        
        /* 4. 모바일에서 버튼을 화면 꽉 차게 만들어 터치하기 쉽게 변경 */
        .stButton>button { width: 100% !important; height: 50px !important; font-size: 16px !important; font-weight: bold !important; }
    }
</style>
""", unsafe_allow_html=True)

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

def generate_unified_market_briefing(up_df, th_df):
    if up_df.empty or th_df.empty:
        return "데이터가 부족하여 종합 브리핑을 생성할 수 없습니다."

    # 평균치 계산
    up_avg = up_df['등락률'].mean()
    th_avg = th_df['등락률'].mean()

    # 상/하위 데이터 추출
    top_up = up_df.iloc[0]
    bot_up = up_df.iloc[-1]
    top_th = th_df.iloc[0]

    # 🌟 AI 시장 상태 판별 로직 (업종과 테마의 상관관계 분석)
    if up_avg > 0.3 and th_avg > 0.3:
        status, icon = "전방위 강세장 (Risk-On)", "🔥"
        desc = "업종의 큰 돈(기관/외인)과 테마의 빠른 돈(개인)이 모두 상승을 가리키고 있습니다. 주도주 중심의 적극적인 비중 확대가 유효합니다."
    elif up_avg < -0.3 and th_avg < -0.3:
        status, icon = "전방위 약세장 (Risk-Off)", "❄️"
        desc = "거시적(업종) 하방 압력과 투심(테마) 악화가 겹쳤습니다. 현금 비중을 높이고 하락 방어력이 좋은 대형주 위주로 짧게 대응하십시오."
    elif up_avg > 0 and th_avg <= 0:
        status, icon = "실적/대형주 주도장", "🏢"
        desc = "테마성 투기 자금은 빠지고 있으나, 굵직한 업종 사이클은 버티고 있습니다. 펀더멘털이 튼튼한 우량주 중심의 시장입니다."
    else: 
        status, icon = "테마/개별주 장세 (순환매)", "🎯"
        desc = "시장 전체의 지수(업종)는 부진하나, 특정 테마로 돈이 쏠리고 있습니다. 지수보다는 이슈 중심의 트레이딩(단타/스윙)이 유리합니다."

    briefing = f"""
<div style="background-color: #1E1E2E; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #FF4B4B;">
<h3 style="margin-top: 0;">🧠 AI 머니플로우 종합 피드백</h3>
현재 시장은 <strong>{icon} {status}</strong> 구간입니다. <br><span style="color: #A0A0B0; font-size: 14px;">{desc}</span>

<hr style="border-color: #333;">

<h4 style="margin-bottom: 10px;">📊 1. 시장 온도 비교 (업종 vs 테마)</h4>
<ul style="margin-bottom: 15px;">
    <li><strong>업종 평균 (큰 파도):</strong> <span style="color: {'#FF4B4B' if up_avg > 0 else '#1C83E1'}; font-weight: bold;">{up_avg:+.2f}%</span> (현재 주도: <strong>{top_up['이름']}</strong> <code>{top_up['등락률']:+.2f}%</code>)</li>
    <li><strong>테마 평균 (빠른 물결):</strong> <span style="color: {'#FF4B4B' if th_avg > 0 else '#1C83E1'}; font-weight: bold;">{th_avg:+.2f}%</span> (현재 주도: <strong>{top_th['이름']}</strong> <code>{top_th['등락률']:+.2f}%</code>)</li>
</ul>

<h4 style="margin-bottom: 10px;">💡 2. 트레이딩 전략 인사이트</h4>
<ul style="margin-bottom: 0;">
    <li><strong>🎯 롱(Long) 전략: 현재 시장의 큰 자금은 [{top_up['이름']}] 업종으로, 빠른 단기 자금은 [{top_th['이름']}] 테마로 쏠리고 있습니다. 우량주 스윙을 원하신다면 업종 대장주를, 단기 변동성을 노리신다면 테마 대장주 중 AI 타점이 높은 종목을 분리하여 공략하십시오.
    <li><strong>⚠️ 숏(Short) 회피:</strong> <strong>[{bot_up['이름']}]</strong> 업종은 현재 자금 이탈이 가장 심각합니다. (<code>{bot_up['등락률']:+.2f}%</code>). 해당 섹터의 매매는 당분간 보류하십시오.</li>
</ul>
</div>
"""
    return briefing

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

@st.cache_data(ttl=3600*24)
def get_stock_tags_mapping():
    """KRX 종목 상세 정보를 이용해 업종과 테마(주요제품) 꼬리표를 생성합니다."""
    try:
        df_desc = fdr.StockListing('KRX-DESC')
        mapping = {}
        for _, row in df_desc.iterrows():
            sector = str(row.get('Sector', '')).strip()
            industry = str(row.get('Industry', '')).strip()
            
            if sector == 'nan' or not sector: sector = "분류없음"
            if industry == 'nan' or not industry: industry = "특징없음"
            
            # 너무 긴 테마(Industry) 텍스트는 가독성을 위해 자름
            if len(industry) > 15:
                industry = industry[:15] + "..."
                
            mapping[row['Code']] = f"🏢 {sector} | 🏷️ {industry}"
        return mapping
    except:
        return {}
    
# --- [신규/수정] 네이버 금융 직접 크롤링 엔진 ---
@st.cache_data(ttl=1800)
def get_naver_market_data(group_type="upjong", count=50):
    """
    업종/테마별 상세 페이지를 정밀 파싱하여 
    최고 상승률(1등)과 최저 상승률(꼴등)을 동시에 찾습니다.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
    
    data = []
    page = 1
    seen_codes = set() # 🌟 중복 검사용 저장소
    
    # 1. 리스트 수집
    while True:
        if group_type == "upjong":
            list_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
        else:
            list_url = f"https://finance.naver.com/sise/theme.naver?page={page}"
        
        try:
            res = requests.get(list_url, headers=headers, timeout=10)
            res.encoding = 'euc-kr'
            soup = BeautifulSoup(res.text, 'html.parser')
            
            table = soup.select_one('table.type_1')
            if not table: break

            rows = table.select('tr')
            page_item_count = 0
            is_duplicate_page = False # 🌟 중복 페이지 탈출 플래그
            
            for row in rows:
                cols = row.select('td')
                if len(cols) >= 2:
                    link_tag = cols[0].find('a')
                    if link_tag and 'no=' in link_tag.get('href', ''):
                        code = link_tag['href'].split('no=')[-1].split('&')[0] 
                        
                        # 🌟 이미 수집한 코드라면, 마지막 페이지를 넘어서 중복 페이지를 돌고 있는 것!
                        if code in seen_codes:
                            is_duplicate_page = True
                            break
                            
                        seen_codes.add(code) # 새로운 코드는 저장소에 등록
                        
                        name = link_tag.text.strip()
                        change_text = cols[1].text.strip().replace('%', '').replace('+', '')
                        try: change_val = float(change_text)
                        except: change_val = 0.0
                        
                        data.append({"이름": name, "등락률": change_val, "code": code})
                        page_item_count += 1
            
            # 업종이거나, 데이터가 없거나, 중복 페이지가 감지되면 즉시 루프 탈출
            if group_type == "upjong" or page_item_count == 0 or is_duplicate_page:
                break
                
            if page > 15: # 혹시 모를 록인(Lock-in) 방지
                break
                
            page += 1
            
        except Exception as e:
            break

    full_df = pd.DataFrame(data).sort_values("등락률", ascending=False).reset_index(drop=True)
    
    # 2. 상세 페이지 크롤링
    final_list = []
    for i in range(min(count, len(full_df))):
        row = full_df.iloc[i]
        detail_url = f"https://finance.naver.com/sise/sise_group_detail.naver?type={group_type}&no={row['code']}"
        
        try:
            d_res = requests.get(detail_url, headers=headers, timeout=5)
            d_res.encoding = 'euc-kr'
            d_soup = BeautifulSoup(d_res.text, 'html.parser')
            
            stock_table = d_soup.select_one('table.type_5')
            max_change, min_change = -999.0, 999.0
            top_name, bottom_name = "-", "-"
            
            target_td_idx = 3 if group_type == "upjong" else 4
            
            if stock_table:
                stock_rows = stock_table.select('tr')
                for s_row in stock_rows:
                    name_cell = s_row.select_one('td.name a')
                    tds = s_row.select('td')
                    
                    if name_cell and len(tds) > target_td_idx:
                        s_name = name_cell.text.strip()
                        change_text = tds[target_td_idx].text.strip().replace('%', '').replace('+', '').replace(',', '')
                        if not change_text: continue 
                        
                        try:
                            s_change = float(change_text)
                            if s_change > max_change:
                                max_change = s_change
                                top_name = s_name
                            if s_change < min_change:
                                min_change = s_change
                                bottom_name = s_name
                        except:
                            continue
                            
            if max_change == -999.0: max_change = 0.0
            if min_change == 999.0: min_change = 0.0
                
            final_list.append({
                "이름": row['이름'],
                "등락률": row['등락률'],
                "1등주(대장)": top_name,
                "1등 수익률": max_change,
                "꼴등주(부진)": bottom_name,
                "꼴등 수익률": min_change
            })
            
        except Exception:
            continue

    detail_df = pd.DataFrame(final_list)
    return full_df, detail_df
    
def generate_live_sector_briefing(df, g_type="업종"):
    if df.empty: return "현재 분석 가능한 데이터가 없습니다."
    
    top = df.iloc[0]
    bottom = df.iloc[-1]
    avg_change = df['등락률'].mean()
    
    # 시장 온도 상태 판별
    if avg_change > 0.5:
        status_icon, status_text = "🔥", "강세 (불장)"
        status_desc = "대부분의 산업에 돈이 유입되고 있는 활기찬 상태입니다. 적극적인 종목 발굴이 유리합니다."
    elif avg_change < -0.5:
        status_icon, status_text = "❄️", "약세 (냉장고)"
        status_desc = "시장 전체의 엔진이 식어가는 중입니다. 무리한 매수보다는 현금 비중을 늘리고 관망할 때입니다."
    else:
        status_icon, status_text = "☁️", "혼조세 (안개)"
        status_desc = "오르는 곳과 내리는 곳이 팽팽합니다. 방향성이 정해질 때까지 방망이를 짧게 잡아야 합니다."
    
    briefing = f"### 🚀 실시간 {g_type} 수급 브리핑\n\n"
    briefing += f"현재 시장의 {g_type} 기류는 평균 **{avg_change:+.2f}%** 변동하며 **{status_icon} {status_text}**를 기록 중입니다.\n\n"
    
    # 상세 분석 카드 (HTML/Markdown 활용)
    briefing += f"""
---
#### 🌡️ 시황 온도계: "이 수치는 어떤 의미인가요?"
* **평균 변동률 ({avg_change:+.2f}%):** {status_desc}
* **주도 섹터 ({top['이름']}):** 남들보다 **{top['등락률'] - avg_change:+.2f}%p** 더 강한 에너지를 보입니다. 현재 시장의 '주인공'입니다.
* **하락 섹터 ({bottom['이름']}):** 시장의 하락세보다 훨씬 더 깊게 눌리고 있습니다. 자금이 빠르게 이탈 중인 '위험지역'입니다.

#### 💡 주린이를 위한 투자 가이드
1. **상대적 강세에 주목:** 전체 평균({avg_change:+.2f}%)보다 높은 수익률을 기록 중인 섹터는 하락장에서도 누군가 계속 사고 있다는 증거입니다.
2. **엇박자 주의:** 평균은 마이너스인데 혼자 폭등하는 섹터는 '테마성 급등'일 확률이 높으니 추격 매수에 주의하세요.
3. **바닥 확인:** 하락 섹터가 며칠째 최하위라면, 투매가 끝나고 반등이 나올 '눌림목' 후보가 될 수 있습니다.
---
"""
    # 기존 특이사항 로직 유지
    if "조선" in top['이름'] or "운수장비" in top['이름']:
        briefing += "\n> 🔔 **특이사항:** 조선/중공업 사이클에 강한 수급이 포착되었습니다. 대형 수주 뉴스나 환율 효과를 점검하십시오."
    elif "건설" in top['이름']:
        briefing += "\n> 🔔 **특이사항:** 건설/인프라 섹터에 온기가 돌고 있습니다. 정책 변화나 금리 동향이 반영되었을 가능성이 높습니다."
        
    return briefing

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
    tag_map = get_stock_tags_mapping() # 🌟 꼬리표 매핑 데이터 로드
    
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
            tag_info = tag_map.get(ticker, "🏢 분류없음 | 🏷️ -") # 🌟 해당 종목의 꼬리표 추출
            news_score, _ = (0.0, []) if market_type == "ETF/KR" else get_news_sentiment_details(t_name, display=15)
            final_prob = max(0.0, min(100.0, base_prob + (news_score * 5.0)))
            
            results.append({"종목명": t_name, "코드": ticker,"업종/테마 태그": tag_info, "현재가": int(df_chart['Close'].iloc[-1]), "기본확률(AI)": base_prob, "뉴스점수": news_score, "최종확률": final_prob})
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
    start_idx = max(0, len(df_plot) - 146)
    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, 
                      xaxis=dict(type='date', range=[df_plot.index[start_idx], df_plot.index[-1]]), 
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig

def draw_correlation_network(market="KOSPI", top_n=30):
    try:
        df_list = fdr.StockListing(market)
        tickers = df_list.sort_values('Marcap', ascending=False).head(top_n)['Code'].tolist()
        names = df_list.sort_values('Marcap', ascending=False).head(top_n)['Name'].tolist()
        t_map = dict(zip(tickers, names))
        
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
        
        G = nx.Graph()
        THRESHOLD = 0.7 
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= THRESHOLD:
                    G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr)
                    
        pos = nx.spring_layout(G, k=1.2, seed=42) 
        
        # --- 1. 엣지(선) 데이터 생성 ---
        edge_x, edge_y = [], []
        # 🌟 엣지 툴팁용 중앙점 데이터
        edge_mid_x, edge_mid_y, edge_hover_text = [], [], []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # 선의 중앙 지점 계산 및 툴팁 텍스트 생성
            # 🌟 툴팁이 더 잘 걸리도록 중앙뿐만 아니라 여러 지점에 포인트를 심습니다.
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: # 30%, 50%, 70% 지점에 포인트 배치
                edge_mid_x.append(x0 + (x1 - x0) * ratio)
                edge_mid_y.append(y0 + (y1 - y0) * ratio)
                edge_hover_text.append(f"🔗 연결: {edge[0]} ↔ {edge[1]}<br>📊 상관계수: {edge[2]['weight']:.4f}")
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, 
                                line=dict(width=0.7, color='rgba(150, 150, 150, 0.4)'), 
                                hoverinfo='none', mode='lines')

        # 🌟 엣지 중앙에 보이지 않는 점을 배치하여 툴팁 구현
        edge_info_trace = go.Scatter(
            x=edge_mid_x, y=edge_mid_y, mode='markers',
            marker=dict(size=5, color='rgba(0,0,0,0)'), # 투명한 점
            text=edge_hover_text, hoverinfo='text'
        )
        
        # --- 2. 노드(종목) 데이터 생성 ---
        node_x, node_y, node_labels, node_hover_text, node_size = [], [], [], [], []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(node)
            
            # 해당 종목의 최근 누적 수익률 계산 (정보 제공용)
            cum_ret = (df_prices[node] + 1).prod() - 1
            degree = G.degree(node)
            
            # 🌟 노드 툴팁 텍스트 고도화
            node_hover_text.append(
                f"🏢 <b>{node}</b><br>" +
                f"📈 60일 누적수익률: {cum_ret*100:+.2f}%<br>" +
                f"🕸️ 연결된 종목 수: {degree}개<br>" +
                f"💡 {node}와 동조화된 종목들을 확인하세요."
            )
            node_size.append(18 + (degree * 3)) 
            
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', 
            text=node_labels, textposition="bottom center",
            hovertext=node_hover_text, hoverinfo='text', 
            marker=dict(
                showscale=True, colorscale='Viridis', size=node_size,
                color=[G.degree(n) for n in G.nodes()], 
                line_width=2, colorbar=dict(title="연결도", thickness=15)
            ))
            
        fig = go.Figure(data=[edge_trace, edge_info_trace, node_trace],
             layout=go.Layout(
                title=dict(text=f'🕸️ {market} 핵심 테마 동조화 맵 (상계 {THRESHOLD}↑)', font=dict(size=16)),
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
GRU_PATH = r"weather_advisor_v6_master_D.pt"
LGB_PATH = r"weather_advisor_v6_master_D_lgb.pkl"
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
    menu = st.sidebar.radio("모드 선택", ["단일 종목 스캐너", "섹터 주도주 레이더", "스윙 타점 스캐너", "자금 흐름 네트워크 맵", "ETF 스캐너"], horizontal=True, label_visibility="collapsed")

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
                # 🌟 [수정포인트 1] 데이터 수집 기간 연장 (월봉 출력을 위해 1000일(약 3년)치 데이터 수집)
                df_chart = fdr.DataReader(ticker, (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d'))
                macro_df = load_macro_feature_data()
                feats_df, v_date, i_r, f_r, df_plot_daily = prepare_master_features(ticker, df_chart, macro_df)
                
                if not feats_df.empty and len(feats_df) >= 60:
                    curr_p, prev_p = df_chart['Close'].iloc[-1], df_chart['Close'].iloc[-2]
                    st.metric(label=f"{name} ({ticker})", value=f"{int(curr_p):,}원", delta=f"{int(curr_p-prev_p):+}원 ({(curr_p-prev_p)/prev_p*100:+.2f}%)")
                    
                    briefing_container = st.container()
                    
                    with st.expander("🎛️ 매크로 스트레스 테스트 (What-If 시뮬레이터)", expanded=False):
                        st.info("만약 오늘 밤 나스닥이 폭락하거나 환율이 치솟는다면, 이 종목의 내일 상승 확률은 어떻게 변할지 테스트해보세요.")
                        col_s1, col_s2, col_s3 = st.columns(3)
                        sim_nasdaq = col_s1.slider("🇺🇸 나스닥 변동 (%)", -5.0, 5.0, 0.0, 0.5,
                                                   help="간밤에 미국 기술주(나스닥)가 폭락하거나 폭등했을 때, 다음 날 해당 종목에 미칠 충격을 시뮬레이션합니다.")
                        sim_usdkrw = col_s2.slider("💵 환율 변동 (%)", -3.0, 3.0, 0.0, 0.1,
                                                   help="원/달러 환율이 급등(원화 가치 하락)하면 외국인 자금 이탈 우려가 커져 증시에 악재로 작용하는 경향이 있습니다.")
                        sim_vix = col_s3.slider("😨 VIX 공포지수 변동 (%)", -20.0, 20.0, 0.0, 1.0,
                                                help="시장의 공포지수입니다. VIX가 치솟으면 전 세계적인 투자 심리가 얼어붙어 증시에 강한 하방 압력을 줍니다.")
                    
                    # 시뮬레이션 데이터 복사 및 변동치 적용
                    sim_feats_df = feats_df.copy()
                    sim_feats_df.loc[sim_feats_df.index[-1], 'nasdaq_ret'] += (sim_nasdaq / 100.0)
                    sim_feats_df.loc[sim_feats_df.index[-1], 'usd_krw_ret'] += (sim_usdkrw / 100.0)
                    sim_feats_df.loc[sim_feats_df.index[-1], 'vix_ret'] += (sim_vix / 100.0)

                    # 시뮬레이션 적용된 데이터로 스케일링 및 AI 예측 (고정된 일봉 모델 사용)
                    scaled_feat = RobustScaler().fit_transform(sim_feats_df.tail(60).values)
                    inp = torch.FloatTensor(scaled_feat).unsqueeze(0).to(device)
                    
                    with torch.no_grad(): gru_prob = torch.softmax(model_gru(inp), dim=1).cpu().numpy()[0][1]
                    lgb_prob = model_lgb.predict_proba(scaled_feat[-1].reshape(1, -1))[0][1]
                    base_prob_pct = ((gru_prob * 0.5) + (lgb_prob * 0.5)) * 100
                    
                    sentiment_score, news_items = get_news_sentiment_details(name, display=100)
                    news_impact = sentiment_score * 5.0
                    final_prob_pct = max(0.0, min(100.0, base_prob_pct + news_impact))
                    
                    st.markdown("---")
                    
                    # 🌟 [수정포인트 2] 타임프레임 선택 라디오 버튼 (UI 배치 변경)
                    tf_col, _ = st.columns([1, 2])
                    with tf_col:
                        timeframe = st.radio("📊 차트 및 AI 분석 기준 선택", ["일봉 (단기 5일)", "주봉 (중기 4주)", "월봉 (장기 3개월)"], horizontal=True)
                    
                    # 🌟 [수정포인트] 타임프레임 선택 및 데이터 리샘플링 로직
                    df_chart_plot = df_plot_daily.copy()
                    
                    if timeframe != "일봉 (단기 5일)":
                        # 주봉/월봉 변환
                        rule = 'W-FRI' if timeframe == "주봉 (중기 4주)" else 'ME'
                        df_chart_plot = df_chart.resample(rule).agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
                        
                        # 일목균형표 기본선 계산
                        h9, l9 = df_chart_plot['High'].rolling(9).max(), df_chart_plot['Low'].rolling(9).min()
                        h26, l26 = df_chart_plot['High'].rolling(26).max(), df_chart_plot['Low'].rolling(26).min()
                        h52, l52 = df_chart_plot['High'].rolling(52).max(), df_chart_plot['Low'].rolling(52).min()
                        
                        tenkan_sen = (h9 + l9) / 2
                        kijun_sen = (h26 + l26) / 2
                        span_a_raw = (tenkan_sen + kijun_sen) / 2
                        span_b_raw = (h52 + l52) / 2
                        
                        # 🌟 [핵심 해결] 미래 날짜(26칸)를 만들어 차트 꼬리에 붙여줍니다.
                        # 주봉이면 26주, 월봉이면 26개월 치 빈 껍데기(Index)를 만듭니다.
                        future_dates = pd.date_range(start=df_chart_plot.index[-1] + pd.Timedelta(days=1), periods=26, freq=rule)
                        future_df = pd.DataFrame(index=future_dates)
                        df_chart_plot = pd.concat([df_chart_plot, future_df])
                        
                        # 값을 넣고 26칸을 앞(미래)으로 밀어냅니다.
                        tmp_span_a = pd.Series(index=df_chart_plot.index, dtype=float)
                        tmp_span_a.loc[df_chart.resample(rule).last().dropna().index] = span_a_raw
                        
                        tmp_span_b = pd.Series(index=df_chart_plot.index, dtype=float)
                        tmp_span_b.loc[df_chart.resample(rule).last().dropna().index] = span_b_raw
                        
                        df_chart_plot['tenkan_sen'] = tenkan_sen
                        df_chart_plot['kijun_sen'] = kijun_sen
                        df_chart_plot['senkou_span_a'] = tmp_span_a.shift(26)
                        df_chart_plot['senkou_span_b'] = tmp_span_b.shift(26)
                        
                        st.caption(f"💡 **안내:** 현재 보여지는 차트는 **{timeframe}** 추세 확인용입니다. 하단의 AI 타점 확률은 최적화된 **'일봉(단기 스윙)'** 데이터를 기준으로 계산되었습니다.")
                    
                    # 🌟 [수정포인트 4] AI 브리핑 동적 생성 로직 (함수 대신 인라인으로 교체하여 유연성 확보)
                    date_str = f"({v_date} 장마감 기준)" if v_date else "(수급 정보 없음)"
                    briefing = f"[{name} {timeframe} 트레이딩 브리핑] {date_str}\n\n"
                    briefing += f"AI 모델 및 뉴스 센티먼트를 종합한 최종 기술적 상승 확률은 **{final_prob_pct:.1f}%** 입니다.\n\n"

                    if timeframe == "일봉 (단기 5일)":
                        # 기존 일봉 백테스트 결과 유지
                        if final_prob_pct >= 70: briefing += f"[S급] 초고도 확신 (승률 84.8%): 강력한 매수 타이밍입니다. (TP 4% 목표)\n"
                        elif final_prob_pct >= 60: briefing += f"[A급] 강한 확신 (승률 57.7%): 단기 상승 에너지가 긍정적인 자리입니다.\n"
                        else: briefing += f"[일반 매수] (승률 50.9%): 확률적 우위가 크지 않은 애매한 구간이므로 관망을 권장합니다.\n"
                    
                    elif timeframe == "주봉 (중기 4주)":
                        # 🌟 업데이트: 20일 예측 팩트 데이터 적용
                        if final_prob_pct >= 70: briefing += f"[S급] 중기 대세 상승 (승률 75.8%): 4주간 홀딩해도 매우 안전한 최상위 스윙 타점입니다. (TP 8% 목표)\n"
                        elif final_prob_pct >= 60: briefing += f"[A급] 단기 대응 요망 (승률 50.0%): 중기(4주)로 끌고 가기엔 리스크가 있습니다. 5일 이내 단기 수익 실현을 권장합니다.\n"
                        else: briefing += f"[관망]: 중기 추세의 방향성이 불확실합니다.\n"
                        
                    elif timeframe == "월봉 (장기 3개월)":
                        # 🌟 업데이트: 60일 예측 팩트 데이터 적용
                        if final_prob_pct >= 70: briefing += f"[S급] 역사적 변곡점 (승률 76.9%): 시장에 극히 드물게 나타나는 초장기 바닥/대세 상승 초입입니다! 적극적인 비중 확대를 고려하십시오. (TP 12% 목표)\n"
                        elif final_prob_pct >= 60: briefing += f"[A급] 장기 우상향 (승률 55.9%): 거시적으로 무난한 상승장 흐름에 탑승하고 있습니다.\n"
                        else: briefing += f"[시장 평균] (승률 58.5%): 시장 평균적인 움직임을 보이고 있습니다. 개별 종목의 펀더멘털 분석이 추가로 필요합니다.\n"
                        
                    briefing += "\n💡 **수급/기술적 코멘트:**\n"
                    if f_r > 0.001 and i_r > 0.001: briefing += "현재 외국인과 기관의 쌍끌이 매수가 유입 중입니다. "
                    elif f_r > 0.001: briefing += "외국인 자금이 유입되며 하방을 방어 중입니다. "
                    elif i_r > 0.001: briefing += "기관의 저가 매수세가 들어오고 있습니다. "
                    elif f_r < -0.001 and i_r < -0.001: briefing += "현재 메이저 양매도가 출회 중이므로 접근에 주의하십시오. "
                    else: briefing += "메이저 수급의 뚜렷한 이탈이나 유입은 감지되지 않습니다. "

                    rsi = feats_df['rsi'].iloc[-1]
                    stoch = feats_df['stoch'].iloc[-1]
                    if rsi > 0.7 or stoch > 0.8: briefing += "차트가 단기 과열권에 진입했습니다. 급등 시 추격 매수보다는 조정을 대기하세요.\n"
                    elif rsi < 0.3 or stoch < 0.2: briefing += "단기 낙폭 과대 구간입니다. 기술적 반등을 노린 분할 매수가 유효합니다.\n"
                    else: briefing += "기술적 지표는 안정적인 적정 구간에 위치해 있습니다.\n"
                    
                    if news_items:
                        pos_news = max(news_items, key=lambda x: x['score'])
                        neg_news = min(news_items, key=lambda x: x['score'])
                        if pos_news['score'] >= 0.25 or neg_news['score'] <= -0.25:
                            briefing += f"\n🗣️ **AI 이슈 요약:**\n"
                            if pos_news['score'] >= 0.25: briefing += f"- 🔥 강력한 호재: [{pos_news['title']}]\n"
                            if neg_news['score'] <= -0.25: briefing += f"- 🛑 주의할 악재: [{neg_news['title']}]\n"

                    # 브리핑 출력
                    briefing_html = apply_jurin_help(briefing)
                    with briefing_container:
                        st.markdown(f"""
                        <div style="background-color: rgba(0, 230, 118, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #00E676; margin-bottom: 20px; line-height: 1.6;">
                            {briefing_html}
                        </div>
                        """, unsafe_allow_html=True)

                    # 차트 및 점수판 출력
                    col1, col2 = st.columns([2, 1])
                    with col1: 
                        st.plotly_chart(draw_ichimoku_chart(df_chart_plot), use_container_width=True)
                    
                    with col2:
                        is_simulated = sim_nasdaq != 0 or sim_usdkrw != 0 or sim_vix != 0
                        title_prefix = "🔬 [시뮬레이션 적용됨]" if is_simulated else "1차: 2 AI 앙상블"
                        
                        st.subheader(title_prefix)
                        cA, cB, cC = st.columns(3)
                        cA.metric("GRU", f"{gru_prob*100:.1f}%")
                        cB.metric("LGBM", f"{lgb_prob*100:.1f}%")
                        cC.metric("기본 확률", f"{base_prob_pct:.1f}%")
                        
                        st.markdown("---")
                        st.subheader("📰 2차: 뉴스 센티먼트 융합")
                        cD, cE, cF = st.columns(3)
                        news_emoji = "🔥" if sentiment_score > 0 else ("🛑" if sentiment_score < 0 else "➖")
                        cD.metric(f"뉴스 ({news_emoji})", f"{sentiment_score:+.2f}점")
                        cE.metric("가산점", f"{news_impact:+.1f}%p")
                        
                        delta_str = "시뮬레이션!" if is_simulated else None
                        cF.metric("최종 확신도", f"{final_prob_pct:.1f}%", delta=delta_str, delta_color="inverse")
                        
                        st.markdown("---")
                        st.subheader(f"🎯 {timeframe.split(' ')[0]} 매매 가이드")
                        
                        if timeframe == "일봉 (단기 5일)":
                            tp_rate, sl_rate = 4.0, -3.0
                        elif timeframe == "주봉 (중기 4주)":
                            tp_rate, sl_rate = 8.0, -5.0
                        else: # 월봉 (장기 3개월)
                            tp_rate, sl_rate = 12.0, -10.0
                            
                        target_price = curr_p * (1 + (tp_rate / 100))
                        stop_loss = curr_p * (1 + (sl_rate / 100))
                        
                        # 🌟 [수정 완료] 거대한 st.metric 대신 깔끔한 텍스트 리스트로 통일
                        st.write(f"- 적정 매수가: `{int(curr_p):,}원`")
                        st.write(f"- 목표가 (+{tp_rate}%): `{int(target_price):,}원`")
                        st.write(f"- 손절가 ({sl_rate}%): `{int(stop_loss):,}원`")
                                                
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

elif menu == "섹터 주도주 레이더":
        st.title("섹터 & 테마 머니플로우")
        st.info("시장의 큰 파도(업종)와 빠른 물결(테마)을 동시에 분석하여 자금의 종착지를 찾습니다.")
        
        # 🌟 1. 탭을 그리기 전에 데이터를 먼저 한 번에 수집합니다.
        with st.spinner("업종 및 테마 전수조사 및 AI 데이터 분석 중 (약 10~15초 소요)..."):
            full_up, detail_up = get_naver_market_data("upjong", 76)
            full_th, detail_th = get_naver_market_data("theme", 264)
        
        # 🌟 2. 설명 바로 밑에 '종합 피드백'을 크게 띄워줍니다.
        if not full_up.empty and not full_th.empty:
            st.markdown(generate_unified_market_briefing(full_up, full_th), unsafe_allow_html=True)
                
            tab1, tab2 = st.tabs(["업종별 사이클 (Industry)", "테마별 수급 (Theme)"])
        
            with tab1:
                with st.spinner("업종별 실시간 데이터를 수집 중..."):
                    # 🌟 수정포인트 1: 두 개의 변수로 나누어 받습니다.
                    full_up, detail_up = get_naver_market_data("upjong", 79)
                
                    if not full_up.empty and not detail_up.empty:
                        # 🌟 수정포인트 2: 브리핑에는 전체 통계(full_up)를 넣습니다.
                        st.markdown(generate_live_sector_briefing(full_up, "업종"))       
                    
                        # 🌟 수정포인트 3: 표에는 1등/꼴등이 매핑된 상세 데이터(detail_up)를 출력합니다.
                        st.dataframe(
                            detail_up.style.format({
                                "등락률": "{:+.2f}%", 
                                "1등 수익률": "{:+.2f}%", 
                                "꼴등 수익률": "{:+.2f}%"
                            }).map(
                                lambda x: 'color: #FF4B4B; font-weight: bold' if x > 0 else ('color: #1C83E1' if x < 0 else 'color: gray'), 
                                subset=["등락률", "1등 수익률", "꼴등 수익률"]
                            ),
                            use_container_width=True
                        )
                    else: st.error("업종 데이터를 불러올 수 없습니다.")

            with tab2:
                with st.spinner("테마별 실시간 데이터를 전수조사 중입니다. (최초 1회 약 10~20초 소요)..."):
                    # 🌟 수정포인트 1: 두 개의 변수로 나누어 받습니다.
                    full_th, detail_th = get_naver_market_data("theme", 264)
                
                    if not full_th.empty and not detail_th.empty:
                        # 🌟 수정포인트 2: 브리핑에는 전체 통계(full_th)를 넣습니다.
                        st.markdown(generate_live_sector_briefing(full_th, "테마"))
                    
                        # 🌟 수정포인트 3: 표에는 1등/꼴등이 매핑된 상세 데이터(detail_th)를 출력합니다.
                        st.dataframe(
                            detail_th.style.format({
                                "등락률": "{:+.2f}%", 
                                "1등 수익률": "{:+.2f}%", 
                                "꼴등 수익률": "{:+.2f}%"
                            }).map(
                                lambda x: 'color: #FF4B4B; font-weight: bold' if x > 0 else ('color: #1C83E1' if x < 0 else 'color: gray'), 
                                subset=["등락률", "1등 수익률", "꼴등 수익률"]
                            ),
                            use_container_width=True
                        )
                    else: st.error("테마 데이터를 불러올 수 없습니다.")
                
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
            cols_order = ['종목명', '코드', '업종/테마 태그', '현재가', '기본확률(AI)', '뉴스점수', '최종확률']
            format_df = display_df[cols_order]
            
            format_df['기본확률(AI)'] = format_df['기본확률(AI)'].apply(lambda x: f"{x:.1f}%")
            format_df['뉴스점수'] = format_df['뉴스점수'].apply(lambda x: f"{x:+.2f}점")
            format_df['최종확률'] = format_df['최종확률'].apply(lambda x: f"{x:.1f}%")
            
            sniper_s_df = format_df[display_df['최종확률'] >= 70.0].sort_values(by=display_df['최종확률'].name, ascending=False).reset_index(drop=True)
            sniper_a_df = format_df[(display_df['최종확률'] >= 60.0) & (display_df['최종확률'] < 70.0)].sort_values(by=display_df['최종확률'].name, ascending=False).reset_index(drop=True)
            
            st.markdown("---")
            # 🌟 [보너스] AI 바텀업 주도 섹터 자동 추출 로직
            from collections import Counter
            high_prob_df = pd.concat([sniper_s_df, sniper_a_df])
            if not high_prob_df.empty:
                sectors = [tag.split('|')[0].replace('🏢', '').strip() for tag in high_prob_df['업종/테마 태그'] if '분류없음' not in tag]
                if sectors:
                    top_sector = Counter(sectors).most_common(1)[0][0]
                    st.error(f"🧠 **[AI 바텀업 분석]** 오늘 S급/A급 타점이 가장 많이 포착된 시장의 숨은 주도 업종은 **[{top_sector}]** 입니다!")
                    
            st.markdown("---")
            if not sniper_s_df.empty:
                st.error(f"🔥 **[S급] 70% 이상 초고도 확신 타점 ({len(sniper_s_df)}건)**")
                st.dataframe(sniper_s_df, use_container_width=True)
            else: st.warning("🔥 오늘 장은 70% 이상 확신할 만한 S급 매수 타점이 없습니다.")
                
            if not sniper_a_df.empty:
                st.info(f"🚀 **[A급] 60% 이상 매수 우위 타점 ({len(sniper_a_df)}건)**")
                st.dataframe(sniper_a_df, use_container_width=True)
                
            st.markdown("---")
            st.write("📋 전체 스캔 결과 (B급 이하 포함)")
            st.dataframe(format_df.sort_values(by=display_df['최종확률'].name, ascending=False).reset_index(drop=True), use_container_width=True)
