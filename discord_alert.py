import os
import argparse
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import pytz
from bs4 import BeautifulSoup
import ta
from sklearn.preprocessing import RobustScaler
import FinanceDataReader as fdr
import time
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# 🌟 1. V6 마스터 AI 모델 구조 (19 Feature)
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

# 🌟 2. 설정 변수
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
# GitHub Actions에서 실행될 때를 위해 상대 경로 사용
MODEL_PATH = "weather_advisor_v6_master_D.pt" 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_bot_model():
    model = SwingBinaryMasterGRU(input_size=19)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None

def send_discord(title, fields_data, color):
    if not DISCORD_WEBHOOK_URL:
        print("❌ 웹훅 주소가 없습니다!")
        return
        
    payload = {
        "content": "📢 **[AI Quant V6 Radar]**",
        "embeds": [{
            "title": title,
            "description": f"기준일시: {datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M')} (KST)",
            "color": color, 
            "fields": fields_data,
            "footer": {"text": "V6 Master (수급/매크로 통합 엔진)"}
        }]
    }
    requests.post(DISCORD_WEBHOOK_URL, json=payload, headers={"Content-Type": "application/json"})

# V6 데이터 파이프라인
def load_macro_feature_data():
    end_dt = datetime.today().strftime('%Y-%m-%d')
    start_dt = (datetime.today() - timedelta(days=200)).strftime('%Y-%m-%d')
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
        if not tables or len(tables) < 2: return 0, 0
            
        rows = tables[1].find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 9 and cols[0].text.strip(): 
                try:
                    inst_str = cols[5].text.strip().replace(',', '')
                    for_str = cols[6].text.strip().replace(',', '')
                    return (int(inst_str) if inst_str else 0), (int(for_str) if for_str else 0)
                except: break
        return 0, 0
    except: return 0, 0

def extract_features_v6(ticker, df_chart, macro_df):
    if len(df_chart) < 60: return pd.DataFrame()
    df = df_chart.copy().join(macro_df, how='left')
    df.ffill(inplace=True); df.bfill(inplace=True)

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
    
    time.sleep(0.05)
    inst_net, for_net = get_naver_supply_demand(ticker)
    feats['inst_ratio'] = 0.0; feats['foreigner_ratio'] = 0.0 
    
    last_idx = feats.index[-1]
    last_vol = vol.iloc[-1] + 1e-8
    feats.loc[last_idx, 'inst_ratio'] = inst_net / last_vol
    feats.loc[last_idx, 'foreigner_ratio'] = for_net / last_vol
    feats['usd_krw_ret'] = df['usd_krw_ret']
    feats['nasdaq_ret'] = df['nasdaq_ret']
    
    return feats.dropna()

def run_scanner(mode="morning"):
    model = load_bot_model()
    if not model: return
    
    try:
        # 코스닥 시총 상위 100개로 타겟팅 (V6 모델은 코스닥에서 승률이 좋음)
        df_list = fdr.StockListing('KOSDAQ')
        tickers = df_list.sort_values('Marcap', ascending=False).head(100)['Code'].tolist()
        names = df_list.sort_values('Marcap', ascending=False).head(100)['Name'].tolist()
        t_map = dict(zip(tickers, names))
    except: return
        
    results = []
    macro_df = load_macro_feature_data()
    
    for ticker in tickers:
        try:
            df = fdr.DataReader(ticker, (datetime.now() - pd.Timedelta(days=150)).strftime('%Y-%m-%d'))
            
            # [타임머신 로직] 오후 채점 모드일 경우 오늘 캔들 가리기
            if mode == "afternoon" and len(df) > 2:
                pred_df = df.iloc[:-1] 
            else:
                pred_df = df
                
            f_df = extract_features_v6(ticker, pred_df, macro_df)
            if f_df.empty or len(f_df) < 60: continue
            
            recent = f_df.tail(60).values
            scaled = RobustScaler().fit_transform(recent)
            input_t = torch.FloatTensor(scaled).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_t)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            
            # V6는 이진분류 (0: 하락/패스, 1: 상승)
            res_dict = {
                "종목명": t_map[ticker],
                "상승확률": probs[1] * 100, 
                "예측시점가격": int(pred_df['Close'].iloc[-1])
            }
            
            if mode == "afternoon":
                res_dict["오늘종가"] = int(df['Close'].iloc[-1])
                
            results.append(res_dict)
        except: continue

    rank_df = pd.DataFrame(results)
    if rank_df.empty: return

    # 🎯 1. 매월 1일 알람
    if mode == "monthly":
        top = rank_df.sort_values("상승확률", ascending=False).head(5)
        fields = [{"name": f"🏆 {i+1}위: {row['종목명']}", "value": f"📈 AI 확신도: **{row['상승확률']:.1f}%** | 현재가: {row['예측시점가격']:,}원", "inline": False} for i, row in top.reset_index().iterrows()]
        send_discord("🗓️ [월간] 이번 달 집중 공략 AI 추천주 TOP 5", fields, 3066993)

    # 🎯 2. 매일 08:45 알람 (1~5위)
    elif mode == "morning":
        top_up = rank_df.sort_values("상승확률", ascending=False).head(5)
        # 하락 확률은 (100 - 상승확률)
        top_down = rank_df.sort_values("상승확률", ascending=True).head(5)
        
        fields = [{"name": f"🚀 상승 저격 {i+1}위: {row['종목명']}", "value": f"확률: **{row['상승확률']:.1f}%** | 어제 종가: {row['예측시점가격']:,}원", "inline": False} for i, row in top_up.reset_index().iterrows()]
        fields.append({"name": "---", "value": "📉 **하락 위험(접근 금지) TOP 5**", "inline": False})
        fields.extend([{"name": f"⚠️ 하락 위험 {i+1}위: {row['종목명']}", "value": f"하락 확률: **{(100 - row['상승확률']):.1f}%** | 어제 종가: {row['예측시점가격']:,}원", "inline": False} for i, row in top_down.reset_index().iterrows()])
        
        send_discord("🌅 [장 시작 전] 오늘 장 AI 주도주 & 회피주 브리핑", fields, 15158332)

    # 🎯 3. 매일 16:00 알람 (아침 예측 결과 채점표 1~5위)
    elif mode == "afternoon":
        top_up = rank_df.sort_values("상승확률", ascending=False).head(5)
        fields = []
        hit_count = 0
        total_profit = 0
        
        for i, row in top_up.reset_index().iterrows():
            change_pct = ((row['오늘종가'] - row['예측시점가격']) / row['예측시점가격']) * 100
            total_profit += change_pct
            if change_pct > 0: hit_count += 1
            
            emoji = "🔴 적중" if change_pct > 0 else ("🔵 실패" if change_pct < 0 else "⚪ 보합")
            fields.append({
                "name": f"📝 아침 픽 {i+1}위: {row['종목명']}", 
                "value": f"아침 기준가: {row['예측시점가격']:,}원 ➡️ 마감가: {row['오늘종가']:,}원\n결과: {emoji} **({change_pct:+.2f}%)**", 
                "inline": False
            })
            
        avg_profit = total_profit / 5
        win_rate = (hit_count / 5) * 100
        fields.append({"name": "📊 **[오늘의 AI 성적표]**", "value": f"✅ 승률: **{win_rate:.0f}%**\n💰 평균 수익률: **{avg_profit:+.2f}%**", "inline": False})
        send_discord("🏁 [장 마감] 오늘 아침 AI 예측 채점표", fields, 10181046)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["morning", "afternoon", "monthly"])
    args = parser.parse_args()
    
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {args.mode} 모드 실행 중...")
    
    if args.mode == "morning" and now.day == 1:
        run_scanner("monthly")
        
    run_scanner(args.mode)