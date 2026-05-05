import os
import argparse
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import pytz
import ta
from sklearn.preprocessing import RobustScaler
import FinanceDataReader as fdr
from bs4 import BeautifulSoup
import joblib
import warnings
import concurrent.futures  # 🌟 멀티스레딩 엔진
import random              # 🌟 네이버 차단 우회용 난수 발생기
import time
import sys
import holidays

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# 🌟 1. V6 마스터 AI 모델 구조 (24 Features)
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

# 🌟 2. 설정 변수
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GRU_PATH = "weather_advisor_v6_master_D.pt" 
LGB_PATH = "weather_advisor_v6_master_D_lgb.pkl"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ensemble_models():
    model_gru = SwingBinaryMasterGRU(input_size=24)
    try:
        model_gru.load_state_dict(torch.load(GRU_PATH, map_location=device, weights_only=True))
        model_gru.to(device).eval()
        model_lgb = joblib.load(LGB_PATH)
        return model_gru, model_lgb
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None, None

def send_discord(title, fields_data, color):
    if not DISCORD_WEBHOOK_URL:
        print("❌ 웹훅 주소가 없습니다!")
        return
        
    payload = {
        "content": "📢 **[AI Quant V6 Sniper]**",
        "embeds": [{
            "title": title,
            "description": f"기준일시: {datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M')} (KST)",
            "color": color, 
            "fields": fields_data,
            "footer": {"text": "V6 앙상블 스나이퍼 (A/S급 타점 전용)"}
        }]
    }
    requests.post(DISCORD_WEBHOOK_URL, json=payload, headers={"Content-Type": "application/json"})

# --- 3. V6 데이터 파이프라인 ---
def load_macro_feature_data():
    end_dt = datetime.today().strftime('%Y-%m-%d')
    start_dt = (datetime.today() - timedelta(days=200)).strftime('%Y-%m-%d')
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

def get_naver_market_data(group_type="upjong", count=50):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
    data = []
    page = 1
    seen_codes = set()
    
    while True:
        list_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong" if group_type == "upjong" else f"https://finance.naver.com/sise/theme.naver?page={page}"
        try:
            res = requests.get(list_url, headers=headers, timeout=10)
            res.encoding = 'euc-kr'
            soup = BeautifulSoup(res.text, 'html.parser')
            table = soup.select_one('table.type_1')
            if not table: break

            rows = table.select('tr')
            page_item_count = 0
            is_duplicate_page = False
            for row in rows:
                cols = row.select('td')
                if len(cols) >= 2:
                    link_tag = cols[0].find('a')
                    if link_tag and 'no=' in link_tag.get('href', ''):
                        code = link_tag['href'].split('no=')[-1].split('&')[0] 
                        if code in seen_codes:
                            is_duplicate_page = True
                            break
                        seen_codes.add(code)
                        name = link_tag.text.strip()
                        change_text = cols[1].text.strip().replace('%', '').replace('+', '')
                        try: change_val = float(change_text)
                        except: change_val = 0.0
                        data.append({"이름": name, "등락률": change_val, "code": code})
                        page_item_count += 1
            if group_type == "upjong" or page_item_count == 0 or is_duplicate_page: break
            if page > 15: break
            page += 1
        except: break

    full_df = pd.DataFrame(data).sort_values("등락률", ascending=False).reset_index(drop=True)
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
                for s_row in stock_table.select('tr'):
                    name_cell = s_row.select_one('td.name a')
                    tds = s_row.select('td')
                    if name_cell and len(tds) > target_td_idx:
                        s_name = name_cell.text.strip()
                        change_text = tds[target_td_idx].text.strip().replace('%', '').replace('+', '').replace(',', '')
                        if not change_text: continue 
                        try:
                            s_change = float(change_text)
                            if s_change > max_change: max_change, top_name = s_change, s_name
                            if s_change < min_change: min_change, bottom_name = s_change, s_name
                        except: continue
            if max_change == -999.0: max_change = 0.0
            if min_change == 999.0: min_change = 0.0
            final_list.append({"이름": row['이름'], "등락률": row['등락률'], "1등주(대장)": top_name, "1등 수익률": max_change, "꼴등주(부진)": bottom_name, "꼴등 수익률": min_change})
        except: continue

    return full_df, pd.DataFrame(final_list)

def extract_features_v6(ticker, df_chart, macro_df):
    if len(df_chart) < 60: return pd.DataFrame()
    
    sd_df = get_naver_supply_demand_history(ticker, pages=4)
    df = df_chart.copy().join(macro_df, how='left')
    
    if not sd_df.empty: 
        df = df.join(sd_df, how='left')
    else: 
        df['inst_net'] = 0; df['foreigner_net'] = 0
        
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
    
    feats['inst_ratio'] = df['inst_net'] / (vol + 1e-8)
    feats['foreigner_ratio'] = df['foreigner_net'] / (vol + 1e-8)
    feats['inst_ratio_5d'] = df['inst_net'].rolling(5).sum() / (vol.rolling(5).sum() + 1e-8)
    feats['foreigner_ratio_5d'] = df['foreigner_net'].rolling(5).sum() / (vol.rolling(5).sum() + 1e-8)
    
    feats['usd_krw_ret'] = df['usd_krw_ret']
    feats['nasdaq_ret'] = df['nasdaq_ret']
    feats['kospi_ret'] = df['kospi_ret']
    feats['kosdaq_ret'] = df['kosdaq_ret']
    feats['vix_ret'] = df['vix_ret']
    
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats.dropna(inplace=True)
    
    feature_cols = [
        'ret', 'dist_ma', 'macd_hist', 'adx', 'rsi', 'stoch', 'bb_pband', 'atr_pct', 'obv_ret', 'mfi', 
        'bb_width', 'cci', 'roc', 'cmf', 'will_r', 'inst_ratio', 'foreigner_ratio', 'inst_ratio_5d', 'foreigner_ratio_5d',
        'usd_krw_ret', 'nasdaq_ret', 'kospi_ret', 'kosdaq_ret', 'vix_ret'
    ]
    return feats[feature_cols]

# 🌟 [신규] 단일 종목 처리 함수 (멀티스레딩 대응)
def process_single_ticker(ticker, name, market, mode, macro_df, model_gru, model_lgb):
    try:
        # 네이버 금융 봇 차단 방지를 위한 미세 딜레이 (0.1 ~ 0.5초 랜덤)
        time.sleep(random.uniform(0.1, 0.5))
        
        df = fdr.DataReader(ticker, (datetime.now() - pd.Timedelta(days=150)).strftime('%Y-%m-%d'))
        
        if mode == "afternoon" and len(df) > 2:
            pred_df = df.iloc[:-1] 
        else:
            pred_df = df
            
        f_df = extract_features_v6(ticker, pred_df, macro_df)
        if f_df.empty or len(f_df) < 60: return None
        
        scaled = RobustScaler().fit_transform(f_df.tail(60).values)
        input_t = torch.FloatTensor(scaled).unsqueeze(0).to(device)
        
        with torch.no_grad():
            gru_prob = torch.softmax(model_gru(input_t), dim=1).cpu().numpy()[0][1]
        lgb_prob = model_lgb.predict_proba(scaled[-1].reshape(1, -1))[0][1]
        final_prob = (gru_prob * 0.5 + lgb_prob * 0.5) * 100
        
        curr_price = int(pred_df['Close'].iloc[-1])
        tp_price = int(curr_price * 1.04)
        sl_price = int(curr_price * 0.97)
        
        res_dict = {
            "시장": market,
            "종목명": name,
            "코드": ticker,
            "최종확률": final_prob, 
            "예측시점가격": curr_price,
            "목표가": tp_price,
            "손절가": sl_price
        }
        
        if mode == "afternoon":
            res_dict["오늘종가"] = int(df['Close'].iloc[-1])
            
        return res_dict
    except:
        return None

# --- 4. 메인 스캐너 및 알람 실행 로직 ---
def run_scanner(mode="morning"):
    model_gru, model_lgb = load_ensemble_models()
    if not model_gru or not model_lgb: return
    
    try:
        # 🌟 1. 시가총액(Marcap) 대신 '거래대금(Amount)' 상위 종목으로 필터링하여 시장의 핫이슈를 추적합니다.
        df_krx = fdr.StockListing('KRX')
        df_kospi = df_krx[df_krx['Market'] == 'KOSPI'].sort_values('Amount', ascending=False).head(500)
        df_kosdaq = df_krx[df_krx['Market'] == 'KOSDAQ'].sort_values('Amount', ascending=False).head(500)
        df_list = pd.concat([df_kospi, df_kosdaq])
        
        tickers = df_list['Code'].tolist()
        names = df_list['Name'].tolist()
        markets = df_list['Market'].tolist()
    except Exception as e: 
        print(f"종목 리스트 로드 에러: {e}")
        return
        
    macro_df = load_macro_feature_data()
    results = []
    
    print(f"🚀 총 {len(tickers)}개 종목 (KOSPI 500 + KOSDAQ 500) 멀티스레드 스캔 시작...")
    
    # 🌟 2. 멀티스레딩(Multi-threading) 적용: 10개 종목을 동시에 분석
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_ticker, t, n, m, mode, macro_df, model_gru, model_lgb): t for t, n, m in zip(tickers, names, markets)}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
            completed_count += 1
            if completed_count % 50 == 0:
                print(f"🔄 병렬 분석 진행 중... [{completed_count} / {len(tickers)}] 완료", flush=True)

    rank_df = pd.DataFrame(results)
    if rank_df.empty: return

    if mode == "morning":
        rank_df.to_csv("morning_scan_result.csv", index=False, encoding='utf-8-sig')
        print("✅ [1/4] 1,000종목 병렬 스캔 CSV 저장 완료")

        print("🔄 섹터/테마 데이터 수집 중...")
        try:
            _, detail_up = get_naver_market_data("upjong", 76)
            detail_up.to_csv("sector_upjong.csv", index=False, encoding='utf-8-sig')
            _, detail_th = get_naver_market_data("theme", 264)
            detail_th.to_csv("sector_theme.csv", index=False, encoding='utf-8-sig')
            print("✅ [2/4] 섹터/테마 CSV 저장 완료")
        except Exception as e:
            print(f"❌ 섹터/테마 수집 실패: {e}")

        print("🔄 ETF 레이더 스캔 중...")
        try:
            etf_list = fdr.StockListing('ETF/KR')
            etf_tickers = etf_list.sort_values('Volume', ascending=False).head(20)['Symbol'].tolist()
            etf_names = etf_list.sort_values('Volume', ascending=False).head(20)['Name'].tolist()
            etf_markets = ["ETF"] * len(etf_tickers)
            
            etf_results = []
            # ETF도 멀티스레드로 빠르게 긁어옵니다.
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as etf_executor:
                etf_futures = [etf_executor.submit(process_single_ticker, t, n, m, mode, macro_df, model_gru, model_lgb) for t, n, m in zip(etf_tickers, etf_names, etf_markets)]
                for future in concurrent.futures.as_completed(etf_futures):
                    res = future.result()
                    if res: etf_results.append(res)
            
            pd.DataFrame(etf_results).to_csv("etf_scanner_result.csv", index=False, encoding='utf-8-sig')
            print("✅ [3/4] ETF CSV 저장 완료")
        except Exception as e:
            print(f"❌ ETF 수집 실패: {e}")

        print("✅ [4/4] 디스코드 알람 전송 준비...")
        s_class = rank_df[rank_df["최종확률"] >= 70.0].sort_values("최종확률", ascending=False)
        a_class = rank_df[(rank_df["최종확률"] >= 60.0) & (rank_df["최종확률"] < 70.0)].sort_values("최종확률", ascending=False)
        
        fields = []
        if not s_class.empty:
            fields.append({"name": "🔥 **[S급] 초고도 확신 타점 (승률 85%)**", "value": "적극적인 비중 베팅을 고려할 만한 강력한 상승 신호입니다.", "inline": False})
            fields.extend([{"name": f"🎯 [{row['시장']}] {row['종목명']}", "value": f"확률: **{row['최종확률']:.1f}%**\n💵 적정가: `{row['예측시점가격']:,}원`\n🚀 목표가: `{row['목표가']:,}원` (+4%)\n🛑 손절가: `{row['손절가']:,}원` (-3%)", "inline": False} for _, row in s_class.head(5).iterrows()])
            
        if not a_class.empty:
            fields.append({"name": "🚀 **[A급] 강한 확신 타점 (승률 60%↑)**", "value": "매수 우위 구간입니다. 수급과 호가를 체크하며 진입하세요.", "inline": False})
            fields.extend([{"name": f"✅ [{row['시장']}] {row['종목명']}", "value": f"확률: **{row['최종확률']:.1f}%**\n💵 적정가: `{row['예측시점가격']:,}원`\n🚀 목표가: `{row['목표가']:,}원` (+4%)\n🛑 손절가: `{row['손절가']:,}원` (-3%)", "inline": False} for _, row in a_class.head(5).iterrows()])
            
        if not fields:
            fields.append({"name": "🛑 **관망 권장**", "value": "오늘 장은 60% 이상 확신할 만한 S급/A급 매수 타점이 포착되지 않았습니다.", "inline": False})
            
        send_discord("🌅 [08:45] 1,000종목 스캔 AI 주도주 브리핑", fields, 15158332)

    elif mode == "afternoon":
        picks = rank_df[rank_df["최종확률"] >= 60.0].sort_values("최종확률", ascending=False)
        fields = []
        
        if picks.empty:
            top_pick = rank_df.sort_values("최종확률", ascending=False).head(1)
            row = top_pick.iloc[0]
            change_pct = ((row['오늘종가'] - row['예측시점가격']) / row['예측시점가격']) * 100
            emoji = "🔴 상승 (놓침)" if change_pct > 0 else ("🔵 하락 (방어성공)" if change_pct < 0 else "⚪ 보합")
            
            fields.append({"name": "💤 오늘 아침 추천 타점 없음 (관망 채점)", "value": "아침에는 60% 이상의 종목이 없어 매수를 쉬었습니다. 가장 점수가 높았던(B급 1등) 종목의 오후 결과를 복기합니다.", "inline": False})
            fields.append({
                "name": f"📝 [{row['시장']}] {row['종목명']} (아침 확률: {row['최종확률']:.1f}%)", 
                "value": f"시작가: {row['예측시점가격']:,}원 ➡️ 마감가: {row['오늘종가']:,}원\n관망 결과: {emoji} **({change_pct:+.2f}%)**", 
                "inline": False
            })
        else:
            hit_count = 0
            total_profit = 0
            # 카톡/디스코드 메시지 길이 제한 방지를 위해 상위 10개만 평가
            for i, row in picks.head(10).iterrows():
                change_pct = ((row['오늘종가'] - row['예측시점가격']) / row['예측시점가격']) * 100
                total_profit += change_pct
                if change_pct > 0: hit_count += 1
                
                emoji = "🔴 적중" if change_pct > 0 else ("🔵 실패" if change_pct < 0 else "⚪ 보합")
                fields.append({
                    "name": f"📝 [{row['시장']}] {row['종목명']} (아침 확률: {row['최종확률']:.1f}%)", 
                    "value": f"시작가: {row['예측시점가격']:,}원 ➡️ 마감가: {row['오늘종가']:,}원\n결과: {emoji} **({change_pct:+.2f}%)**", 
                    "inline": False
                })
                
            avg_profit = total_profit / len(picks.head(10))
            win_rate = (hit_count / len(picks.head(10))) * 100
            fields.insert(0, {"name": "📊 **[오늘의 상위 10픽 스나이퍼 성적표]**", "value": f"✅ 승률: **{win_rate:.0f}%**\n💰 평균 수익률: **{avg_profit:+.2f}%**\n---", "inline": False})
            
        send_discord("🏁 [16:00] 오늘 아침 S/A급 픽 채점 결과", fields, 10181046)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["morning", "afternoon", "monthly"])
    args = parser.parse_args()
    
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    
    # 🛡️ [신규 패치] 주말 및 휴장일 완벽 차단 방어막 🛡️
    # 1. 주말 차단 (5: 토요일, 6: 일요일)
    if now.weekday() >= 5:
        print(f"[{now.strftime('%Y-%m-%d')}] 주말이므로 스캔을 실행하지 않고 봇을 종료합니다.")
        sys.exit(0)
        
    # 2. 법정 공휴일 및 대체 공휴일 차단
    kr_holidays = holidays.KR(years=now.year)
    if now.date() in kr_holidays:
        print(f"[{now.strftime('%Y-%m-%d')}] 법정 공휴일이므로 스캔을 실행하지 않고 봇을 종료합니다.")
        sys.exit(0)
        
    # 3. 한국거래소(KRX) 특수 휴장일 차단 (5월 1일 근로자의 날, 12월 31일 연말 폐장일)
    if (now.month == 5 and now.day == 1) or (now.month == 12 and now.day == 31):
        print(f"[{now.strftime('%Y-%m-%d')}] KRX 특수 휴장일이므로 스캔을 실행하지 않고 봇을 종료합니다.")
        sys.exit(0)
        
    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {args.mode} 모드 정상 영업일 확인 완료. 스캔을 시작합니다...")
        
    run_scanner(args.mode)
