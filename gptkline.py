import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Optional candlestick (if installed)
try:
    import mplfinance as mpf
    HAS_MPF = True
except Exception:
    HAS_MPF = False

st.set_page_config(page_title="自動日報表 (K線 + 技術解讀)", layout="wide")

# ----------------- helper functions -----------------
@st.cache_data(ttl=60*30)
def download_data(ticker: str, period: str = "6mo", interval: str = "1d"):
    """下載資料並做基本清理"""
    df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
    if df.empty:
        return df
    # 處理 MultiIndex columns（近期 yfinance 版本對單一 ticker 也使用 MultiIndex）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)  # 移除 ticker 層級，保留屬性如 'Open', 'Close'
    # 優先使用 Adjusted Close 做指標計算（若有）
    if 'Adj Close' in df.columns:
        df['Close_for_calc'] = df['Adj Close']
    else:
        df['Close_for_calc'] = df['Close']
    df.dropna(inplace=True)
    return df

def compute_indicators(df: pd.DataFrame):
    c = df['Close_for_calc']
    df['EMA10'] = c.ewm(span=10, adjust=False).mean()
    df['EMA30'] = c.ewm(span=30, adjust=False).mean()
    df['EMA40'] = c.ewm(span=40, adjust=False).mean()

    # MACD (DIF, DEA, MACD hist)
    exp12 = c.ewm(span=12, adjust=False).mean()
    exp26 = c.ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['DIF'] - df['DEA']  # positive => 多方動能

    # RSI (14日)
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 成交量平均
    df['VOL_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    return df

def classify_candle(o, h, l, c):
    """簡單分類K線：大陽、小陽、十字、長上影、長下影、長紅長黑等"""
    body = abs(c - o)
    full = h - l + 1e-9
    upper = h - max(c, o)
    lower = min(c, o) - l
    body_ratio = body / full if full > 0 else 0

    if body_ratio < 0.15:
        typ = "十字/小實體（盤整）"
    else:
        if c > o:
            typ = "陽線（紅K）"
        else:
            typ = "陰線（綠K）"

    # 影線判別
    if lower > body * 2 and lower > upper:
        typ += "，長下影（可能支撐或吸籌）"
    elif upper > body * 2 and upper > lower:
        typ += "，長上影（可能壓力或獲利了結）"
    elif body_ratio > 0.7:
        typ += "，實體長（趨勢強烈）"

    return typ

def single_candle_interpret(row, ema10, ema30, vol_ma20, rsi):
    o, h, l, c, v = row['Open'], row['High'], row['Low'], row['Close_for_calc'], row['Volume']
    typ = classify_candle(o, h, l, c)
    # 價位相對均線
    if c > ema10:
        pos = "收在 EMA10 之上（短線偏多）"
    elif c > ema30:
        pos = "跌破 EMA10 但守住 EMA30（短線整理）"
    else:
        pos = "跌破 EMA30（短期偏弱）"
    # 量能
    if np.isnan(vol_ma20) or vol_ma20 == 0:
        vol_note = "成交量資料不足"
    else:
        ratio = v / vol_ma20
        if ratio > 1.5:
            vol_note = f"量大（{ratio:.2f}x 近20日均量）"
        elif ratio < 0.7:
            vol_note = f"量縮（{ratio:.2f}x 近20日均量）"
        else:
            vol_note = f"量正常（{ratio:.2f}x 近20日均量）"
    # RSI 補充
    rsi_note = f"RSI={rsi:.1f}（" + ("超買>70，需防回檔" if rsi > 70 else "超賣<30，潛在反彈" if rsi < 30 else "中性50附近") + "）"
    return f"{typ}；{pos}；{vol_note}；{rsi_note}"

def overall_trend_text(df):
    last = df.iloc[-1]
    ema10, ema30, ema40 = last['EMA10'], last['EMA30'], last['EMA40']
    # slope 判斷：比較最近 5 天的 EMA10 與 5 天前
    slope10 = ema10 - df['EMA10'].iloc[-5] if len(df) >= 6 else ema10 - df['EMA10'].iloc[0]
    slope30 = ema30 - df['EMA30'].iloc[-5] if len(df) >= 6 else ema30 - df['EMA30'].iloc[0]
    trend = ""
    if ema10 > ema30 > ema40 and slope10 > 0 and slope30 > 0:
        trend = "中期趨勢仍偏多（均線多頭排列且向上）"
    elif ema10 < ema30 < ema40 and slope10 < 0:
        trend = "中期偏空（均線空頭排列）"
    else:
        trend = "趨勢分歧或整理（需關注均線與量能）"
    return trend

def macd_status(df):
    # 判斷 DIF 與 DEA 的最近變化
    dif = df['DIF'].iloc[-1]
    dea = df['DEA'].iloc[-1]
    hist = df['MACD_hist'].iloc[-1]
    hist_trend = "上升" if (df['MACD_hist'].iloc[-1] > df['MACD_hist'].iloc[-3] if len(df)>=3 else df['MACD_hist'].iloc[-1]>0) else "下降或收斂"
    cross = ""
    if len(df) >= 2:
        if dif > dea and df['DIF'].iloc[-2] <= df['DEA'].iloc[-2]:
            cross = "（近期出現 MACD 黃金交叉）"
        elif dif < dea and df['DIF'].iloc[-2] >= df['DEA'].iloc[-2]:
            cross = "（近期出現 MACD 死亡交叉）"
    return f"DIF={dif:.3f}, DEA={dea:.3f}, MACD_hist={hist:.3f}；柱狀態趨勢：{hist_trend} {cross}"

def rsi_status(last_rsi):
    if last_rsi > 70:
        return "RSI>70（超買，短期有回檔風險）"
    elif last_rsi < 30:
        return "RSI<30（超賣，潛在反彈機會）"
    else:
        return f"RSI={last_rsi:.1f}（中性，無明顯極端）"

def historical_context(df):
    """分析前因：最近10日趨勢變化、關鍵高低點"""
    if len(df) < 10:
        return "資料不足，無法完整分析歷史脈絡。"
    
    last10 = df.tail(10)
    price_change_10d = (df['Close_for_calc'].iloc[-1] / df['Close_for_calc'].iloc[-10] - 1) * 100
    vol_change_10d = (last10['Volume'].mean() / df['VOL_MA20'].iloc[-10] - 1) * 100 if not np.isnan(df['VOL_MA20'].iloc[-10]) else 0
    
    # 關鍵轉折點：最近10日內最大漲跌日
    daily_returns = last10['Close_for_calc'].pct_change()
    max_up_day = daily_returns.idxmax()
    max_down_day = daily_returns.idxmin()
    max_up_pct = daily_returns.max() * 100
    max_down_pct = daily_returns.min() * 100
    
    context = (
        f"**前10日脈絡：** 價格累計變動 {price_change_10d:+.1f}%，量能相對20日前均量變動 {vol_change_10d:+.1f}%。\n"
        f"關鍵事件：{max_up_day.strftime('%m-%d')} 大漲{max_up_pct:+.1f}%（可能受利多消息或技術突破）；"
        f"{max_down_day.strftime('%m-%d')} 大跌{max_down_pct:.1f}%（可能遇壓力或負面因素）。\n"
        "整體前因顯示：近期波動加劇，需留意是否延續上漲動能或轉入震盪。"
    )
    return context

def future_scenarios(df, last_rsi, macd_hist_trend, price_pos, vol_ratio):
    """分析後果：基於當前指標的未來情境"""
    scenarios = []
    
    # 多頭情境
    if price_pos == "收在 EMA10 之上" and macd_hist_trend == "上升" and last_rsi < 70:
        scenarios.append("**樂觀情境（機率中高）：** 若放量突破近期高點，MACD柱續擴張，可望延續多頭至下個阻力（預估上漲5-10%），適合加碼追漲。")
    else:
        scenarios.append("**樂觀情境（機率中低）：** 需等待回測EMA10止穩並放量反彈，方有上攻機會。")
    
    # 空頭情境
    if price_pos == "已跌破 EMA30" or last_rsi > 70:
        scenarios.append("**悲觀情境（機率中高）：** 若量縮跌破EMA30，MACD轉負，可能加速下探近期低點（預估下跌5-8%），建議減倉避險。")
    else:
        scenarios.append("**悲觀情境（機率中低）：** 守住EMA30並出現長下影陽線，可化解下行壓力。")
    
    # 中性/整理
    if 30 <= last_rsi <= 70 and vol_ratio < 1.2:
        scenarios.append("**中性情境（機率高）：** 持續盤整於EMA10-30區間，等待突破訊號；RSI中性無極端，宜觀望或小倉波段。")
    
    return "\n".join(scenarios)

def important_levels(df):
    last = df.iloc[-1]
    ema10, ema30, ema40 = last['EMA10'], last['EMA30'], last['EMA40']
    recent_high = df['High'].rolling(window=20, min_periods=1).max().iloc[-1]
    recent_low  = df['Low'].rolling(window=20, min_periods=1).min().iloc[-1]
    return {
        "EMA10": ema10,
        "EMA30": ema30,
        "EMA40": ema40,
        "recent_high_20": recent_high,
        "recent_low_20": recent_low
    }

def generate_detailed_report(df, ticker):
    """組合最終詳細綜合解讀文字（中文）"""
    last5 = df.tail(5)
    last = df.iloc[-1]
    last_rsi = last['RSI']
    vol_ratio = last['Volume'] / last['VOL_MA20'] if not np.isnan(last['VOL_MA20']) else 1

    # quick conclusion
    quick = overall_trend_text(df)

    # indicators
    macd = macd_status(df)
    vol_now = last['Volume']
    vol_ma20 = last['VOL_MA20']
    vol_note = "量不足" if np.isnan(vol_ma20) else f"當日量 {vol_now:.0f}，20日均量 {vol_ma20:.0f}（比率 {vol_ratio:.2f}x）"
    rsi_note = rsi_status(last_rsi)

    # per-candle table (新增RSI)
    rows = []
    for idx, row in last5.iterrows():
        rows.append({
            "日期": idx.strftime("%Y-%m-%d"),
            "開": round(row['Open'],2),
            "高": round(row['High'],2),
            "低": round(row['Low'],2),
            "收": round(row['Close_for_calc'],2),
            "成交量": int(row['Volume']),
            "RSI": f"{row['RSI']:.1f}",
            "單根解讀": single_candle_interpret(row, row['EMA10'], row['EMA30'], row['VOL_MA20'], row['RSI'])
        })
    per_candle_df = pd.DataFrame(rows)

    # important levels
    lv = important_levels(df)
    levels_text = (
        f"EMA10={lv['EMA10']:.3f}，EMA30={lv['EMA30']:.3f}，EMA40={lv['EMA40']:.3f}。\n"
        f"最近20日高點 {lv['recent_high_20']:.2f}，低點 {lv['recent_low_20']:.2f}。"
    )

    # historical context (前因)
    hist_context = historical_context(df)

    # future scenarios (後果)
    macd_trend = "上升" if len(df) >= 3 and last['MACD_hist'] > df['MACD_hist'].iloc[-3] else "中性"
    price_pos = ("收在 EMA10 之上" if last['Close_for_calc'] > lv['EMA10']
                 else "收在 EMA10 與 EMA30 之間" if last['Close_for_calc'] > lv['EMA30']
                 else "已跌破 EMA30")
    future_scen = future_scenarios(df, last_rsi, macd_trend, price_pos, vol_ratio)

    # composite meaning
    macd_hist_3ago = df['MACD_hist'].iloc[-3] if len(df) >= 3 else 0
    macd_hint = "動能仍正但柱體縮小（需留意動能是否繼續衰竭）" if len(df) >= 3 and last['MACD_hist'] > 0 and last['MACD_hist'] < macd_hist_3ago else \
               ("動能擴張（上攻續有機會）" if len(df) >= 3 and last['MACD_hist'] > macd_hist_3ago else "動能偏弱或收斂")

    composite = (
        f"目前價格 {price_pos}；{macd_hint}。異動量能：{vol_note}。{rsi_note}\n"
        f"{hist_context}\n"
        f"未來情境觀察：\n{future_scen}\n"
        "綜合來看：\n"
        "- 中期趨勢：" + quick + "\n"
        f"- MACD 與動能狀態：{macd}\n"
        f"- 重要價位：{levels_text}\n"
    )

    # concrete suggestions (更全面建議，整合前因後果)
    advice = [
        "**持有者建議：** 依前10日漲跌脈絡，若近期大漲後RSI超買，保守者減半倉位設停損於EMA30下方5%；積極者若MACD擴張，可持倉觀察突破近期高點。",
        "**新多單進場：** 避免追高，等待回測EMA10/30並出現長下影+放量訊號（參考前因關鍵低點），或RSI回落至50以下再布局，目標上攻10%空間。",
        "**短線/做空策略：** 若跌破EMA30且量放大（類似前因大跌日），可短空至EMA40或近期低點，停損設近期高點上方；RSI>70時為理想空點。",
        "**風險控管與後果規避：** 控倉不超總資產10%，分批操作；監控未來情境，若中性盤整持續，轉為觀望；總原則：順勢而為，嚴守停損以防黑天鵝。"
    ]

    advice_text = "\n".join([f"- {a}" for a in advice])

    # package
    report = {
        "ticker": ticker,
        "quick_summary": quick,
        "indicators_text": macd,
        "volume_note": vol_note,
        "rsi_note": rsi_note,
        "levels_text": levels_text,
        "composite_text": composite,
        "advice_text": advice_text,
        "per_candle_df": per_candle_df,
        "current_price": last['Close_for_calc'],
        "ema10": lv['EMA10'],
        "ema30": lv['EMA30'],
        "ema40": lv['EMA40'],
        "recent_high": lv['recent_high_20'],
        "recent_low": lv['recent_low_20'],
        "price_pos": price_pos,
        "macd_trend": macd_trend,
        "last_rsi": last_rsi,
        "vol_ratio": vol_ratio
    }
    return report

def generate_holding_advice(report, shares, cost_price=None):
    """根據持股數量生成個性化持倉建議"""
    if shares <= 0:
        return None
    
    ticker = report['ticker']
    current_price = report['current_price']
    ema10, ema30, ema40 = report['ema10'], report['ema30'], report['ema40']
    recent_high, recent_low = report['recent_high'], report['recent_low']
    price_pos = report['price_pos']
    macd_trend = report['macd_trend']
    last_rsi = report['last_rsi']
    vol_ratio = report['vol_ratio']
    
    # 假設成本價：如果未提供，預設為 EMA30 或輸入
    if cost_price is None:
        cost_price = round(ema30, 2)  # 假設成本在 EMA30 附近
    market_value = round(shares * current_price, 0)
    
    # 持倉規模評估（假設總資產未知，簡化為單股原則）
    scale_note = "持倉規模偏大（超過單一股票的 10% 原則）" if shares * current_price > 10000 else "持倉規模適中"
    recent_trend = "震盪加劇，短線有回調風險" if last_rsi > 60 or vol_ratio > 1.5 else "趨勢穩定，中線機會"
    
    # 建議邏輯：基於技術面動態生成
    reduce_pct = "30–50%" if last_rsi > 70 or price_pos == "已跌破 EMA30" else "20–30%"
    retain_shares = int(shares * 0.5)  # 簡化計算
    stop_loss = round(ema30 * 0.95, 2)  # EMA30 下方 5%
    breakout_level = round(recent_high, 2)
    downside_target = round(ema40, 2)
    
    holding_context = f"""📌 你的持倉情境

假設成本價在 ${cost_price} 附近（因為當前價格就在 EMA10/30 區間），市值大約 ${market_value:,} 左右。

{scale_note}。

近期走勢：{recent_trend}。"""
    
    strategies = f"""✅ 建議策略
1. 倉位管理

建議先降風險：可考慮 減倉 {reduce_pct}，鎖定部分利潤或降低風險。

若你仍看好中長期 → 保留 {retain_shares:,}–{int(shares * 0.6):,} 股，避免全倉承擔波動。

2. 止損 / 防守位

EMA30 = ${ema30:.2f} 是關鍵支撐。

建議設 止損在 ${stop_loss}–${ema30 * 0.96:.2f} 區間（跌破 EMA40 附近），若放量下跌則立即止損。

3. 上漲應對

若能 站穩 EMA10 (${ema10:.2f}) 並放量反彈 → 可暫時持有，觀察能否突破 近期高點 ${breakout_level}。

若放量突破 ${breakout_level}，可考慮再加回部分倉位。

4. 下跌應對

若跌破 ${ema30:.2f}（EMA30），需警惕 → 可能下探 ${downside_target}（EMA40）甚至 ${recent_low:.2f} 低點。

一旦出現此情境 → 建議空倉或只留小倉位。

5. 短線對沖（進階策略）

如果你能操作期權（美股常用）：

可以賣出部分 Covered Call（例如 ${breakout_level + 0.5} strike, 1 個月內到期），收取權利金降低成本，同時對沖震盪風險。

這樣即使股價整理，你仍可獲得期權收入。"""
    
    action_summary = f"""🎯 行動建議（簡化版）

立即執行：減倉至 {int(shares * 0.5):,} 股左右 → 分散風險。

守住 EMA30 (${ema30:.2f})：可繼續觀望，若跌破就嚴格止損。

若突破 ${breakout_level} 並放量：可重新加碼或持有待漲。

若回落到 ${stop_loss} 以下：建議清倉，等待新機會。"""
    
    return f"好的 👍 你有 {shares:,} 股 {ticker}，我幫你依據上面整理的技術面 + 風險控管來給具體操作建議：\n\n{holding_context}\n\n{strategies}\n\n{action_summary}"

# ----------------- Streamlit UI -----------------
st.title("📈 自動日報表：最後 5 日K + 詳細綜合解讀（含前因後果）")
with st.sidebar:
    st.header("設定")
    ticker = st.text_input("股票代號（例：TSLA、AAPL、0700.HK）", value="TSLA")
    period = st.selectbox("歷史資料區間", options=["1mo","3mo","6mo","1y","2y"], index=2)
    interval = "1d"
    show_candlestick = st.checkbox("顯示 K 線圖（需要 mplfinance）", value=False)
    
    # 新增持股數量輸入
    shares = st.number_input("你的持股數量（股）", min_value=0, value=0, step=100)
    cost_price = st.number_input("你的平均成本價（選填，預設使用 EMA30）", min_value=0.0, value=0.0, step=0.01) if shares > 0 else 0.0
    
    run_button = st.button("生成報表")

if run_button:
    with st.spinner("下載資料並計算指標..."):
        df = download_data(ticker, period=period, interval=interval)
        if df.empty:
            st.error("找不到資料或下載失敗，請確認股票代號或網路連線。")
        else:
            df = compute_indicators(df)
            if len(df) < 20:
                st.warning("資料筆數較少（<20），部分指標可能不足或不精準。")
            elif len(df) < 10:
                st.warning("資料筆數過少（<10），前因分析將簡化。")
            elif len(df) < 3:
                st.warning("資料筆數過少（<3），MACD 部分分析將簡化。")

            report = generate_detailed_report(df, ticker)

            # Layout: 左：圖，右：綜合解讀
            col1, col2 = st.columns([2,3])

            with col1:
                st.subheader("走勢圖與均線")
                # 優先用 mplfinance 畫 K 線（若使用者勾選且有安裝）
                if show_candlestick and HAS_MPF:
                    mpf_df = df[['Open','High','Low','Close_for_calc','Volume']].copy()
                    mpf_df.rename(columns={'Close_for_calc':'Close'}, inplace=True)
                    addplots = [
                        mpf.make_addplot(df['EMA10'], color='orange'),
                        mpf.make_addplot(df['EMA30'], color='red'),
                        mpf.make_addplot(df['EMA40'], color='blue')
                    ]
                    fig, axlist = mpf.plot(mpf_df, type='candle', style='charles',
                                           addplot=addplots, volume=True, returnfig=True, figsize=(10,6))
                    st.pyplot(fig)
                else:
                    # fallback: 價格與EMA線 + 成交量條
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), gridspec_kw={'height_ratios':[3,1]})
                    ax1.plot(df.index, df['Close_for_calc'], label='收盤價', linewidth=1)
                    ax1.plot(df.index, df['EMA10'], label='EMA10', linewidth=1)
                    ax1.plot(df.index, df['EMA30'], label='EMA30', linewidth=1)
                    ax1.plot(df.index, df['EMA40'], label='EMA40', linewidth=1)
                    ax1.legend(loc='upper left')
                    ax1.set_title(f"{ticker} 收盤價與EMA")
                    # volume bars
                    ax2.bar(df.index, df['Volume'])
                    ax2.set_title("成交量")
                    fig.tight_layout()
                    st.pyplot(fig)

                # allow download last5 as CSV
                csv_bytes = report['per_candle_df'].to_csv(index=False).encode('utf-8')
                st.download_button("下載最近5根K之表格（CSV）", data=csv_bytes, file_name=f"{ticker}_last5.csv", mime="text/csv")

            with col2:
                st.subheader("🔎 詳細綜合解讀")
                st.markdown(f"**快速結論：** {report['quick_summary']}")
                st.markdown(f"**指標摘要：** {report['indicators_text']}")
                st.markdown(f"**RSI 狀態：** {report['rsi_note']}")
                st.markdown(f"**量能觀察：** {report['volume_note']}")
                st.markdown(f"**重要價位：** {report['levels_text']}")
                st.markdown("**綜合說明：**")
                st.write(report['composite_text'])
                st.markdown("**具體建議（整合前因後果）**")
                st.write(report['advice_text'])

            st.subheader("最近 5 根 K 線逐根解讀（含RSI）")
            st.dataframe(report['per_candle_df'])

            # 新增：持倉操作建議
            if shares > 0:
                st.subheader("💼 個性化持倉操作建議")
                holding_advice = generate_holding_advice(report, shares, cost_price if cost_price > 0 else None)
                st.markdown(holding_advice)

            st.info("提示：程式使用 'Adj Close'（若有）做指標計算。請記得將停損與倉位依照你的風險承受度調整。新增RSI輔助超買超賣判斷。")
            st.success("報表產生完成 ✅")

# Footer / Notes
st.markdown("---")
st.markdown("若要我：")
st.markdown("- 幫你把停損改成以 ATR 動態設定（例如 1.5 * ATR）請按「需要 ATR 支援」。")
st.markdown("- 或把報表自動發到 Telegram / Email，我可以把 webhook 範例加上去（需要你提供 API token）。")

