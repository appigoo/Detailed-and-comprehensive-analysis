# -*- coding: utf-8 -*-
"""
自動日報表 v2.0：K線形態 + ATR風控 + 支撐阻力 + 多空評分 + 個性化持倉
改進重點：
1. 資料層：curl_cffi chrome 模擬（防 Streamlit Cloud 被 Yahoo 封鎖）+ 多層 fallback
2. 指標層：新增 ATR14、布林通道、Wilder RSI、EMA斜率、52週高低距離
3. K線層：吞噬/錘子/射擊之星/十字/跳空 形態辨識（用原始OHLC，不用調整價）
4. 結構層：擺動高低點（Pivot）自動偵測 → 動態支撐/阻力
5. 評分層：多空綜合評分 0-100（8個因子加權，附逐項理由）
6. 情境層：以 ATR 計算具體目標價/失效價，非空泛百分比
7. 持倉層：真實成本損益、集中度、ATR止損、風險倉位計算、Covered Call 標準履約價
8. UI層：Plotly 互動圖（K線+MACD+RSI 三聯圖）、session_state 持久化、AI Prompt 產生器
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- curl_cffi（防封鎖）----------
try:
    from curl_cffi import requests as curl_requests
    HAS_CURL = True
except Exception:
    HAS_CURL = False

st.set_page_config(page_title="自動日報表 Pro", layout="wide", page_icon="📈")

# ---------- 標準設計語言 ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+TC:wght@400;500;700&display=swap');
html, body, [class*="st-"], .stMarkdown { font-family: 'Noto Sans TC', sans-serif; }
.stApp { background-color: #f5f2eb; }
[data-testid="stMetricValue"], [data-testid="stMetricDelta"], code {
    font-family: 'IBM Plex Mono', monospace;
}
[data-testid="stMetric"] {
    background: #ffffff; border-radius: 12px; padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
div[data-testid="stExpander"] { background: #ffffff; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 1) 資料層
# =====================================================
@st.cache_data(ttl=60 * 15, show_spinner=False)
def download_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """多層下載：curl_cffi → 原生 yf.download → Ticker.history"""
    df = pd.DataFrame()

    # Layer 1: curl_cffi chrome impersonation
    if HAS_CURL:
        try:
            sess = curl_requests.Session(impersonate="chrome")
            df = yf.download(ticker, period=period, interval="1d",
                             auto_adjust=False, progress=False,
                             threads=False, session=sess)
        except Exception:
            df = pd.DataFrame()

    # Layer 2: 原生
    if df is None or df.empty:
        try:
            df = yf.download(ticker, period=period, interval="1d",
                             auto_adjust=False, progress=False, threads=False)
        except Exception:
            df = pd.DataFrame()

    # Layer 3: Ticker.history
    if df is None or df.empty:
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d",
                                           auto_adjust=False)
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[~df.index.duplicated(keep="last")]
    # 指標用調整價（含股息/拆股）；K線形態用原始 OHLC
    df["Close_calc"] = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    return df


# =====================================================
# 2) 指標層
# =====================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close_calc"]

    for span in (10, 30, 40):
        df[f"EMA{span}"] = c.ewm(span=span, adjust=False).mean()

    # MACD
    exp12 = c.ewm(span=12, adjust=False).mean()
    exp26 = c.ewm(span=26, adjust=False).mean()
    df["DIF"] = exp12 - exp26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["DIF"] - df["DEA"]

    # Wilder RSI（比 SMA 版更貼近券商顯示）
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # ATR14（Wilder）
    prev_c = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_c).abs(),
                    (df["Low"] - prev_c).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # 布林通道
    df["BB_mid"] = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["BB_up"] = df["BB_mid"] + 2 * bb_std
    df["BB_dn"] = df["BB_mid"] - 2 * bb_std

    df["VOL_MA20"] = df["Volume"].rolling(20, min_periods=1).mean()
    df["VOL_ratio"] = df["Volume"] / df["VOL_MA20"]
    return df


def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3):
    """擺動高低點（fractal pivot）→ 動態支撐阻力"""
    highs, lows = [], []
    h, l = df["High"].values, df["Low"].values
    for i in range(left, len(df) - right):
        if h[i] == h[i - left:i + right + 1].max():
            highs.append((df.index[i], float(h[i])))
        if l[i] == l[i - left:i + right + 1].min():
            lows.append((df.index[i], float(l[i])))
    return highs[-6:], lows[-6:]


def nearest_levels(price: float, pivot_highs, pivot_lows):
    res = sorted([p for _, p in pivot_highs if p > price])
    sup = sorted([p for _, p in pivot_lows if p < price], reverse=True)
    return (res[0] if res else None), (sup[0] if sup else None)


# =====================================================
# 3) K線形態層（原始 OHLC，紅漲綠跌）
# =====================================================
def candle_pattern(row, prev):
    o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    upper = h - max(c, o)
    lower = min(c, o) - l
    br = body / rng
    tags = []

    base = "十字/小實體" if br < 0.12 else ("陽線(綠K)" if c > o else "陰線(紅K)")
    if br >= 0.7:
        base += "・長實體"
    tags.append(base)

    if lower > 2 * body and lower > upper:
        tags.append("錘子/長下影(下方有承接)")
    elif upper > 2 * body and upper > lower:
        tags.append("射擊之星/長上影(上方有賣壓)")

    if prev is not None:
        po, pc = prev["Open"], prev["Close"]
        # 吞噬
        if c > o and pc < po and c >= po and o <= pc and body > abs(pc - po):
            tags.append("看漲吞噬")
        elif c < o and pc > po and c <= po and o >= pc and body > abs(pc - po):
            tags.append("看跌吞噬")
        # 跳空
        if l > prev["High"]:
            tags.append(f"向上跳空(缺口 {prev['High']:.2f}–{l:.2f})")
        elif h < prev["Low"]:
            tags.append(f"向下跳空(缺口 {h:.2f}–{prev['Low']:.2f})")
        # 內包
        if h <= prev["High"] and l >= prev["Low"]:
            tags.append("Inside Bar(收斂待變)")
    return "，".join(tags)


def single_candle_interpret(row, prev):
    pattern = candle_pattern(row, prev)
    c = row["Close_calc"]
    if c > row["EMA10"]:
        pos = "收在EMA10上(短多)"
    elif c > row["EMA30"]:
        pos = "EMA10-30之間(整理)"
    else:
        pos = "跌破EMA30(短弱)"
    vr = row["VOL_ratio"]
    vol = f"量{vr:.2f}x" + ("(放量)" if vr > 1.5 else "(縮量)" if vr < 0.7 else "")
    rsi = row["RSI"]
    rsi_txt = f"RSI {rsi:.0f}" + ("(超買)" if rsi > 70 else "(超賣)" if rsi < 30 else "")
    return f"{pattern}；{pos}；{vol}；{rsi_txt}"


# =====================================================
# 4) 評分層：多空綜合評分 0-100
# =====================================================
def bull_bear_score(df):
    last = df.iloc[-1]
    p = last["Close_calc"]
    score, reasons = 50, []

    def add(pts, why):
        nonlocal score
        score += pts
        reasons.append(("+" if pts > 0 else "") + f"{pts}｜{why}")

    # 1. 價格 vs 均線
    if p > last["EMA10"]:
        add(8, "收在 EMA10 之上")
    elif p < last["EMA30"]:
        add(-8, "跌破 EMA30")
    # 2. 均線排列
    if last["EMA10"] > last["EMA30"] > last["EMA40"]:
        add(8, "均線多頭排列")
    elif last["EMA10"] < last["EMA30"] < last["EMA40"]:
        add(-8, "均線空頭排列")
    # 3. EMA10 斜率（5日）
    if len(df) >= 6:
        slope = last["EMA10"] - df["EMA10"].iloc[-6]
        add(5 if slope > 0 else -5, f"EMA10 五日斜率{'向上' if slope > 0 else '向下'}")
    # 4. MACD 柱
    if last["MACD_hist"] > 0:
        add(6, "MACD 柱為正")
    else:
        add(-6, "MACD 柱為負")
    # 5. MACD 柱動能（3日）
    if len(df) >= 4:
        expanding = last["MACD_hist"] > df["MACD_hist"].iloc[-4]
        add(5 if expanding else -5, f"MACD 柱三日{'擴張' if expanding else '收斂'}")
    # 6. RSI 區間
    rsi = last["RSI"]
    if 50 <= rsi <= 65:
        add(5, f"RSI {rsi:.0f} 多方健康區")
    elif rsi > 70:
        add(-5, f"RSI {rsi:.0f} 超買(追高風險)")
    elif rsi < 30:
        add(3, f"RSI {rsi:.0f} 超賣(反彈條件)")
    elif rsi < 45:
        add(-4, f"RSI {rsi:.0f} 偏弱")
    # 7. 量價配合
    up_day = last["Close"] > last["Open"]
    if last["VOL_ratio"] > 1.3:
        add(5 if up_day else -6, f"放量{'上漲(量價齊揚)' if up_day else '下跌(賣壓沉重)'}")
    # 8. 布林位置
    if not np.isnan(last["BB_up"]):
        if p > last["BB_up"]:
            add(-3, "突破布林上軌(短線過熱)")
        elif p < last["BB_dn"]:
            add(3, "跌破布林下軌(超跌)")

    score = int(np.clip(score, 0, 100))
    if score >= 70:
        verdict = "偏多"
    elif score >= 55:
        verdict = "中性偏多"
    elif score > 45:
        verdict = "中性"
    elif score > 30:
        verdict = "中性偏空"
    else:
        verdict = "偏空"
    return score, verdict, reasons


# =====================================================
# 5) 前因分析（可驗證事實，不猜消息面）
# =====================================================
def historical_context(df):
    if len(df) < 11:
        return "資料不足，無法完整分析前10日脈絡。"
    chg10 = (df["Close_calc"].iloc[-1] / df["Close_calc"].iloc[-11] - 1) * 100
    last10 = df.tail(10)
    ret = last10["Close_calc"].pct_change()
    up_i, dn_i = ret.idxmax(), ret.idxmin()

    # 連漲連跌
    signs = np.sign(df["Close_calc"].diff().tail(10).dropna())
    streak, cur = 1, signs.iloc[-1]
    for s in signs.iloc[::-1][1:]:
        if s == cur and s != 0:
            streak += 1
        else:
            break
    streak_txt = f"目前連{'漲' if cur > 0 else '跌'} {streak} 日" if cur != 0 else "最新一日平盤"

    # 距20日高低點
    hi20 = df["High"].tail(20).max()
    lo20 = df["Low"].tail(20).min()
    p = df["Close_calc"].iloc[-1]
    return (
        f"近10日累計 {chg10:+.1f}%；{streak_txt}。"
        f"單日最大漲幅 {up_i.strftime('%m-%d')} ({ret.max()*100:+.1f}%)、"
        f"最大跌幅 {dn_i.strftime('%m-%d')} ({ret.min()*100:.1f}%)，"
        f"顯示波動{'明顯放大' if (ret.abs().max() > 0.05) else '在正常範圍'}。"
        f"現價距20日高點 {hi20:.2f} 為 {(p/hi20-1)*100:+.1f}%，"
        f"距20日低點 {lo20:.2f} 為 {(p/lo20-1)*100:+.1f}%。"
    )


# =====================================================
# 6) 情境層（ATR 具體價位）
# =====================================================
def future_scenarios(df, res, sup):
    last = df.iloc[-1]
    p, atr = last["Close_calc"], last["ATR14"]
    hi20 = df["High"].tail(20).max()
    lo20 = df["Low"].tail(20).min()
    r_level = res if res else hi20
    s_level = sup if sup else lo20

    bull = (f"**🟢 多方情境（觸發：放量站上 {r_level:.2f}）**\n"
            f"目標一 {r_level + 1.5*atr:.2f}（+1.5 ATR）、目標二 {r_level + 3*atr:.2f}（+3 ATR）；"
            f"失效條件：突破後跌回 {r_level - 0.5*atr:.2f} 以下（假突破）。")
    bear = (f"**🔴 空方情境（觸發：放量跌破 {s_level:.2f}）**\n"
            f"目標一 {s_level - 1.5*atr:.2f}（-1.5 ATR）、目標二 {max(s_level - 3*atr, lo20 - atr):.2f}；"
            f"失效條件：跌破後收回 {s_level + 0.5*atr:.2f} 之上（假跌破）。")
    rng = (f"**🟡 盤整情境（未觸發上述任一）**\n"
           f"預期在 {s_level:.2f}–{r_level:.2f} 區間震盪"
           f"（區間寬度 {(r_level-s_level)/p*100:.1f}%，日均波動 ATR={atr:.2f}，"
           f"約 {atr/p*100:.1f}%/日），縮量時宜觀望，等突破再表態。")
    return "\n\n".join([bull, bear, rng])


# =====================================================
# 7) 持倉層
# =====================================================
def round_strike(x, step=5.0):
    return round(x / step) * step

def generate_holding_advice(rep, shares, cost_price, capital, risk_pct):
    p = rep["current_price"]
    atr = rep["atr"]
    ema10, ema30, ema40 = rep["ema10"], rep["ema30"], rep["ema40"]
    res, sup = rep["res"], rep["sup"]
    score, verdict = rep["score"], rep["verdict"]
    mv = shares * p

    lines = [f"### 💼 持倉診斷：{shares:,} 股 {rep['ticker']}（市值 ${mv:,.0f}）\n"]

    # --- 真實損益 ---
    if cost_price and cost_price > 0:
        pnl = (p - cost_price) * shares
        pnl_pct = (p / cost_price - 1) * 100
        state = "獲利" if pnl >= 0 else "虧損"
        lines.append(f"**成本 ${cost_price:.2f} → 現價 ${p:.2f}**：未實現{state} "
                     f"**${pnl:+,.0f}（{pnl_pct:+.1f}%）**。")
        if pnl_pct > 15:
            lines.append("已有可觀利潤 → 重點是**保護利潤**：可將止損上移至成本上方或用移動止損，讓利潤奔跑但不回吐過半。")
        elif pnl_pct < -8:
            lines.append("虧損已超過一般單筆風險上限 → 重點是**止血**：不要因為想回本而加倉攤平，先評估技術面是否已破壞。")
    else:
        lines.append("_（未輸入成本價，以下只做技術面持倉管理，不含損益計算。）_")

    # --- 集中度 ---
    if capital and capital > 0:
        conc = mv / capital * 100
        lines.append(f"**倉位集中度：{conc:.1f}%**（本倉市值 / 總資金）。"
                     + ("超過 20%，單一股票風險偏高，波動一次就影響整體淨值。" if conc > 20
                        else "在 20% 以內，屬可承受範圍。"))
    else:
        lines.append("_（未輸入總資金，無法評估集中度——建議填入，這是風控最重要的數字。）_")

    # --- ATR 止損 ---
    stop_atr = p - 1.5 * atr
    stop_tech = min(ema30, sup) if sup else ema30
    stop_use = max(stop_atr, stop_tech * 0.995)  # 取較貼近現價、且不低於結構位太多者
    lines.append(f"\n**🛡️ 止損（雙軌制）**\n"
                 f"- 波動止損：現價 - 1.5×ATR = **${stop_atr:.2f}**（ATR={atr:.2f}）\n"
                 f"- 結構止損：{'支撐位 $%.2f' % sup if sup else 'EMA30 $%.2f' % ema30} 下方\n"
                 f"- 建議取兩者中較高者執行：**${stop_use:.2f}**，放量跌破當日就出，不留過夜。")

    # --- 風險倉位計算 ---
    if capital and capital > 0:
        risk_amt = capital * risk_pct / 100
        per_share_risk = max(p - stop_use, 0.01)
        max_shares = int(risk_amt / per_share_risk)
        lines.append(f"\n**📐 風險反推倉位**（單筆風險 {risk_pct:.1f}% = ${risk_amt:,.0f}，"
                     f"每股風險 ${per_share_risk:.2f}）\n"
                     f"- 此止損下合理持股上限：**{max_shares:,} 股**\n"
                     + (f"- 你目前 {shares:,} 股 **超出 {shares - max_shares:,} 股** → 建議減至上限附近，"
                        f"讓最壞情況虧損鎖定在 {risk_pct:.1f}%。"
                        if shares > max_shares else
                        f"- 你目前 {shares:,} 股在上限內，風險敞口合理。"))

    # --- 依評分給方向 ---
    lines.append(f"\n**🎯 依多空評分（{score}/100，{verdict}）的操作優先序**")
    if score >= 60:
        lines.append(f"1. 持有為主，止損 ${stop_use:.2f} 跟隨上移\n"
                     f"2. 放量突破 {'$%.2f' % res if res else '近期高點'} → 可加碼（加碼部分用突破點下方 1×ATR 止損）\n"
                     f"3. 不必急於減倉，除非跌破止損")
    elif score >= 45:
        lines.append(f"1. 中性市況：持仓不加不減，嚴守 ${stop_use:.2f}\n"
                     f"2. 若倉位超過風險上限 → 先減到上限\n"
                     f"3. 等突破 {'$%.2f' % res if res else '壓力'} 或跌破 {'$%.2f' % sup if sup else '支撐'} 再表態")
    else:
        lines.append(f"1. 技術面偏空：**先減倉 1/3–1/2**，剩餘倉位止損 ${stop_use:.2f}\n"
                     f"2. 反彈到 EMA10（${ema10:.2f}）不過 → 可再減\n"
                     f"3. 跌破 ${stop_use:.2f} → 清倉等右側訊號，不接刀")

    # --- Covered Call ---
    strike = round_strike(p + 1.5 * atr, 5)
    lines.append(f"\n**⚙️ 進階：Covered Call 對沖**（適合願意在 ${strike:.0f} 出貨者）\n"
                 f"- 賣出 {shares // 100} 張（每張100股）Strike **${strike:.0f}**、"
                 f"約 30 天到期的 Call（≈現價 +1.5 ATR，被行使機率中低）\n"
                 f"- 效果：橫盤/小跌時收權利金降成本；代價：漲過 ${strike:.0f} 的利潤封頂\n"
                 f"- 若評分轉空（<45）不建議用 CC 代替止損——CC 只能對沖小跌，擋不住趨勢下殺。")

    lines.append("\n---\n_以上為技術面推演與風控框架，非投資建議；請依自身風險承受度執行。_")
    return "\n".join(lines)


# =====================================================
# 8) 報告組裝
# =====================================================
def generate_report(df, ticker):
    last = df.iloc[-1]
    p = last["Close_calc"]
    ph, pl = find_pivots(df)
    res, sup = nearest_levels(p, ph, pl)
    score, verdict, reasons = bull_bear_score(df)

    # 逐根K線表
    rows = []
    sub = df.tail(5)
    for i, (idx, row) in enumerate(sub.iterrows()):
        loc = df.index.get_loc(idx)
        prev = df.iloc[loc - 1] if loc > 0 else None
        rows.append({
            "日期": idx.strftime("%Y-%m-%d"),
            "開": round(row["Open"], 2), "高": round(row["High"], 2),
            "低": round(row["Low"], 2), "收": round(row["Close"], 2),
            "漲跌%": f"{(row['Close_calc']/df['Close_calc'].iloc[loc-1]-1)*100:+.2f}%" if loc > 0 else "—",
            "量比": f"{row['VOL_ratio']:.2f}x",
            "RSI": f"{row['RSI']:.0f}",
            "解讀": single_candle_interpret(row, prev),
        })

    hi20 = df["High"].tail(20).max()
    lo20 = df["Low"].tail(20).min()

    return {
        "ticker": ticker,
        "current_price": p,
        "prev_close": df["Close_calc"].iloc[-2] if len(df) > 1 else p,
        "atr": last["ATR14"],
        "rsi": last["RSI"],
        "vol_ratio": last["VOL_ratio"],
        "ema10": last["EMA10"], "ema30": last["EMA30"], "ema40": last["EMA40"],
        "dif": last["DIF"], "dea": last["DEA"], "hist": last["MACD_hist"],
        "res": res, "sup": sup, "hi20": hi20, "lo20": lo20,
        "pivot_highs": ph, "pivot_lows": pl,
        "score": score, "verdict": verdict, "reasons": reasons,
        "context": historical_context(df),
        "scenarios": future_scenarios(df, res, sup),
        "per_candle_df": pd.DataFrame(rows),
    }


def build_ai_prompt(rep, df):
    """一鍵產生餵給 AI 的結構化 Prompt（配合日常 SMC/PA 分析工作流）"""
    last5 = df.tail(5)[["Open", "High", "Low", "Close", "Volume"]].round(2)
    return f"""你是專業美股交易分析師，請用繁體中文以 SMC + Price Action 框架分析以下 {rep['ticker']} 日線數據，
輸出：1)當前市場結構(HH/HL或LH/LL) 2)關鍵流動性區域 3)具體交易計畫(進場/止損/目標,R:R) 4)主要風險。

現價 {rep['current_price']:.2f}｜ATR14 {rep['atr']:.2f}｜RSI {rep['rsi']:.1f}｜量比 {rep['vol_ratio']:.2f}x
EMA10/30/40: {rep['ema10']:.2f} / {rep['ema30']:.2f} / {rep['ema40']:.2f}
MACD: DIF {rep['dif']:.3f}, DEA {rep['dea']:.3f}, Hist {rep['hist']:.3f}
最近擺動高點: {[round(p,2) for _,p in rep['pivot_highs']]}
最近擺動低點: {[round(p,2) for _,p in rep['pivot_lows']]}
20日高/低: {rep['hi20']:.2f} / {rep['lo20']:.2f}
多空評分: {rep['score']}/100（{rep['verdict']}）

最近5日OHLCV:
{last5.to_string()}
"""


# =====================================================
# 9) 圖表層（Plotly 三聯圖，紅漲綠跌）
# =====================================================
def make_chart(df, rep):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.15, 0.15, 0.15],
                        vertical_spacing=0.02,
                        subplot_titles=("", "成交量", "MACD", "RSI"))

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#2e8b57", increasing_fillcolor="#2e8b57",
        decreasing_line_color="#d64541", decreasing_fillcolor="#d64541",
        name="K線"), row=1, col=1)

    for span, color in [(10, "#e8a33d"), (30, "#d64541"), (40, "#3b6fb5")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA{span}"],
                                 line=dict(width=1.2, color=color),
                                 name=f"EMA{span}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], line=dict(width=0.6, color="#999", dash="dot"),
                             name="布林上軌", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_dn"], line=dict(width=0.6, color="#999", dash="dot"),
                             name="布林下軌", showlegend=False, fill="tonexty",
                             fillcolor="rgba(150,150,150,0.06)"), row=1, col=1)

    # 支撐阻力水平線
    if rep["res"]:
        fig.add_hline(y=rep["res"], line=dict(color="#d64541", width=1, dash="dash"),
                      annotation_text=f"阻力 {rep['res']:.2f}", row=1, col=1)
    if rep["sup"]:
        fig.add_hline(y=rep["sup"], line=dict(color="#2e8b57", width=1, dash="dash"),
                      annotation_text=f"支撐 {rep['sup']:.2f}", row=1, col=1)

    vol_colors = np.where(df["Close"] >= df["Open"], "#2e8b57", "#d64541")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors,
                         name="成交量", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["VOL_MA20"],
                             line=dict(width=1, color="#e8a33d"),
                             name="量MA20", showlegend=False), row=2, col=1)

    hist_colors = np.where(df["MACD_hist"] >= 0, "#2e8b57", "#d64541")
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], marker_color=hist_colors,
                         name="MACD柱", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["DIF"], line=dict(width=1, color="#e8a33d"),
                             name="DIF", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["DEA"], line=dict(width=1, color="#3b6fb5"),
                             name="DEA", showlegend=False), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(width=1.2, color="#7a5cc4"),
                             name="RSI", showlegend=False), row=4, col=1)
    fig.add_hline(y=70, line=dict(color="#d64541", width=0.8, dash="dot"), row=4, col=1)
    fig.add_hline(y=30, line=dict(color="#2e8b57", width=0.8, dash="dot"), row=4, col=1)

    fig.update_layout(
        height=760, xaxis_rangeslider_visible=False,
        paper_bgcolor="#f5f2eb", plot_bgcolor="#ffffff",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.02, x=0),
        font=dict(family="Noto Sans TC"),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # 隱藏週末空缺
    return fig


# =====================================================
# UI
# =====================================================
st.title("📈 自動日報表 Pro：K線形態 × ATR風控 × 多空評分")

with st.sidebar:
    st.header("⚙️ 設定")
    ticker = st.text_input("股票代號", value="TSLA").strip().upper()
    period = st.selectbox("歷史資料區間", ["3mo", "6mo", "1y", "2y"], index=1)
    st.divider()
    st.subheader("💼 持倉（選填）")
    shares = st.number_input("持股數量（股）", min_value=0, value=0, step=10)
    cost_price = st.number_input("平均成本價（0=未提供）", min_value=0.0, value=0.0, step=0.01)
    capital = st.number_input("總交易資金 $（0=未提供）", min_value=0.0, value=0.0, step=1000.0)
    risk_pct = st.slider("單筆可承受風險 %", 0.5, 5.0, 2.0, 0.5)
    st.divider()
    run = st.button("🚀 生成報表", use_container_width=True, type="primary")

if run:
    with st.spinner("下載資料並計算指標…"):
        df = download_data(ticker, period)
        if df.empty:
            st.error("找不到資料。可能原因：代號錯誤 / Yahoo 暫時封鎖（稍後再試）/ 網路問題。")
            st.stop()
        if len(df) < 40:
            st.warning(f"資料僅 {len(df)} 筆（<40），EMA40/布林等指標未完全收斂，解讀僅供參考。")
        df = compute_indicators(df)
        st.session_state["df"] = df
        st.session_state["rep"] = generate_report(df, ticker)

if "rep" in st.session_state:
    df = st.session_state["df"]
    rep = st.session_state["rep"]

    # ---- 指標速覽列 ----
    chg = (rep["current_price"] / rep["prev_close"] - 1) * 100
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("現價", f"${rep['current_price']:.2f}", f"{chg:+.2f}%")
    c2.metric("多空評分", f"{rep['score']}/100", rep["verdict"], delta_color="off")
    c3.metric("RSI(14)", f"{rep['rsi']:.1f}")
    c4.metric("ATR(14)", f"${rep['atr']:.2f}", f"{rep['atr']/rep['current_price']*100:.1f}%/日", delta_color="off")
    c5.metric("量比", f"{rep['vol_ratio']:.2f}x")
    c6.metric("支撐 / 阻力",
              f"{rep['sup']:.0f} / {rep['res']:.0f}" if rep["sup"] and rep["res"] else "—")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 圖表", "🔎 綜合解讀", "💼 持倉建議", "🤖 AI Prompt"])

    with tab1:
        st.plotly_chart(make_chart(df, rep), use_container_width=True)
        st.subheader("最近 5 根 K 線逐根解讀")
        st.dataframe(rep["per_candle_df"], use_container_width=True, hide_index=True)
        st.download_button("⬇️ 下載最近5根K（CSV）",
                           rep["per_candle_df"].to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{ticker}_last5.csv", mime="text/csv")

    with tab2:
        colA, colB = st.columns([1, 1])
        with colA:
            st.subheader(f"多空評分：{rep['score']}/100（{rep['verdict']}）")
            st.progress(rep["score"] / 100)
            with st.expander("查看逐項評分理由", expanded=True):
                for r in rep["reasons"]:
                    st.markdown(f"- {r}")
            st.subheader("📍 關鍵價位")
            st.markdown(
                f"- 最近阻力（擺動高點）：**{'$%.2f' % rep['res'] if rep['res'] else '20日高 $%.2f' % rep['hi20']}**\n"
                f"- 最近支撐（擺動低點）：**{'$%.2f' % rep['sup'] if rep['sup'] else '20日低 $%.2f' % rep['lo20']}**\n"
                f"- EMA10 / 30 / 40：${rep['ema10']:.2f} / ${rep['ema30']:.2f} / ${rep['ema40']:.2f}\n"
                f"- 20日高 / 低：${rep['hi20']:.2f} / ${rep['lo20']:.2f}\n"
                f"- MACD：DIF {rep['dif']:.3f}｜DEA {rep['dea']:.3f}｜柱 {rep['hist']:.3f}"
            )
        with colB:
            st.subheader("📜 前因（近10日脈絡）")
            st.markdown(rep["context"])
            st.subheader("🔮 後果（三情境 + ATR具體價位）")
            st.markdown(rep["scenarios"])

    with tab3:
        if shares > 0:
            st.markdown(generate_holding_advice(
                rep, shares,
                cost_price if cost_price > 0 else None,
                capital if capital > 0 else None,
                risk_pct))
        else:
            st.info("在左側輸入持股數量（建議連成本價與總資金一起填），即可獲得含真實損益、集中度、ATR止損與風險倉位計算的個性化建議。")

    with tab4:
        st.markdown("複製下方 Prompt 貼給 Claude / Groq，可直接接上你的 SMC + Price Action 日常分析流程：")
        st.code(build_ai_prompt(rep, df), language="text")

    st.caption("指標以調整後收盤價計算；K線形態以原始OHLC判定。本工具為技術面推演與風控框架，非投資建議。")
else:
    st.info("👈 在左側設定股票代號後，點「生成報表」開始。")
