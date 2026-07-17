"""
==============================================================================
 S&P 500 Short-Term Swing/Day-Trade Scanner  (v2 — แก้ไขจุดอ่อนจาก Code Review)
==============================================================================
วัตถุประสงค์:
    สแกนหุ้นทั้งหมดในดัชนี S&P 500 เพื่อหาโอกาสเทรดระยะสั้น (Day Trade / Hold 1-2 วัน)
    ตามเงื่อนไข Trend Following + MACD/EMA Trigger พร้อมคำนวณ Position Sizing
    แบบจำกัดความเสี่ยง (Risk Management) โดยอัตโนมัติ

วิธีใช้งาน (VS Code):
    1. ติดตั้งไลบรารีที่จำเป็น:
       pip install pandas yfinance requests lxml tabulate
    2. สแกนหาหุ้นวันนี้ (โหมดปกติ):
       python sp500_swing_scanner.py
    3. ทดสอบย้อนหลังเต็มรูปแบบ แบบ TP ตายตัว (Historical Backtest — ใช้เวลานานกว่าปกติมาก):
       python sp500_swing_scanner.py backtest
    4. ทดสอบย้อนหลังแบบ Trailing Stop (ไม่มี TP ตายตัว เปรียบเทียบกับข้อ 3):
       python sp500_swing_scanner.py backtest-trailing
    5. ทดสอบย้อนหลังแบบ Hybrid (Partial Profit ที่ 1R + Trailing Stop สำหรับไม้ที่เหลือ):
       python sp500_swing_scanner.py backtest-hybrid
    6. ผลลัพธ์จะแสดงเป็นตารางใน terminal และบันทึกเป็นไฟล์ CSV ในโฟลเดอร์เดียวกัน

การแก้ไขจากเวอร์ชันก่อนหน้า (Code Review Fixes):
    1. [BUG FIX] MACD Cross Detection: เดิมใช้ break ทันทีที่เจอสัญญาณ cross ในอดีต
       ทำให้รายงาน True ผิดพลาดหากวันล่าสุด MACD ตัดกลับลงไปแล้ว ตอนนี้บังคับเช็คว่า
       "วันล่าสุด" ต้องยังคง MACD > Signal อยู่จริง ไม่ใช่แค่เคยตัดขึ้นในอดีต
    2. [DATA QUALITY] เพิ่ม HISTORY_PERIOD เป็น 2 ปี (จากเดิม 9 เดือน) และเพิ่มจำนวนแท่ง
       ขั้นต่ำที่ต้องมีก่อนเชื่อค่า EMA/MACD เพื่อลด Warm-up Bias ให้ใกล้เคียงกับ TradingView/โบรกเกอร์มากขึ้น
    3. [RELIABILITY] ดึงตาราง Wikipedia แบบค้นหาตารางที่มีคอลัมน์ "Symbol" จริง แทนการ
       hardcode tables[0] ป้องกันพังหากโครงสร้างหน้าเว็บเปลี่ยน
    4. [RATE LIMIT] เพิ่ม time.sleep() คั่นระหว่างแต่ละ batch ตอนดึงข้อมูลจาก yfinance
       เพื่อลดความเสี่ยงโดน Yahoo Finance บล็อกชั่วคราว
    5. [MARKET REGIME] เพิ่มการเช็คเทรนด์ของ SPY (ดัชนีแม่) ก่อนอนุญาตให้ออกสัญญาณ
       ถ้าตลาดรวมเป็นขาลง จะแจ้งเตือนเสี่ยง Bull Trap แนบไปกับทุกสัญญาณ
    6. [EARNINGS RISK] เช็ควันประกาศงบการเงินของหุ้นที่ผ่านเกณฑ์ทั้งหมด หากใกล้ประกาศงบ
       จะแจ้งเตือน Gap Risk ไว้ในผลลัพธ์
    7. [SANITY CHECK] เตือนผู้ใช้หากรันสคริปต์ระหว่างตลาดสหรัฐฯ ยังเปิดอยู่ (แท่งเทียนวันนั้น
       ยังไม่ปิด ค่า Indicator อาจเปลี่ยนแปลงอีกเมื่อตลาดปิดจริง)

คำเตือน:
    - สคริปต์นี้เป็นเครื่องมือช่วยกรองหุ้นตามเงื่อนไขทางเทคนิคเท่านั้น
      ไม่ใช่คำแนะนำการลงทุน (Not Financial Advice)
    - ผู้ใช้ควรตรวจสอบข้อมูลซ้ำและบริหารความเสี่ยงด้วยวิจารณญาณของตนเองเสมอ
    - yfinance ดึงข้อมูลจาก Yahoo Finance ซึ่งอาจมีความหน่วงหรือคลาดเคลื่อนได้
      โดยเฉพาะช่วง Pre-Market / Post-Market
==============================================================================
"""

import glob
import os
import re
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION — ปรับค่าตรงนี้ได้ตามต้องการ
# ==============================================================================

CONFIG = {
    # --- เงินทุนและอัตราแลกเปลี่ยน ---
    "CAPITAL_THB": 200_000,          # ทุนตั้งต้น (บาท)
    "USD_THB_RATE": 35.0,            # เรทแลกเปลี่ยนโดยประมาณ (บาท/USD)

    # --- Liquidity & Price Filter ---
    "MIN_AVG_VOLUME_30D": 2_000_000, # วอลุ่มเฉลี่ย 30 วัน ต้องมากกว่านี้
    "MIN_PRICE": 10.0,                # ราคาต่ำสุดที่ยอมรับ (USD)
    "MAX_PRICE": 300.0,               # ราคาสูงสุดที่ยอมรับ (USD)

    # --- Indicator Periods ---
    "EMA_FAST": 20,
    "EMA_SLOW": 50,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "ATR_PERIOD": 14,

    # --- Trigger Logic ---
    "MACD_CROSS_LOOKBACK_DAYS": 2,   # ตรวจ MACD ตัด Signal ในกี่วันล่าสุด
    "EMA20_BOUNCE_THRESHOLD_PCT": 0.015,  # ราคาห่างจาก EMA20 ไม่เกินกี่% ถือว่า "เด้งจาก EMA20"

    # --- Risk Management ---
    "RISK_PER_TRADE_PCT": 0.02,      # ห้ามขาดทุนเกิน 2% ของทุนรวมต่อ 1 เทรด
    "ATR_STOPLOSS_MULTIPLIER": 2.0,  # SL = Close - 2*ATR
    # [ทดลองแล้วและ Revert กลับ] เคยลด RR เป็น 1.5 เพราะคิดว่า TP เดิม (2.0) ไกลเกินไป แต่ผล
    # Backtest จริงพิสูจน์ว่า RR=1.5 แย่กว่า (Expectancy +0.32% < +0.38%, PF 1.17 < 1.195)
    # เพราะการลด TP ไปตัดขาเทรดที่ปล่อยรันได้ไกลกว่าออกเร็วเกินไป โดยฝั่งขาดทุนไม่ได้ลดตาม
    # จึงกลับมาใช้ RR=2.0 ตามเดิม (ดูสถิติเทียบละเอียดในบทสนทนา) — ปัญหา "เงินอยู่ตรงหน้าแต่หลุดมือ"
    # แก้ด้วย Trailing Stop แทน (ดูส่วน Trailing Stop ด้านล่าง) ซึ่งตรงจุดกว่าการตั้ง TP ตายตัว
    "RISK_REWARD_RATIO": 2.0,        # TP = Entry + (Risk * 2) — กลับไปใช้ค่าที่พิสูจน์แล้วว่าดีกว่า
    "FIRST_TRANCHE_PCT": 0.50,       # ไม้แรกซื้อ 50% ของจำนวนหุ้นที่คำนวณได้

    # --- Trailing Stop (ทางเลือกแทน TP ตายตัว — เลื่อน SL ตามเมื่อราคาวิ่งไปในทางที่ดี) ---
    "ENABLE_TRAILING_STOP_SUGGESTION": True,  # แสดงคำแนะนำ Trailing Stop ในผลลัพธ์การสแกนทุกวัน
    "TRAILING_STOP_ATR_MULTIPLIER": 2.0,      # ระยะ Trail = กี่เท่าของ ATR (ใช้ค่าเดียวกับ SL เริ่มต้นโดย default)

    # --- Hybrid: Partial Profit ที่ 1R + ปล่อยที่เหลือวิ่งด้วย Trailing Stop ---
    "PARTIAL_TP_R_MULTIPLE": 1.0,     # ขายไม้แรกล็อกกำไรเมื่อราคาวิ่งไปได้ N เท่าของระยะเสี่ยง (1R)
    "PARTIAL_TP_SHARE_PCT": 0.50,     # สัดส่วนโพซิชันที่ขายตอนชน Partial TP (ที่เหลือปล่อยวิ่งด้วย Trailing)
    "MOVE_STOP_TO_BREAKEVEN_AFTER_PARTIAL": True,  # หลังขายไม้แรกแล้ว เลื่อน SL ของไม้ที่เหลือมาที่จุดเข้าซื้อทันที (กันขาดทุนซ้ำ)

    # --- Data ---
    "HISTORY_PERIOD": "2y",          # ดึงข้อมูลย้อนหลัง 2 ปี ลด Warm-up Bias ของ EMA50/MACD
    "MIN_BARS_REQUIRED": 150,        # จำนวนแท่งขั้นต่ำก่อนเชื่อค่า EMA/MACD (มากกว่า EMA_SLOW เฉยๆ)
    "BATCH_DOWNLOAD_SIZE": 50,       # จำนวนหุ้นต่อ 1 batch ตอนดึงข้อมูล (กัน rate limit)
    "SLEEP_BETWEEN_BATCHES_SEC": 2,  # หน่วงเวลาระหว่าง batch กันโดน Yahoo Finance บล็อก

    # --- Market Regime Filter ---
    "ENABLE_MARKET_REGIME_FILTER": True,   # เช็คเทรนด์ SPY ก่อนอนุญาตออกสัญญาณ
    "STRICT_MARKET_REGIME_FILTER": False,  # True = ไม่แสดงผลเลยถ้า SPY เป็นขาลง / False = แสดงผลแต่แปะคำเตือน

    # --- Earnings Gap Risk ---
    "CHECK_EARNINGS_RISK": True,     # เช็ควันประกาศงบของหุ้นที่ผ่านเกณฑ์ทั้งหมด (ก่อนจัดอันดับ Top N)
    "ENABLE_EARNINGS_BLACKOUT": True,  # True = ตัดหุ้นที่ใกล้ประกาศงบออกจากผลลัพธ์ไปเลย (ไม่ใช่แค่เตือน)
    "EARNINGS_BLACKOUT_DAYS": 2,     # ตัดออกถ้าใกล้ประกาศงบภายในกี่วัน (รวมวันนี้)
    "EARNINGS_WARNING_DAYS": 5,      # นอกช่วง Blackout แต่ยังใกล้ภายในกี่วัน ให้แจ้งเตือนไว้เฉยๆ (ไม่ตัดออก)

    # --- Output ---
    "TOP_N_RESULTS": 5,              # แสดงผลลัพธ์สุดท้ายกี่ตัว (คัดเฉพาะที่น่าสนใจที่สุด)

    # --- Historical Backtest Engine (ทดสอบย้อนหลังเต็มรูปแบบ) ---
    "HISTORICAL_BACKTEST_YEARS": 3,       # ทดสอบย้อนหลังกี่ปี
    "SLIPPAGE_PCT": 0.001,                # Slippage ตอนเข้าซื้อ (0.1% = ซื้อแพงกว่าราคาเปิดเล็กน้อย สมจริงกว่า)
    "MAX_HOLD_DAYS": 10,                  # ถือได้นานสุดกี่วันเทรด ถ้ายังไม่ชน TP/SL ให้ปิดที่ราคาปิดวันนั้น (กัน "OPEN" ค้างตลอดกาล)
    # หมายเหตุ: อีกทางเลือกหนึ่งในการแก้ปัญหา TP ไปไม่ถึงคือ "ยืด MAX_HOLD_DAYS" แทนการลด RR
    # แต่เนื่องจากกลยุทธ์นี้ออกแบบมาเป็น Day Trade/Hold 1-2 วันตั้งแต่ต้น การถือยาวขึ้นจะเบี่ยงเบน
    # จากเจตนาเดิม จึงเลือกลด RISK_REWARD_RATIO แทน (ดูเหตุผลเต็มด้านบน) — ถ้าต้องการทดลองแนวทาง
    # ยืดเวลาแทน ลองปรับค่านี้เป็น 15-20 แล้วรัน `python sp500_swing_scanner.py backtest` เทียบผลดู
    "HISTORICAL_BACKTEST_MAX_TICKERS": None,  # จำกัดจำนวนหุ้นที่ทดสอบ (None = ทดสอบทั้ง S&P 500, ใส่เลขถ้าอยากให้รันเร็วขึ้นตอนทดสอบ)

    # --- Auto Backtest (Forward-Tracking รายวัน) ---
    "ENABLE_AUTO_BACKTEST": True,     # เช็คผลหุ้น Top N ของ "รอบก่อนหน้า" อัตโนมัติทุกครั้งที่รัน
    "BACKTEST_HISTORY_FILE": "backtest_history.csv",  # ไฟล์บันทึกสถิติสะสม Win Rate

    # --- Active Trades Tracker (ติดตามหุ้นที่ยังเปิดออเดอร์ (OPEN) อยู่ทุกวัน) ---
    "ENABLE_ACTIVE_TRADE_TRACKER": True,   # เปิด/ปิดการติดตามหุ้นเก่าที่ยังถืออยู่ก่อนเริ่มสแกนวันใหม่
    "ACTIVE_TRADE_TIME_STOP_DAYS": 5,      # Time Stop: ถือเกินกี่วันทำการแล้วยังไม่ชน TP/SL ให้บังคับขายทิ้ง
    "ACTIVE_TRADE_LOOKBACK_PERIOD": "1y",  # ช่วงข้อมูลย้อนหลังที่ดึงมาเพื่อคำนวณ MACD ให้แม่นยำ (ต้องยาวพอสำหรับ EMA26/MACD)
}


# ==============================================================================
# 2. ดึงรายชื่อหุ้น S&P 500 จาก Wikipedia
# ==============================================================================

def get_sp500_tickers() -> list:
    """
    ดึงรายชื่อหุ้นทั้งหมดในดัชนี S&P 500 จาก Wikipedia
    คืนค่าเป็น list ของ ticker symbol ที่ยฟinance ใช้งานได้ (แปลง . เป็น - เช่น BRK.B -> BRK-B)
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    print(f"[INFO] กำลังดึงรายชื่อหุ้น S&P 500 จาก Wikipedia ...")

    # Wikipedia จะตอบกลับ 403 Forbidden ถ้า request ไม่มี User-Agent
    # (pd.read_html เรียกตรงๆ จะไม่ส่ง header นี้ให้) จึงต้องดึง HTML ผ่าน
    # requests พร้อมใส่ header ปลอมเป็นเบราว์เซอร์ก่อน แล้วค่อยส่งต่อให้ pandas แปลงตาราง
    import io
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text))

    # [FIX] เดิม hardcode tables[0] ซึ่งจะพังทันทีถ้า Wikipedia แก้โครงสร้างหน้า
    # (เช่น เพิ่มตารางใหม่ไว้ด้านบน) ตอนนี้ค้นหาตารางที่มีคอลัมน์ "Symbol" จริงแทน
    sp500_table = None
    for t in tables:
        if "Symbol" in t.columns:
            sp500_table = t
            break

    if sp500_table is None:
        raise RuntimeError(
            "ไม่พบตารางที่มีคอลัมน์ 'Symbol' ในหน้า Wikipedia — "
            "โครงสร้างหน้าเว็บอาจเปลี่ยนไป กรุณาตรวจสอบ URL ด้วยตนเอง"
        )

    tickers = sp500_table["Symbol"].tolist()
    # yfinance ใช้ '-' แทน '.' สำหรับหุ้นบางตัว เช่น BRK.B, BF.B
    tickers = [t.replace(".", "-") for t in tickers]

    print(f"[INFO] พบหุ้นทั้งหมด {len(tickers)} ตัวในดัชนี S&P 500")
    return tickers


# ==============================================================================
# 3. ดึงข้อมูลราคาย้อนหลัง (EOD) ผ่าน yfinance
# ==============================================================================

def download_price_data(tickers: list, period: str, batch_size: int) -> dict:
    """
    ดึงข้อมูลราคาย้อนหลังรายวัน (OHLCV) สำหรับหุ้นทุกตัวใน tickers
    ดึงเป็น batch เพื่อลดโอกาสโดน rate-limit จาก Yahoo Finance
    คืนค่าเป็น dict {ticker: DataFrame}
    """
    all_data = {}
    n = len(tickers)
    total_batches = (n + batch_size - 1) // batch_size

    for i in range(0, n, batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"[INFO] ดาวน์โหลดข้อมูล batch {batch_num} "
              f"({i + 1}-{min(i + batch_size, n)} จาก {n} ตัว) ...")
        try:
            raw = yf.download(
                tickers=batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"[WARN] batch นี้ดาวน์โหลดล้มเหลว: {e}")
            continue
        finally:
            # [FIX] หน่วงเวลาระหว่าง batch เพื่อลดความเสี่ยงโดน Yahoo Finance
            # rate-limit / บล็อกชั่วคราว (ไม่หน่วงหลัง batch สุดท้าย)
            if batch_num < total_batches:
                time.sleep(CONFIG["SLEEP_BETWEEN_BATCHES_SEC"])

        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = raw.copy()
                else:
                    df = raw[ticker].copy()
                df = df.dropna(how="all")
                if df.empty or len(df) < CONFIG["MIN_BARS_REQUIRED"]:
                    continue
                all_data[ticker] = df
            except Exception:
                continue

    print(f"[INFO] ดึงข้อมูลสำเร็จทั้งหมด {len(all_data)} ตัว")
    return all_data


# ==============================================================================
# 4. คำนวณ Indicators: EMA, MACD, ATR
# ==============================================================================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """คำนวณ EMA20, EMA50, MACD/Signal, ATR14 แล้วเพิ่มเป็นคอลัมน์ใหม่ใน DataFrame"""
    df = df.copy()

    # --- EMA ---
    df["EMA20"] = df["Close"].ewm(span=CONFIG["EMA_FAST"], adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=CONFIG["EMA_SLOW"], adjust=False).mean()

    # --- MACD ---
    ema_fast = df["Close"].ewm(span=CONFIG["MACD_FAST"], adjust=False).mean()
    ema_slow = df["Close"].ewm(span=CONFIG["MACD_SLOW"], adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=CONFIG["MACD_SIGNAL"], adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # --- ATR (True Range แบบ Wilder's smoothing) ---
    high_low = df["High"] - df["Low"]
    high_close_prev = (df["High"] - df["Close"].shift(1)).abs()
    low_close_prev = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["ATR14"] = true_range.ewm(alpha=1 / CONFIG["ATR_PERIOD"], adjust=False).mean()

    # --- Average Volume 30D ---
    df["AvgVol30"] = df["Volume"].rolling(window=30).mean()

    return df


# ==============================================================================
# 5. Screening Logic: Liquidity + Trend + Trigger
# ==============================================================================

def check_signal(df: pd.DataFrame) -> dict:
    """
    ตรวจสอบเงื่อนไขการเข้าซื้อของหุ้น 1 ตัว จากข้อมูล indicator ล่าสุด
    คืนค่า dict ที่บอกว่าผ่านเงื่อนไขหรือไม่ พร้อมเหตุผลประกอบ
    """
    if len(df) < CONFIG["MIN_BARS_REQUIRED"]:
        return {"pass": False}

    last = df.iloc[-1]
    close = last["Close"]
    avg_vol30 = last["AvgVol30"]

    # --- 1) Liquidity & Price Filter ---
    if pd.isna(avg_vol30) or avg_vol30 <= CONFIG["MIN_AVG_VOLUME_30D"]:
        return {"pass": False}
    if not (CONFIG["MIN_PRICE"] <= close <= CONFIG["MAX_PRICE"]):
        return {"pass": False}

    # --- 2) Trend Filter: Close > EMA20 > EMA50 ---
    is_uptrend = close > last["EMA20"] > last["EMA50"]
    if not is_uptrend:
        return {"pass": False}

    # --- 3) Trigger: MACD cross above Signal (ใน N วันล่าสุด) ---
    # [BUG FIX] เดิมใช้ break ทันทีที่เจอ cross ในอดีต โดยไม่เช็คว่า "วันล่าสุด"
    # ยังคงยืนเหนือ Signal Line อยู่หรือไม่ ทำให้รายงาน True ผิดพลาดได้ในกรณีที่
    # ราคาร่วงจน MACD ตัดกลับลงไปแล้วหลังจากวันที่ cross ขึ้น
    # ตอนนี้บังคับ 2 เงื่อนไขร่วมกัน:
    #   (ก) เกิดการ cross ขึ้นจริงในช่วง lookback วันที่ผ่านมา
    #   (ข) ข้อมูลของ "วันล่าสุด" (แถวสุดท้าย) ต้องยังคง MACD > Signal อยู่ ณ ปัจจุบัน
    lookback = CONFIG["MACD_CROSS_LOOKBACK_DAYS"]
    recent = df.iloc[-(lookback + 1):]

    crossed_up_in_window = False
    for i in range(1, len(recent)):
        prev_row = recent.iloc[i - 1]
        curr_row = recent.iloc[i]
        if prev_row["MACD"] <= prev_row["MACD_Signal"] and curr_row["MACD"] > curr_row["MACD_Signal"]:
            crossed_up_in_window = True
            break  # เจอจุด cross แล้ว ไม่ต้องหาต่อ แต่ยังต้องเช็คเงื่อนไข (ข) ด้านล่างอยู่ดี

    macd_still_above_signal = last["MACD"] > last["MACD_Signal"]
    macd_cross_up = crossed_up_in_window and macd_still_above_signal

    # --- 3b) Trigger ทางเลือก: ราคาทะลุ/เด้งขึ้นจาก EMA20 ---
    dist_from_ema20_pct = abs(close - last["EMA20"]) / last["EMA20"]
    ema20_bounce = (
        close > last["EMA20"]
        and dist_from_ema20_pct <= CONFIG["EMA20_BOUNCE_THRESHOLD_PCT"]
    )

    if not (macd_cross_up or ema20_bounce):
        return {"pass": False}

    # --- ประกอบเหตุผลการเข้าซื้อ ---
    reasons = []
    reasons.append(f"เทรนด์ขาขึ้นชัดเจน: Close ${close:.2f} > EMA20 ${last['EMA20']:.2f} > EMA50 ${last['EMA50']:.2f}")
    if macd_cross_up:
        reasons.append(f"MACD ตัด Signal ขึ้น ภายใน {lookback} วันล่าสุด (สัญญาณซื้อ Momentum)")
    if ema20_bounce:
        reasons.append(f"ราคาเด้ง/ยืนเหนือเส้น EMA20 ห่างไม่เกิน {CONFIG['EMA20_BOUNCE_THRESHOLD_PCT']*100:.1f}%")
    reasons.append(f"สภาพคล่องสูง: วอลุ่มเฉลี่ย 30 วัน = {avg_vol30:,.0f} หุ้น/วัน")

    # --- ประเภทของสัญญาณที่เกิด (เพื่อข้อมูลอ้างอิงเท่านั้น ไม่ได้แปลว่า "คู่" ดีกว่า "เดี่ยว") ---
    # [หมายเหตุจาก Historical Backtest] เดิมเคยติดป้าย "สัญญาณคู่ (แข็งแรง)" โดยสันนิษฐานว่า
    # เกิดสัญญาณพร้อมกัน 2 แบบน่าจะแม่นกว่า แต่ผลทดสอบย้อนหลังจริง 3 ปี (8,569 เทรด) พบว่า
    # สัญญาณคู่กลับมี Win Rate ต่ำสุด (46.8%) และ Profit Factor ต่ำกว่า 1 (ขาดทุนสุทธิ)
    # จึงเปลี่ยนป้ายให้เป็นกลาง ไม่บ่งชี้ว่าดีกว่าหรือแย่กว่า เพื่อไม่ให้ผู้ใช้ตัดสินใจผิดจากป้ายชื่อ
    signal_strength = int(macd_cross_up) + int(ema20_bounce)
    # [เพิ่มสำหรับจับคู่กับสถิติ Backtest] signal_type ใช้ key เดียวกับที่ backtest_historical_signals*
    # ใช้บันทึกในคอลัมน์ 'signal_type' (both/macd/ema20) เพื่อให้ดึงสถิติ Win Rate จริงมาแสดงได้
    if macd_cross_up and ema20_bounce:
        signal_type = "both"
        signal_label = "สัญญาณคู่ (MACD+EMA20)"
    elif macd_cross_up:
        signal_type = "macd"
        signal_label = "สัญญาณเดี่ยว (MACD Cross)"
    else:
        signal_type = "ema20"
        signal_label = "สัญญาณเดี่ยว (EMA20 Bounce)"

    return {
        "pass": True,
        "close": close,
        "ema20": last["EMA20"],
        "ema50": last["EMA50"],
        "atr14": last["ATR14"],
        "avg_vol30": avg_vol30,
        "macd_cross_up": macd_cross_up,
        "ema20_bounce": ema20_bounce,
        "signal_strength": signal_strength,
        "signal_type": signal_type,
        "signal_label": signal_label,
        "reason": " | ".join(reasons),
    }


# ==============================================================================
# 6. Position Sizing & Risk Management
# ==============================================================================

def calculate_position(signal: dict, capital_usd: float) -> dict:
    """
    คำนวณ Entry, Stop Loss, Take Profit และจำนวนหุ้นไม้แรก (50%)
    โดยจำกัดความเสี่ยงไม่เกิน RISK_PER_TRADE_PCT ของทุนรวมต่อ 1 เทรด
    พร้อมคำนวณระยะ Trailing Stop แนะนำ (ทั้งแบบ $ Amount และ % ) สำหรับนำไปตั้งค่าจริงในแอปเทรด
    """
    entry = signal["close"]
    atr = signal["atr14"]

    stop_distance = CONFIG["ATR_STOPLOSS_MULTIPLIER"] * atr
    stop_loss = entry - stop_distance
    take_profit = entry + (stop_distance * CONFIG["RISK_REWARD_RATIO"])

    risk_amount_usd = capital_usd * CONFIG["RISK_PER_TRADE_PCT"]

    if stop_distance <= 0:
        return None

    full_shares = risk_amount_usd / stop_distance
    first_tranche_shares = int(full_shares * CONFIG["FIRST_TRANCHE_PCT"])

    if first_tranche_shares < 1:
        return None  # เงินทุนไม่พอซื้อแม้แต่ 1 หุ้นตามกฎ risk management

    # --- Trailing Stop แนะนำ (สำหรับตั้งใน Webull แทน Take Profit ตายตัว) ---
    # ระยะ Trail เท่ากับ ATR_MULTIPLIER เท่าของ ATR เดียวกับที่ใช้คำนวณ SL เริ่มต้น
    # (ปรับแยกได้ที่ CONFIG["TRAILING_STOP_ATR_MULTIPLIER"] ถ้าอยากให้ตามห่าง/ใกล้กว่าเดิม)
    trailing_stop_distance = CONFIG["TRAILING_STOP_ATR_MULTIPLIER"] * atr
    trailing_stop_pct = (trailing_stop_distance / entry) * 100

    return {
        "entry": round(entry, 2),
        "take_profit": round(take_profit, 2),
        "stop_loss": round(stop_loss, 2),
        "shares_first_tranche": first_tranche_shares,
        "risk_amount_usd": round(risk_amount_usd, 2),
        "position_value_usd": round(first_tranche_shares * entry, 2),
        "trailing_stop_amount": round(trailing_stop_distance, 2),
        "trailing_stop_pct": round(trailing_stop_pct, 2),
    }


# ==============================================================================
# 6b. Market Regime Filter — เช็คเทรนด์ของดัชนีแม่ (SPY) ก่อนอนุญาตออกสัญญาณ
# ==============================================================================

def check_market_regime() -> dict:
    """
    ดึงข้อมูล SPY (ETF ที่อ้างอิงดัชนี S&P 500) มาเช็คเทรนด์ภาพรวมตลาด
    ถ้า SPY เป็นขาลง สัญญาณซื้อรายตัวที่เจอ อาจเป็นแค่การเด้งเพื่อลงต่อ (Bull Trap)
    คืนค่า dict บอกว่าตลาดรวมเป็นขาขึ้นหรือไม่ พร้อมคำอธิบาย
    """
    try:
        spy = yf.download(
            tickers="SPY", period=CONFIG["HISTORY_PERIOD"], interval="1d",
            auto_adjust=True, progress=False,
        )
        if spy.empty or len(spy) < CONFIG["MIN_BARS_REQUIRED"]:
            return {"is_bullish": None, "description": "ไม่สามารถดึงข้อมูล SPY ได้เพียงพอ ข้ามการเช็ค Market Regime"}

        # yf.download คืน MultiIndex columns เมื่อ tickers เป็น string เดี่ยวในบางเวอร์ชัน — flatten กันไว้
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)

        spy_ind = compute_indicators(spy)
        last = spy_ind.iloc[-1]
        is_bullish = bool(last["Close"] > last["EMA20"] > last["EMA50"])

        desc = (
            f"SPY Close ${last['Close']:.2f} "
            f"{'>' if last['Close'] > last['EMA20'] else '<'} EMA20 ${last['EMA20']:.2f} "
            f"{'>' if last['EMA20'] > last['EMA50'] else '<'} EMA50 ${last['EMA50']:.2f}"
        )
        return {"is_bullish": is_bullish, "description": desc}
    except Exception as e:
        return {"is_bullish": None, "description": f"เช็ค Market Regime ไม่สำเร็จ: {e}"}


# ==============================================================================
# 6c. Earnings Gap Risk Check — เช็ควันประกาศงบของหุ้นที่ผ่านเกณฑ์แล้วเท่านั้น
# ==============================================================================

def get_days_until_earnings(ticker: str):
    """
    เช็คว่าหุ้นตัวนี้จะประกาศงบการเงินอีกกี่วัน (นับรวมวันนี้ = 0)
    คืนค่าเป็นจำนวนวัน (int, >= 0) หรือ None ถ้าไม่มีข้อมูล/ดึงไม่สำเร็จ/ไม่มีประกาศงบในเร็วๆ นี้
    """
    try:
        t = yf.Ticker(ticker)
        edates = t.get_earnings_dates(limit=4)
        if edates is None or edates.empty:
            return None

        now = pd.Timestamp.now(tz=edates.index.tz) if edates.index.tz is not None else pd.Timestamp.now()
        upcoming = edates[edates.index >= now]
        if upcoming.empty:
            return None

        next_date = upcoming.index.min()
        return max(0, (next_date - now).days)
    except Exception:
        # ดึงข้อมูล earnings ไม่สำเร็จ (yfinance ไม่การันตีว่าจะมีข้อมูลนี้ทุกตัว) — ข้ามไปแบบเงียบๆ
        return None


def check_earnings_risk(ticker: str, warning_days: int) -> str:
    """
    คืนค่าข้อความคำเตือน Gap Risk ถ้าใกล้ประกาศงบภายใน warning_days วัน
    หรือ "" ถ้าไม่พบความเสี่ยง/ดึงข้อมูลไม่ได้
    """
    days_until = get_days_until_earnings(ticker)
    if days_until is not None and days_until <= warning_days:
        return f"⚠️ ใกล้ประกาศงบใน {days_until} วัน เสี่ยง Gap ทะลุ Stoploss"
    return ""


def is_earnings_blackout(ticker: str, blackout_days: int):
    """
    เช็คว่าหุ้นตัวนี้อยู่ในช่วง 'Earnings Blackout' หรือไม่ (ใกล้ประกาศงบเกินกว่าจะรับความเสี่ยง Gap ได้)
    ถ้าใช่ ให้ตัดออกจากผลลัพธ์ไปเลย แทนที่จะแค่เตือน (ป้องกันเคสแบบ C ที่ชน SL เพราะ Earnings Gap)
    คืนค่า (is_blackout: bool, days_until: int หรือ None)
    """
    days_until = get_days_until_earnings(ticker)
    if days_until is not None and days_until <= blackout_days:
        return True, days_until
    return False, days_until


# ==============================================================================
# 6d. Sanity Check — เตือนถ้ารันระหว่างตลาดสหรัฐฯ ยังเปิดอยู่ (แท่งเทียนยังไม่ปิด)
# ==============================================================================

def check_market_hours_warning() -> str:
    """
    ตลาดหุ้นสหรัฐฯ (NYSE/NASDAQ) เปิด 09:30-16:00 ET ซึ่งเทียบเวลาไทย (UTC+7)
    ประมาณ 20:30-03:00 (ข้ามคืน) ขึ้นกับช่วง Daylight Saving Time
    ถ้ารันสคริปต์ในช่วงนี้ แท่งเทียนรายวันล่าสุดยังไม่ปิด ค่า Indicator อาจเปลี่ยนได้อีก
    คืนค่าข้อความเตือน หรือ "" ถ้าอยู่นอกช่วงเวลาตลาดเปิด
    """
    now = datetime.now().time()
    market_open_thai_start = datetime.strptime("20:30", "%H:%M").time()
    market_open_thai_end = datetime.strptime("03:00", "%H:%M").time()

    # ช่วงเวลาที่ข้ามเที่ยงคืน (20:30 -> 24:00 -> 03:00)
    is_market_hours = now >= market_open_thai_start or now <= market_open_thai_end

    if is_market_hours:
        return (
            "⚠️  [SANITY CHECK] ขณะนี้อยู่ในช่วงเวลาที่ตลาดหุ้นสหรัฐฯ น่าจะยังเปิดอยู่ "
            "(ประมาณ 20:30-03:00 น. เวลาไทย ผันแปรตาม DST)\n"
            "    แท่งเทียนรายวันล่าสุด (วันนี้) ยังไม่ปิดแท่งจริง ค่า EMA/MACD ที่คำนวณได้"
            " เป็นค่า ณ ขณะนี้เท่านั้น\n"
            "    และอาจเปลี่ยนแปลง หรือสัญญาณที่เจอวันนี้อาจหายไปเมื่อตลาดปิดจริง "
            "แนะนำให้รันอีกครั้งหลังตลาดปิด (Post-Market) เพื่อความแม่นยำ"
        )
    return ""


# ==============================================================================
# 6e. Auto Backtest — เช็คผลหุ้น Top N ของ "รอบก่อนหน้า" อัตโนมัติทุกครั้งที่รัน
# ==============================================================================

def find_previous_top_picks_file() -> str:
    """
    ค้นหาไฟล์ scan_result_top{N}_*.csv ที่ถูกสร้างไว้ล่าสุดในโฟลเดอร์เดียวกัน
    (คือผลลัพธ์ Top N ของการสแกนรอบก่อนหน้า ที่ยังไม่เคย backtest)
    คืนค่า path ของไฟล์ หรือ None ถ้าไม่พบไฟล์เลย (เช่น รันครั้งแรก)
    """
    candidates = glob.glob("scan_result_top*_*.csv")
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _parse_entry_date_from_filename(filepath: str):
    """แกะวันที่ออกจากชื่อไฟล์ เช่น scan_result_top5_20260713_1117.csv -> 2026-07-13"""
    match = re.search(r"(\d{8})_(\d{4})\.csv$", os.path.basename(filepath))
    if not match:
        return None
    date_str = match.group(1)
    return datetime.strptime(date_str, "%Y%m%d")


def backtest_previous_picks(prev_file: str) -> pd.DataFrame:
    """
    เช็คผลลัพธ์จริงของหุ้นที่เคยแนะนำไว้ในไฟล์ prev_file:
    - ดึงราคาย้อนหลังตั้งแต่วันที่แนะนำ (entry_date) จนถึงวันนี้
    - เช็คว่าราคาเคยแตะ Take Profit (High >= TP) หรือ Stop Loss (Low <= SL) หรือไม่
      โดยดูวันไหนเกิดก่อนใน timeline (ถ้าชนทั้งคู่ในวันเดียวกัน ถือว่า SL โดนก่อนเพื่อความระมัดระวัง)
    - ถ้ายังไม่ชนทั้งคู่ ถือว่า "ยังเปิดออเดอร์อยู่" และคำนวณกำไร/ขาดทุนแบบ Mark-to-Market
      จากราคาปิดล่าสุด
    คืนค่าเป็น DataFrame พร้อมผลลัพธ์รายตัว
    """
    entry_date = _parse_entry_date_from_filename(prev_file)
    if entry_date is None:
        print(f"[WARN] ไม่สามารถแกะวันที่จากชื่อไฟล์ {prev_file} ได้ ข้ามการ backtest")
        return pd.DataFrame()

    prev_df = pd.read_csv(prev_file)
    results = []

    for _, row in prev_df.iterrows():
        ticker = row["Ticker"]
        entry = row["ราคาซื้อแนะนำ ($)"]
        tp = row["ราคาเป้าหมาย ($)"]
        sl = row["จุดขาย Stoploss ($)"]

        # [ROBUSTNESS FIX] อ่านจำนวนหุ้นจากคอลัมน์ภาษาอังกฤษ "shares" เป็นหลัก (ไม่ผูกกับ
        # ชื่อคอลัมน์ภาษาไทยตรงเป๊ะ) — เผื่อไฟล์ scan_result_top*.csv เก่าที่ยังไม่มีคอลัมน์นี้
        # (สร้างก่อนอัปเดตนี้) ให้ fallback ไปอ่านคอลัมน์ภาษาไทยแทน แล้วถ้าไม่เจอทั้งคู่ ให้ข้าม
        # แถวนั้นไปพร้อมแจ้งเตือน แทนที่จะปล่อยให้ KeyError ทำให้ทั้งโปรแกรมพัง
        if "shares" in row.index and pd.notna(row["shares"]):
            shares = row["shares"]
        elif "จำนวนซื้อ ไม้แรก (50%)" in row.index and pd.notna(row["จำนวนซื้อ ไม้แรก (50%)"]):
            shares = row["จำนวนซื้อ ไม้แรก (50%)"]
        else:
            print(f"[WARN] ไม่พบคอลัมน์จำนวนหุ้นของ {ticker} ในไฟล์ {prev_file} ข้ามแถวนี้ไป")
            continue

        try:
            hist = yf.download(
                tickers=ticker,
                start=entry_date.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            hist = hist.dropna(how="all")
        except Exception as e:
            print(f"[WARN] ดึงข้อมูล {ticker} เพื่อ backtest ไม่สำเร็จ: {e}")
            continue

        if hist.empty:
            continue

        outcome = "OPEN"
        exit_price = None
        exit_date = None

        for date, day in hist.iterrows():
            hit_sl = day["Low"] <= sl
            hit_tp = day["High"] >= tp
            # ถ้าชนทั้งคู่ในวันเดียวกัน ไม่รู้ว่าอันไหนเกิดก่อนจริงๆ (ไม่มีข้อมูล intraday)
            # จึงถือว่า SL โดนก่อนเสมอ เพื่อความระมัดระวัง (Conservative Assumption)
            if hit_sl:
                outcome = "LOSS"
                exit_price = sl
                exit_date = date
                break
            if hit_tp:
                outcome = "WIN"
                exit_price = tp
                exit_date = date
                break

        last_close = hist.iloc[-1]["Close"]
        days_held = (hist.index[-1] - hist.index[0]).days + 1

        if outcome == "OPEN":
            pnl_per_share = last_close - entry
            current_price = last_close
        else:
            pnl_per_share = exit_price - entry
            current_price = exit_price

        pnl_pct = (pnl_per_share / entry) * 100
        pnl_usd = pnl_per_share * shares

        results.append({
            "backtest_run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "source_file": os.path.basename(prev_file),
            "Ticker": ticker,
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "entry_price": entry,
            "take_profit": tp,
            "stop_loss": sl,
            "current_or_exit_price": round(current_price, 2),
            "days_held": days_held,
            "outcome": outcome,
            "pnl_pct": round(pnl_pct, 2),
            "pnl_usd": round(pnl_usd, 2),
            "shares": shares,
        })

    return pd.DataFrame(results)


def save_backtest_to_history(backtest_df: pd.DataFrame, history_file: str):
    """บันทึกผล backtest รอบนี้ต่อท้ายไฟล์ประวัติสะสม (สร้างไฟล์ใหม่ถ้ายังไม่มี)"""
    if backtest_df.empty:
        return
    file_exists = os.path.exists(history_file)
    backtest_df.to_csv(history_file, mode="a", index=False, header=not file_exists, encoding="utf-8-sig")


def print_backtest_summary(backtest_df: pd.DataFrame, history_file: str):
    """แสดงผล backtest ของรอบนี้ พร้อมสถิติสะสม Win Rate จากประวัติทั้งหมด"""
    if backtest_df.empty:
        return

    print("\n" + "=" * 78)
    print("📊 AUTO BACKTEST — ผลลัพธ์จริงของ Top Picks รอบก่อนหน้า")
    print("=" * 78)

    display_cols = ["Ticker", "entry_date", "entry_price", "take_profit", "stop_loss",
                     "current_or_exit_price", "days_held", "outcome", "pnl_pct", "pnl_usd"]
    try:
        from tabulate import tabulate
        print(tabulate(backtest_df[display_cols], headers="keys", tablefmt="grid", showindex=False))
    except ImportError:
        print(backtest_df[display_cols].to_string(index=False))

    total_pnl = backtest_df["pnl_usd"].sum()
    total_invested = (backtest_df["entry_price"] * backtest_df["shares"]).sum()
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0
    n_win = (backtest_df["outcome"] == "WIN").sum()
    n_loss = (backtest_df["outcome"] == "LOSS").sum()
    n_open = (backtest_df["outcome"] == "OPEN").sum()

    print(f"\nรอบนี้: WIN {n_win} | LOSS {n_loss} | ยังเปิดอยู่ (OPEN) {n_open}")
    print(f"กำไร/ขาดทุนรวมรอบนี้: ${total_pnl:,.2f}  ({total_pnl_pct:+.2f}%)")

    # --- สถิติสะสมจากประวัติทั้งหมด (นับทุกสถานะที่ "ปิดจบแล้ว" คือ WIN/LOSS/TIME_EXIT/MOMENTUM_EXIT) ---
    if os.path.exists(history_file):
        hist = pd.read_csv(history_file)
        hist_latest = hist.sort_values("backtest_run_date").drop_duplicates(
            subset=["source_file", "Ticker"], keep="last"
        )
        closed_statuses = ["WIN", "LOSS", "TIME_EXIT", "MOMENTUM_EXIT"]
        closed = hist_latest[hist_latest["outcome"].isin(closed_statuses)]
        if not closed.empty:
            # ใช้ pnl_usd > 0 ตัดสิน "ชนะ" จริง แทนที่จะดูแค่ label เพราะ TIME_EXIT/MOMENTUM_EXIT
            # ก็อาจปิดกำไรหรือขาดทุนก็ได้ ไม่ได้แปลว่าแพ้เสมอไป
            cum_win = int((closed["pnl_usd"] > 0).sum())
            cum_total = len(closed)
            cum_win_rate = cum_win / cum_total * 100
            cum_pnl = closed["pnl_usd"].sum()

            # [FEATURE] เพิ่มกำไร/ขาดทุนสะสมในหน่วย % (ถ่วงน้ำหนักตามเงินลงทุนจริงต่อไม้)
            # เพื่อให้เห็นภาพว่ากลยุทธ์นี้ทำเงินสุทธิได้กี่เปอร์เซ็นต์ ไม่ใช่แค่จำนวนเงิน
            cum_invested = (closed["entry_price"] * closed["shares"]).sum()
            cum_pnl_pct_weighted = (cum_pnl / cum_invested * 100) if cum_invested else 0
            cum_pnl_pct_avg = closed["pnl_pct"].mean()

            n_time_exit = int((closed["outcome"] == "TIME_EXIT").sum())
            n_momentum_exit = int((closed["outcome"] == "MOMENTUM_EXIT").sum())

            print(f"\n📈 สถิติสะสมทั้งหมด (เฉพาะเทรดที่ปิดแล้ว — WIN/LOSS/TIME_EXIT/MOMENTUM_EXIT): "
                  f"{cum_win}/{cum_total} ชนะ = Win Rate {cum_win_rate:.1f}%")
            print(f"   กำไร/ขาดทุนสะสม: ${cum_pnl:,.2f}  "
                  f"({cum_pnl_pct_weighted:+.2f}% ถ่วงน้ำหนักตามเงินลงทุน  |  "
                  f"เฉลี่ย {cum_pnl_pct_avg:+.2f}% ต่อเทรด)")
            if n_time_exit or n_momentum_exit:
                print(f"   ในจำนวนนี้ตัดจบด้วย Time Stop {n_time_exit} เทรด "
                      f"และ Momentum Exit {n_momentum_exit} เทรด")
        else:
            print("\n📈 สถิติสะสม: ยังไม่มีเทรดที่ปิดจบ (ทุกตัวยังเปิดอยู่)")

    print("=" * 78)


# ==============================================================================
# 6e. Active Trades Tracker — ติดตามหุ้นที่ยังถือ (OPEN) อยู่จากประวัติสะสมทั้งหมด
# ==============================================================================
#
# ต่างจาก AUTO BACKTEST ด้านบน (ซึ่งเช็คแค่ไฟล์ Top-N ของ "รอบก่อนหน้ารอบเดียว" แล้ว
# append แถวใหม่ทุกครั้ง) ฟังก์ชันนี้จะไล่ดู backtest_history.csv ทั้งไฟล์ ดึงเฉพาะแถวที่
# ยังมีสถานะ outcome == "OPEN" ล่าสุดของแต่ละคู่ (source_file, Ticker) มาเช็คว่าราคาล่าสุด
# ชน TP/SL/Time Stop/Momentum Exit หรือยัง แล้วอัปเดตแถวเดิมนั้นกลับเข้าไฟล์ (ไม่สร้างแถวซ้ำ)
# ==============================================================================

def track_active_portfolio_trades(history_file: str) -> pd.DataFrame:
    """
    ติดตามและอัปเดตสถานะหุ้นทุกตัวที่ยังเปิดออเดอร์อยู่ (outcome == "OPEN") ใน history_file

    ขั้นตอน:
      1. อ่าน history_file ทั้งหมด แล้วหาแถว "ล่าสุด" ของแต่ละคู่ (source_file, Ticker)
         ที่ outcome ยังเป็น "OPEN" (กันเคสมีหลายแถวซ้ำจากการรันหลายรอบ)
      2. ดึงราคาย้อนหลังของแต่ละ ticker (ช่วงยาวพอสำหรับคำนวณ MACD ให้แม่นยำ)
      3. เช็คว่านับจาก entry_date เป็นต้นมา ราคาเคยชน take_profit (High >= TP) หรือ
         stop_loss (Low <= SL) หรือยัง (ถ้าชนพร้อมกันในวันเดียว ถือว่า SL โดนก่อน
         เพื่อความระมัดระวัง เหมือนกับ backtest_previous_picks)
      4. ถ้ายังไม่ชน TP/SL แต่ days_held (นับจาก entry_date ถึงวันนี้แบบวันทำการ) เกิน
         ACTIVE_TRADE_TIME_STOP_DAYS -> บังคับปิดเป็น "TIME_EXIT" ที่ราคาปิดล่าสุด
      5. ถ้ายังไม่ชนเงื่อนไขไหนเลย แต่แท่งล่าสุด MACD ตัดหลุดต่ำกว่า MACD_Signal
         -> ปิดเป็น "MOMENTUM_EXIT" ที่ราคาปิดล่าสุด
      6. คำนวณ pnl_pct / pnl_usd แล้วอัปเดตกลับลงในแถวเดิมของ history_file (overwrite ทั้งไฟล์)
      7. แสดง Dashboard สรุปหุ้นกลุ่มนี้ (ใช้ tabulate ถ้ามี)

    คืนค่า: DataFrame ของแถวที่ถูกอัปเดตในรอบนี้ (empty DataFrame ถ้าไม่มีอะไรให้ติดตาม)
    """
    if not os.path.exists(history_file):
        print(f"[INFO] ยังไม่พบไฟล์ {history_file} — ยังไม่มีประวัติออเดอร์ให้ติดตาม (อาจเป็นการรันครั้งแรก)")
        return pd.DataFrame()

    hist = pd.read_csv(history_file)
    if hist.empty or "outcome" not in hist.columns:
        return pd.DataFrame()

    # --- [FIX] กรองเอาเฉพาะแถวที่ outcome เป็น "OPEN" ตั้งแต่แรกก่อน แล้วค่อย groupby หาแถว
    # ล่าสุดของแต่ละคู่ (source_file, Ticker) ภายในกลุ่ม OPEN เท่านั้น — ป้องกันไม่ให้หยิบ
    # หุ้นที่เคยถูกปิดสถานะไปแล้ว (TIME_EXIT/MOMENTUM_EXIT/WIN/LOSS) กลับมาประมวลผลซ้ำ แม้ว่า
    # จะมีแถวซ้ำซ้อนจากไฟล์ source_file ที่ชื่อซ้ำกันก็ตาม (แก้จากเดิมที่ groupby หาแถวล่าสุด
    # จากทุกสถานะก่อน แล้วค่อยกรอง OPEN ทีหลัง ซึ่งเสี่ยงเลือกแถวผิดถ้า sort ไม่ stable)
    open_only = hist[hist["outcome"] == "OPEN"]
    if open_only.empty:
        print("[INFO] ไม่มีหุ้นที่ยังเปิดออเดอร์อยู่ (OPEN) ให้ติดตามในขณะนี้")
        return pd.DataFrame()

    # kind="stable" การันตีว่าถ้า backtest_run_date ซ้ำกันเป๊ะ ลำดับเดิม (ตามที่ถูก append
    # ลงไฟล์จริง) จะไม่ถูกสลับมั่ว ทำให้ .tail(1) ได้แถวที่ append ล่าสุดจริงๆ เสมอ
    open_sorted = open_only.sort_values("backtest_run_date", kind="stable")
    latest_positions_idx = open_sorted.groupby(["source_file", "Ticker"]).tail(1).index
    open_rows = hist.loc[latest_positions_idx]

    if open_rows.empty:
        print("[INFO] ไม่มีหุ้นที่ยังเปิดออเดอร์อยู่ (OPEN) ให้ติดตามในขณะนี้")
        return pd.DataFrame()

    print(f"[INFO] กำลังติดตามสถานะหุ้นที่ยังถืออยู่ {len(open_rows)} ตัว จากประวัติสะสม ...")

    time_stop_days = CONFIG["ACTIVE_TRADE_TIME_STOP_DAYS"]
    lookback_period = CONFIG["ACTIVE_TRADE_LOOKBACK_PERIOD"]
    today = pd.Timestamp.now().normalize()

    updated_idx = []  # เก็บ index ของแถวที่อัปเดตสำเร็จ เพื่อดึงมาแสดง dashboard ทีหลัง

    for row_idx, row in open_rows.iterrows():
        ticker = row["Ticker"]
        entry_price = float(row["entry_price"])
        tp = float(row["take_profit"])
        sl = float(row["stop_loss"])
        shares = float(row["shares"])
        entry_date = pd.to_datetime(row["entry_date"])

        # [FIX] ถ้าข้อมูลตั้งต้นของแถวนี้เพี้ยน/ขาดหายมาตั้งแต่ใน CSV เอง (เช่น entry_price
        # หรือ shares เป็น NaN) ให้ข้ามไปเลยตั้งแต่ต้น ไม่ต้องไปดึงราคาหรือคำนวณต่อ เพราะยังไง
        # ผลลัพธ์ก็จะเป็น nan อยู่ดี ดีกว่าปล่อยให้ nan ไหลไปจนถึงขั้นตอนคำนวณ PnL
        if any(pd.isna(v) for v in (entry_price, tp, sl, shares)) or pd.isna(entry_date):
            print(f"[WARN] {ticker}: ข้อมูลในแถว history ไม่ครบ/เพี้ยน (entry_price/tp/sl/shares/entry_date) "
                  f"— ข้ามแถวนี้ไปก่อน")
            continue

        try:
            # ดึงข้อมูลย้อนหลังยาวพอ (เช่น 1 ปี) เพื่อให้ EMA/MACD คำนวณได้แม่นยำ
            # ไม่ใช่ดึงแค่ตั้งแต่ entry_date เพราะจะสั้นเกินไปสำหรับ MACD_SLOW=26
            data = yf.download(
                tickers=ticker,
                period=lookback_period,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.dropna(how="all")
            # [FIX] บางครั้ง yfinance คืนแท่งล่าสุดมาโดยที่ Close ยังเป็น NaN (เช่นข้อมูล
            # intraday ที่ยังโหลดไม่สมบูรณ์) ถ้าไม่กรองออก last_close จะเป็น NaN แล้วไหล
            # เข้าไปเป็น exit_price / pnl_pct / pnl_usd ทำให้ตาราง Dashboard ขึ้น nan
            data = data.dropna(subset=["Close"])
        except Exception as e:
            print(f"[WARN] ดึงข้อมูล {ticker} เพื่อติดตามสถานะไม่สำเร็จ: {e} — ข้ามตัวนี้ไปก่อน")
            continue

        if data.empty:
            print(f"[WARN] ไม่มีข้อมูลราคาของ {ticker} — ข้ามตัวนี้ไปก่อน")
            continue

        # เฉพาะแท่งเทียนตั้งแต่วันเข้าซื้อเป็นต้นมา ใช้เช็คว่าชน TP/SL หรือยัง
        data_since_entry = data[data.index >= entry_date]
        if data_since_entry.empty:
            continue

        outcome = "OPEN"
        exit_price = None

        # --- 1) เช็ค TP / SL ทีละแท่งตามลำดับเวลา (ชนพร้อมกัน = ถือว่า SL โดนก่อน) ---
        for date, day in data_since_entry.iterrows():
            if day["Low"] <= sl:
                outcome, exit_price = "LOSS", sl
                break
            if day["High"] >= tp:
                outcome, exit_price = "WIN", tp
                break

        last_close = float(data.iloc[-1]["Close"])
        # นับ "วันทำการ" ตั้งแต่ entry_date ถึงวันนี้ (ไม่รวมวันเสาร์-อาทิตย์)
        days_held = int(np.busday_count(entry_date.date(), today.date()))

        # --- 2) Time Stop: ถือเกิน ACTIVE_TRADE_TIME_STOP_DAYS วันทำการแล้วยังไม่ชน TP/SL ---
        if outcome == "OPEN" and days_held > time_stop_days:
            outcome = "TIME_EXIT"
            exit_price = last_close

        # --- 3) Momentum Exit: แท่งล่าสุด MACD ตัดหลุดต่ำกว่า MACD_Signal ---
        if outcome == "OPEN":
            try:
                df_ind = compute_indicators(data.copy())
                df_ind = df_ind.dropna(subset=["MACD", "MACD_Signal"])
                if not df_ind.empty:
                    last_ind = df_ind.iloc[-1]
                    if last_ind["MACD"] < last_ind["MACD_Signal"]:
                        outcome = "MOMENTUM_EXIT"
                        exit_price = last_close
            except Exception as e:
                print(f"[WARN] คำนวณ MACD ของ {ticker} เพื่อเช็ค Momentum Exit ไม่สำเร็จ: {e}")

        current_or_exit_price = last_close if outcome == "OPEN" else exit_price

        # [FIX] ด่านสุดท้ายก่อนคำนวณ PnL: ถ้า current_or_exit_price หลุดมาเป็น None/NaN
        # ไม่ว่าจะจากสาเหตุใด (ข้อมูลราคาขาดหาย, บล็อกคำนวณ MACD error แล้วเผลอลืมเซ็ต
        # exit_price ในโค้ดเวอร์ชันหลังจากนี้ ฯลฯ) ให้ fallback ไปใช้ last_close ที่เรารู้ว่า
        # ไม่ใช่ NaN แน่นอน (ผ่าน dropna(subset=["Close"]) มาแล้วด้านบน) แทนที่จะปล่อยให้
        # nan ไหลลงตาราง Dashboard และไฟล์ CSV
        if current_or_exit_price is None or pd.isna(current_or_exit_price):
            print(f"[WARN] {ticker}: exit_price ว่างเปล่า/เป็น nan — ใช้ last_close (${last_close:.2f}) แทน")
            current_or_exit_price = last_close

        pnl_per_share = current_or_exit_price - entry_price
        pnl_pct = (pnl_per_share / entry_price) * 100
        pnl_usd = pnl_per_share * shares

        # --- อัปเดตกลับเข้าแถวเดิมใน DataFrame หลัก (ใช้ row_idx อ้างอิงตำแหน่งเดิม) ---
        hist.at[row_idx, "current_or_exit_price"] = round(current_or_exit_price, 2)
        hist.at[row_idx, "days_held"] = days_held
        hist.at[row_idx, "outcome"] = outcome
        hist.at[row_idx, "pnl_pct"] = round(pnl_pct, 2)
        hist.at[row_idx, "pnl_usd"] = round(pnl_usd, 2)
        updated_idx.append(row_idx)

    if not updated_idx:
        print("[WARN] ไม่สามารถอัปเดตสถานะหุ้นตัวใดได้เลย (อาจดึงข้อมูลราคาไม่สำเร็จทุกตัว)")
        return pd.DataFrame()

    # --- บันทึกไฟล์ทั้งหมดกลับไป (overwrite) หลังอัปเดตแถวที่เกี่ยวข้องเรียบร้อยแล้ว ---
    hist.to_csv(history_file, index=False, encoding="utf-8-sig")

    updated_df = hist.loc[updated_idx].reset_index(drop=True)

    # --- แสดง Dashboard สรุปหุ้นกลุ่มที่ยังถือ/เพิ่งปิดในรอบนี้ ---
    print("\n" + "=" * 78)
    print("📂 ACTIVE TRADES TRACKER — สถานะหุ้นที่ยังถืออยู่ (อัปเดตก่อนสแกนตัวใหม่)")
    print("=" * 78)

    display_cols = ["Ticker", "entry_date", "entry_price", "take_profit", "stop_loss",
                     "current_or_exit_price", "days_held", "outcome", "pnl_pct", "pnl_usd"]
    display_cols = [c for c in display_cols if c in updated_df.columns]
    try:
        from tabulate import tabulate
        print(tabulate(updated_df[display_cols], headers="keys", tablefmt="grid", showindex=False))
    except ImportError:
        print(updated_df[display_cols].to_string(index=False))
        print("\n[TIP] ติดตั้ง `pip install tabulate` แล้วรันใหม่ จะได้ตารางที่มีเส้นกรอบสวยขึ้น")

    n_win = int((updated_df["outcome"] == "WIN").sum())
    n_loss = int((updated_df["outcome"] == "LOSS").sum())
    n_time = int((updated_df["outcome"] == "TIME_EXIT").sum())
    n_mom = int((updated_df["outcome"] == "MOMENTUM_EXIT").sum())
    n_open = int((updated_df["outcome"] == "OPEN").sum())
    total_pnl = updated_df["pnl_usd"].sum()

    print(f"\nสรุปกลุ่มนี้: WIN {n_win} | LOSS {n_loss} | TIME_EXIT {n_time} | "
          f"MOMENTUM_EXIT {n_mom} | ยังเปิดอยู่ (OPEN) {n_open}")
    print(f"กำไร/ขาดทุนรวม (Mark-to-Market เฉพาะกลุ่มนี้): ${total_pnl:,.2f}")
    print("=" * 78)

    return updated_df


# ==============================================================================
# 6f. Historical Backtest Engine — ทดสอบย้อนหลังเต็มรูปแบบ (ไม่ใช่แค่ Forward-Tracking)
# ==============================================================================
#
# แก้จุดอ่อน 3 ข้อของระบบ Backtest แบบเดิม (Auto Backtest ด้านบนซึ่งเป็น Forward-Tracking):
#   1. [ENTRY REALISM] เดิมใช้ราคาปิดของวันที่เจอสัญญาณเป็นราคาซื้อ (ซื้อไม่ทันจริง)
#      แก้ไข: จำลองซื้อที่ราคาเปิดของ "วันถัดไป" (Next-Day Open) + Slippage
#   2. [GAP-AWARE STOP] เดิมสมมติว่าขายได้ที่ราคา SL เป๊ะๆ เสมอ
#      แก้ไข: ถ้าวันถัดมาราคาเปิดกระโดดต่ำกว่า SL (Gap Down) ต้องตัดขาดทุนที่ราคา Open จริง
#      ไม่ใช่ราคา SL ที่ตั้งไว้ (สะท้อนความเจ็บปวดจริงของ Gap Risk)
#   3. [FULL HISTORY] เดิมสะสมสถิติแค่วันละครั้งจาก Auto Backtest ต้องรอเป็นปีกว่า Win Rate จะนิ่ง
#      แก้ไข: ลูปหาสัญญาณที่เคยเกิดขึ้นจริงย้อนหลัง 2-3 ปี แล้วจำลองเทรดทุกครั้งที่เจอสัญญาณ
#      (แบบไม่ให้โพซิชันซ้อนทับกันในหุ้นตัวเดียวกัน) เพื่อให้ได้ Win Rate ที่มีนัยสำคัญทางสถิติจริง
# ==============================================================================

def _vectorized_signal_mask(df_ind: pd.DataFrame) -> pd.DataFrame:
    """
    หาวันที่ "เกิดสัญญาณซื้อ" ทั้งหมดในประวัติศาสตร์ของหุ้นตัวหนึ่ง แบบ vectorized (เร็วกว่าลูปทีละวันมาก)
    ใช้เงื่อนไขเดียวกับ check_signal() เป๊ะๆ (รวมบั๊กไฟ MACD cross ที่แก้ไปแล้ว) คืนค่า df_ind
    ที่มีคอลัมน์ 'signal' (bool) และ 'signal_type' (macd/ema20/both) เพิ่มเข้ามา
    """
    df = df_ind.copy()

    liquidity_ok = df["AvgVol30"] > CONFIG["MIN_AVG_VOLUME_30D"]
    price_ok = df["Close"].between(CONFIG["MIN_PRICE"], CONFIG["MAX_PRICE"])
    uptrend_ok = (df["Close"] > df["EMA20"]) & (df["EMA20"] > df["EMA50"])

    cross_today = (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1)) & (df["MACD"] > df["MACD_Signal"])
    lookback = CONFIG["MACD_CROSS_LOOKBACK_DAYS"]
    cross_recent = cross_today.rolling(window=lookback, min_periods=1).max().astype(bool)
    macd_still_above = df["MACD"] > df["MACD_Signal"]
    macd_trigger = cross_recent & macd_still_above

    dist_from_ema20_pct = (df["Close"] - df["EMA20"]).abs() / df["EMA20"]
    ema20_bounce = (df["Close"] > df["EMA20"]) & (dist_from_ema20_pct <= CONFIG["EMA20_BOUNCE_THRESHOLD_PCT"])

    enough_bars = pd.Series(np.arange(len(df)) >= CONFIG["MIN_BARS_REQUIRED"], index=df.index)

    signal = liquidity_ok & price_ok & uptrend_ok & (macd_trigger | ema20_bounce) & enough_bars

    df["signal"] = signal
    df["signal_type"] = np.select(
        [macd_trigger & ema20_bounce, macd_trigger, ema20_bounce],
        ["both", "macd", "ema20"],
        default="none",
    )
    return df


def backtest_historical_signals(df_ind: pd.DataFrame, ticker: str) -> list:
    """
    จำลองการเทรดทุกครั้งที่เคยเกิดสัญญาณในประวัติศาสตร์ของหุ้นตัวนี้ โดย:
    - เข้าซื้อที่ Open ของ "วันถัดไป" จากวันที่เกิดสัญญาณ + Slippage (ไม่ใช้ราคาปิดวันสัญญาณ)
    - Stop Loss / Take Profit คำนวณจาก ATR ของวันที่เกิดสัญญาณ เหมือนที่ Scanner ใช้จริง
    - เช็คการออกโพซิชันแบบ Gap-Aware: ถ้าวันไหน Open กระโดดต่ำกว่า SL ให้ตัดขาดทุนที่ Open จริง
      (ไม่ใช่ราคา SL ที่ตั้งไว้) ส่วน Take Profit ยังคงใช้ราคาที่ตั้งไว้ (เป็น Limit Order ปกติ)
    - ถือได้นานสุด MAX_HOLD_DAYS วัน ถ้ายังไม่ชน TP/SL ให้ปิดที่ราคาปิดวันนั้น (TIMEOUT)
    - ไม่เปิดโพซิชันซ้อนทับกัน (รอให้เทรดก่อนหน้าจบก่อน ถึงจะรับสัญญาณถัดไปได้)
    คืนค่าเป็น list ของ dict ผลการเทรดแต่ละครั้ง
    """
    df = _vectorized_signal_mask(df_ind)
    signal_indices = np.where(df["signal"].values)[0]

    trades = []
    next_available_idx = 0  # ดัชนีแถวที่เร็วที่สุดที่จะรับสัญญาณใหม่ได้ (ไม่ให้ซ้อนโพซิชัน)

    for sig_idx in signal_indices:
        if sig_idx < next_available_idx:
            continue  # ยังติดโพซิชันเก่าอยู่ ข้ามสัญญาณนี้ไป

        entry_idx = sig_idx + 1
        if entry_idx >= len(df):
            break  # ไม่มีข้อมูลวันถัดไปให้เข้าซื้อ (สัญญาณเกิดวันสุดท้ายของข้อมูลพอดี)

        signal_row = df.iloc[sig_idx]
        entry_open = df.iloc[entry_idx]["Open"]
        if pd.isna(entry_open) or entry_open <= 0:
            next_available_idx = entry_idx + 1
            continue

        # --- [FIX 1] เข้าซื้อที่ Open ของวันถัดไป + Slippage ---
        entry_price = entry_open * (1 + CONFIG["SLIPPAGE_PCT"])

        atr = signal_row["ATR14"]
        if pd.isna(atr) or atr <= 0:
            next_available_idx = entry_idx + 1
            continue

        stop_distance = CONFIG["ATR_STOPLOSS_MULTIPLIER"] * atr
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + stop_distance * CONFIG["RISK_REWARD_RATIO"]

        outcome, exit_price, exit_reason, exit_idx = None, None, None, None
        max_j = min(entry_idx + CONFIG["MAX_HOLD_DAYS"], len(df))

        # [AUDITED] ทุกเงื่อนไข Low <= stop_loss และ High >= take_profit ด้านล่างนี้ต้องมี
        # `break` ปิดท้ายเสมอ — ถ้าไม่มี ลูปจะเดินต่อไปเช็ควันถัดไปทั้งที่ออเดอร์ควรจบไปแล้ว
        # ทำให้ days_held/exit_price/outcome ผิดเพี้ยนและนับผลซ้ำซ้อน ตรวจสอบแล้วว่าทุกจุด
        # มี break ครบถ้วน (ทั้งแท่งแรกที่เข้าซื้อ และแท่งถัดๆ ไปแบบ Gap-Aware)
        for j in range(entry_idx, max_j):
            day = df.iloc[j]

            if j == entry_idx:
                if day["Low"] <= stop_loss:
                    outcome, exit_price, exit_reason = "LOSS", stop_loss, "SL"
                    exit_idx = j
                    break
                if day["High"] >= take_profit:
                    outcome, exit_price, exit_reason = "WIN", take_profit, "TP"
                    exit_idx = j
                    break
            else:
                # --- [FIX 2] Gap-Aware Stop: ถ้า Open วันนี้ต่ำกว่า SL ไปแล้ว ต้องขายที่ Open จริง ---
                if day["Open"] <= stop_loss:
                    outcome, exit_price, exit_reason = "LOSS", day["Open"], "GAP_SL"
                    exit_idx = j
                    break
                if day["Low"] <= stop_loss:
                    outcome, exit_price, exit_reason = "LOSS", stop_loss, "SL"
                    exit_idx = j
                    break
                if day["High"] >= take_profit:
                    outcome, exit_price, exit_reason = "WIN", take_profit, "TP"
                    exit_idx = j
                    break

        if outcome is None:
            exit_idx = max_j - 1
            exit_price = df.iloc[exit_idx]["Close"]
            exit_reason = "TIMEOUT"
            outcome = "WIN" if exit_price > entry_price else ("LOSS" if exit_price < entry_price else "BREAKEVEN")

        pnl_pct = (exit_price - entry_price) / entry_price * 100
        days_held = exit_idx - entry_idx + 1

        trades.append({
            "Ticker": ticker,
            "signal_date": df.index[sig_idx].strftime("%Y-%m-%d"),
            "signal_type": signal_row["signal_type"],
            "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 2),
            "take_profit": round(take_profit, 2),
            "stop_loss": round(stop_loss, 2),
            "exit_date": df.index[exit_idx].strftime("%Y-%m-%d"),
            "exit_price": round(exit_price, 2),
            "exit_reason": exit_reason,
            "days_held": days_held,
            "outcome": outcome,
            "pnl_pct": round(pnl_pct, 2),
        })

        next_available_idx = exit_idx + 1  # ห้ามรับสัญญาณใหม่จนกว่าเทรดนี้จะจบ

    return trades


def backtest_historical_signals_trailing(df_ind: pd.DataFrame, ticker: str) -> list:
    """
    เหมือน backtest_historical_signals() ทุกอย่าง (Next-Day Open Entry + Slippage, Gap-Aware Stop,
    ไม่ซ้อนโพซิชัน) ยกเว้นจุดเดียวคือ "ไม่มี Take Profit ตายตัว" — แทนที่ด้วย Trailing Stop:

    - SL เริ่มต้น = Entry - (TRAILING_STOP_ATR_MULTIPLIER * ATR ณ วันสัญญาณ) เหมือนเดิม
    - ทุกวันที่ถือต่อ (นับจากวันเข้าซื้อ): เช็คว่าราคาชน SL ปัจจุบันหรือยัง (แบบ Gap-Aware เหมือนเดิม)
      ถ้ายัง ให้ปรับ SL ขึ้นตามราคาสูงสุดที่เคยทำได้ (High สะสม) ลบระยะ Trail เดิม — แต่ SL จะ
      "เลื่อนขึ้นได้ทางเดียว" ห้ามลดต่ำกว่าที่เคยเลื่อนไปแล้ว (ล็อกกำไรที่ทำได้แล้วไว้)
    - ถือได้นานสุด MAX_HOLD_DAYS วันเหมือนเดิม (กันเงินจมค้างไม่มีที่สิ้นสุด)
    คืนค่าเป็น list ของ dict ผลการเทรดแต่ละครั้ง (schema เหมือน backtest_historical_signals
    แต่ไม่มีคอลัมน์ take_profit และ exit_reason จะเป็น TRAIL_SL / GAP_TRAIL_SL / TIMEOUT แทน)
    """
    df = _vectorized_signal_mask(df_ind)
    signal_indices = np.where(df["signal"].values)[0]

    trail_mult = CONFIG["TRAILING_STOP_ATR_MULTIPLIER"]
    trades = []
    next_available_idx = 0

    for sig_idx in signal_indices:
        if sig_idx < next_available_idx:
            continue

        entry_idx = sig_idx + 1
        if entry_idx >= len(df):
            break

        signal_row = df.iloc[sig_idx]
        entry_open = df.iloc[entry_idx]["Open"]
        if pd.isna(entry_open) or entry_open <= 0:
            next_available_idx = entry_idx + 1
            continue

        entry_price = entry_open * (1 + CONFIG["SLIPPAGE_PCT"])

        atr = signal_row["ATR14"]
        if pd.isna(atr) or atr <= 0:
            next_available_idx = entry_idx + 1
            continue

        trail_distance = trail_mult * atr
        current_stop = entry_price - trail_distance
        peak_price = entry_price

        exit_price, exit_reason, exit_idx = None, None, None
        max_j = min(entry_idx + CONFIG["MAX_HOLD_DAYS"], len(df))

        for j in range(entry_idx, max_j):
            day = df.iloc[j]

            if j == entry_idx:
                # วันเข้าซื้อ: ไม่มี gap เพราะเพิ่งเข้าที่ Open ของวันนี้เอง เช็คแค่ Low ทะลุ SL เริ่มต้นไหม
                if day["Low"] <= current_stop:
                    exit_price, exit_reason = current_stop, "TRAIL_SL"
                    exit_idx = j
                    break
                # ปรับ SL ขึ้นตาม High ของวันเข้าซื้อเอง (เผื่อวิ่งบวกทันทีในวันแรก)
                peak_price = max(peak_price, day["High"])
                current_stop = max(current_stop, peak_price - trail_distance)
            else:
                # --- Gap-Aware: ถ้า Open วันนี้ต่ำกว่า SL ที่เลื่อนมาแล้ว ต้องขายที่ Open จริง ---
                if day["Open"] <= current_stop:
                    exit_price, exit_reason = day["Open"], "GAP_TRAIL_SL"
                    exit_idx = j
                    break
                if day["Low"] <= current_stop:
                    exit_price, exit_reason = current_stop, "TRAIL_SL"
                    exit_idx = j
                    break
                # ยังไม่ชน SL วันนี้ -> อัปเดต Trailing Stop ขึ้นตามราคาสูงสุดใหม่ (เลื่อนขึ้นทางเดียว)
                peak_price = max(peak_price, day["High"])
                current_stop = max(current_stop, peak_price - trail_distance)

        if exit_price is None:
            exit_idx = max_j - 1
            exit_price = df.iloc[exit_idx]["Close"]
            exit_reason = "TIMEOUT"

        # [BUG FIX] คำนวณ outcome จากราคาขายจริงเทียบกับราคาซื้อเสมอ ไม่ hardcode "LOSS" ตอนชน
        # Trailing Stop เพราะ Trailing Stop ที่เลื่อนขึ้นมาแล้วอาจถูกชนตอนที่ยังเป็นกำไรอยู่ก็ได้
        # (เช่น วิ่งขึ้น +10% ก่อนย่อกลับมาชน Trailing ที่เลื่อนตามมา ก็ยังนับเป็น WIN ไม่ใช่ LOSS)
        outcome = "WIN" if exit_price > entry_price else ("LOSS" if exit_price < entry_price else "BREAKEVEN")

        pnl_pct = (exit_price - entry_price) / entry_price * 100
        days_held = exit_idx - entry_idx + 1

        trades.append({
            "Ticker": ticker,
            "signal_date": df.index[sig_idx].strftime("%Y-%m-%d"),
            "signal_type": signal_row["signal_type"],
            "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 2),
            "initial_stop_loss": round(entry_price - trail_distance, 2),
            "final_trailing_stop": round(current_stop, 2),
            "exit_date": df.index[exit_idx].strftime("%Y-%m-%d"),
            "exit_price": round(exit_price, 2),
            "exit_reason": exit_reason,
            "days_held": days_held,
            "outcome": outcome,
            "pnl_pct": round(pnl_pct, 2),
        })

        next_available_idx = exit_idx + 1

    return trades


def backtest_historical_signals_hybrid(df_ind: pd.DataFrame, ticker: str) -> list:
    """
    กลยุทธ์ผสม (Partial Profit + Trailing Stop) — เอาข้อดีของทั้งสองแบบมารวมกัน:
    - ขาย PARTIAL_TP_SHARE_PCT ของโพซิชัน (ค่าเริ่มต้น 50%) ทันทีที่ราคาวิ่งไปถึง
      PARTIAL_TP_R_MULTIPLE เท่าของระยะเสี่ยง (ค่าเริ่มต้น 1R) — ล็อกกำไรบางส่วนไว้ก่อน
      ป้องกันปัญหา "เงินอยู่ตรงหน้าแต่หลุดมือ" ที่เจอตอนใช้ Trailing Stop ล้วนๆ
    - เมื่อขายไม้แรกแล้ว เลื่อน SL ของไม้ที่เหลือมาที่ราคาเข้าซื้อทันที (Breakeven, ถ้าเปิด
      MOVE_STOP_TO_BREAKEVEN_AFTER_PARTIAL) แล้วปล่อยไม้ที่เหลือวิ่งต่อด้วย Trailing Stop
      (ไม่มี TP ตายตัวสำหรับไม้ที่สอง — ให้กำไรวิ่งได้เต็มที่)
    - ถ้าราคายังไม่ถึง Partial TP แล้วโดน Stop Loss ก่อน -> เสียทั้งโพซิชันเต็มจำนวน (ยังไม่ได้ขายไม้แรก)
    - ถือได้นานสุด MAX_HOLD_DAYS วันเหมือนกลยุทธ์อื่นๆ
    pnl_pct ที่คืนค่าคือ "Blended PnL" ถ่วงน้ำหนักตามสัดส่วนที่ขายจริงของแต่ละไม้
    """
    df = _vectorized_signal_mask(df_ind)
    signal_indices = np.where(df["signal"].values)[0]

    partial_r = CONFIG["PARTIAL_TP_R_MULTIPLE"]
    partial_share = CONFIG["PARTIAL_TP_SHARE_PCT"]
    move_to_be = CONFIG["MOVE_STOP_TO_BREAKEVEN_AFTER_PARTIAL"]
    trail_mult = CONFIG["TRAILING_STOP_ATR_MULTIPLIER"]

    trades = []
    next_available_idx = 0

    for sig_idx in signal_indices:
        if sig_idx < next_available_idx:
            continue

        entry_idx = sig_idx + 1
        if entry_idx >= len(df):
            break

        signal_row = df.iloc[sig_idx]
        entry_open = df.iloc[entry_idx]["Open"]
        if pd.isna(entry_open) or entry_open <= 0:
            next_available_idx = entry_idx + 1
            continue

        entry_price = entry_open * (1 + CONFIG["SLIPPAGE_PCT"])

        atr = signal_row["ATR14"]
        if pd.isna(atr) or atr <= 0:
            next_available_idx = entry_idx + 1
            continue

        stop_distance = CONFIG["ATR_STOPLOSS_MULTIPLIER"] * atr
        initial_stop = entry_price - stop_distance
        partial_tp_price = entry_price + stop_distance * partial_r
        trail_distance = trail_mult * atr

        current_stop = initial_stop
        peak_price = entry_price
        partial_taken = False
        partial_exit_price = None

        final_exit_price, final_exit_reason, exit_idx = None, None, None
        max_j = min(entry_idx + CONFIG["MAX_HOLD_DAYS"], len(df))

        for j in range(entry_idx, max_j):
            day = df.iloc[j]
            is_entry_day = (j == entry_idx)

            if not partial_taken:
                # --- ยังไม่ขายไม้แรก: เช็ค Stop Loss เต็มจำนวนก่อนเสมอ (SL มาก่อน TP ถ้าชนวันเดียวกัน) ---
                if not is_entry_day and day["Open"] <= current_stop:
                    final_exit_price, final_exit_reason = day["Open"], "GAP_SL_FULL"
                    exit_idx = j
                    break
                if day["Low"] <= current_stop:
                    final_exit_price, final_exit_reason = current_stop, "SL_FULL"
                    exit_idx = j
                    break
                if day["High"] >= partial_tp_price:
                    # ขายไม้แรกที่ Partial TP วันนี้ แล้วเริ่มบริหารไม้ที่สองด้วย Trailing ต่อ
                    partial_taken = True
                    partial_exit_price = partial_tp_price
                    if move_to_be:
                        current_stop = max(current_stop, entry_price)
                    peak_price = max(peak_price, day["High"])
                    current_stop = max(current_stop, peak_price - trail_distance)
                    continue  # ยังไม่ปิดเทรด (ไม้สองยังถืออยู่) ไปเช็ควันถัดไปต่อ
                # ยังไม่ชนอะไรเลยวันนี้ -> รอวันถัดไป (ไม้แรกยังไม่ถึงจุดขาย)
            else:
                # --- ขายไม้แรกไปแล้ว: จัดการไม้ที่สองด้วย Trailing Stop (Gap-aware เหมือนเดิม) ---
                if day["Open"] <= current_stop:
                    final_exit_price, final_exit_reason = day["Open"], "GAP_TRAIL_SL"
                    exit_idx = j
                    break
                if day["Low"] <= current_stop:
                    final_exit_price, final_exit_reason = current_stop, "TRAIL_SL"
                    exit_idx = j
                    break
                peak_price = max(peak_price, day["High"])
                current_stop = max(current_stop, peak_price - trail_distance)

        if final_exit_price is None:
            exit_idx = max_j - 1
            final_exit_price = df.iloc[exit_idx]["Close"]
            final_exit_reason = "TIMEOUT"

        # --- คำนวณ Blended PnL ---
        if not partial_taken:
            # ไม่เคยขายไม้แรกเลย -> ทั้งโพซิชัน exit ที่ราคาเดียวกันหมด (SL_FULL หรือ TIMEOUT)
            blended_pnl_pct = (final_exit_price - entry_price) / entry_price * 100
            exit_reason_label = final_exit_reason
        else:
            tranche1_pnl_pct = (partial_exit_price - entry_price) / entry_price * 100
            tranche2_pnl_pct = (final_exit_price - entry_price) / entry_price * 100
            blended_pnl_pct = (tranche1_pnl_pct * partial_share) + (tranche2_pnl_pct * (1 - partial_share))
            exit_reason_label = f"PARTIAL_TP+{final_exit_reason}"

        days_held = exit_idx - entry_idx + 1
        outcome = "WIN" if blended_pnl_pct > 0 else ("LOSS" if blended_pnl_pct < 0 else "BREAKEVEN")

        trades.append({
            "Ticker": ticker,
            "signal_date": df.index[sig_idx].strftime("%Y-%m-%d"),
            "signal_type": signal_row["signal_type"],
            "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 2),
            "partial_tp_price": round(partial_tp_price, 2),
            "partial_taken": partial_taken,
            "exit_date": df.index[exit_idx].strftime("%Y-%m-%d"),
            "final_exit_price": round(final_exit_price, 2),
            "exit_reason": exit_reason_label,
            "days_held": days_held,
            "outcome": outcome,
            "pnl_pct": round(blended_pnl_pct, 2),
        })

        next_available_idx = exit_idx + 1

    return trades


def run_historical_backtest_hybrid():
    """
    รัน Historical Backtest แบบ Hybrid (Partial Profit ที่ 1R + Trailing Stop สำหรับไม้ที่เหลือ)
    เทียบกับเวอร์ชัน TP ตายตัวล้วน และ Trailing Stop ล้วน
    """
    print("=" * 78)
    print("📈 HISTORICAL BACKTEST ENGINE (HYBRID) — Partial Profit + Trailing Stop")
    print(f"ย้อนหลัง {CONFIG['HISTORICAL_BACKTEST_YEARS']} ปี | "
          f"Slippage {CONFIG['SLIPPAGE_PCT']*100:.2f}% | "
          f"Partial TP ที่ {CONFIG['PARTIAL_TP_R_MULTIPLE']}R ({CONFIG['PARTIAL_TP_SHARE_PCT']*100:.0f}% ของไม้) | "
          f"Trail = {CONFIG['TRAILING_STOP_ATR_MULTIPLIER']}x ATR | "
          f"ถือสูงสุด {CONFIG['MAX_HOLD_DAYS']} วันเทรด")
    print("=" * 78)

    tickers = get_sp500_tickers()
    max_tickers = CONFIG["HISTORICAL_BACKTEST_MAX_TICKERS"]
    if max_tickers:
        tickers = tickers[:max_tickers]
        print(f"[INFO] จำกัดทดสอบแค่ {max_tickers} ตัวแรก (ปรับได้ที่ HISTORICAL_BACKTEST_MAX_TICKERS)")

    download_period = f"{CONFIG['HISTORICAL_BACKTEST_YEARS'] + 1}y"
    price_data = download_price_data(tickers, download_period, CONFIG["BATCH_DOWNLOAD_SIZE"])

    all_trades = []
    print(f"[INFO] กำลังจำลองเทรดย้อนหลัง (Hybrid) ของหุ้น {len(price_data)} ตัว ...")
    for ticker, df in price_data.items():
        try:
            df_ind = compute_indicators(df)
            trades = backtest_historical_signals_hybrid(df_ind, ticker)
            all_trades.extend(trades)
        except Exception:
            continue

    if not all_trades:
        print("\n[RESULT] ไม่พบสัญญาณซื้อเลยในประวัติศาสตร์ช่วงที่ทดสอบ")
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)

    filename = f"historical_backtest_hybrid_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    trades_df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] บันทึกเทรดทั้งหมด ({len(trades_df)} เทรด) ลงไฟล์: {filename}")

    print_historical_backtest_summary_hybrid(trades_df)
    return trades_df


def print_historical_backtest_summary_hybrid(trades_df: pd.DataFrame):
    """สรุปสถิติผลการทดสอบย้อนหลังแบบ Hybrid เทียบกับอีกสองเวอร์ชัน"""
    n_total = len(trades_df)
    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] < 0]
    breakeven = trades_df[trades_df["pnl_pct"] == 0]

    win_rate = len(wins) / n_total * 100 if n_total else 0
    avg_win_pct = wins["pnl_pct"].mean() if not wins.empty else 0
    avg_loss_pct = losses["pnl_pct"].mean() if not losses.empty else 0
    expectancy = (win_rate / 100 * avg_win_pct) + ((1 - win_rate / 100) * avg_loss_pct)

    gross_profit = wins["pnl_pct"].sum()
    gross_loss = abs(losses["pnl_pct"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    n_partial_taken = trades_df["partial_taken"].sum()
    n_never_reached_partial = n_total - n_partial_taken

    print("\n" + "=" * 78)
    print("📊 สรุปผล HISTORICAL BACKTEST (HYBRID: PARTIAL PROFIT + TRAILING)")
    print("=" * 78)
    print(f"จำนวนเทรดทั้งหมด: {n_total:,} เทรด (จาก {trades_df['Ticker'].nunique()} หุ้น)")
    print(f"Win Rate: {win_rate:.1f}%  ({len(wins)} ชนะ / {len(losses)} แพ้ / {len(breakeven)} เสมอทุน)")
    print(f"กำไรเฉลี่ยต่อเทรดที่ชนะ: {avg_win_pct:+.2f}%")
    print(f"ขาดทุนเฉลี่ยต่อเทรดที่แพ้: {avg_loss_pct:+.2f}%")
    print(f"Expectancy (คาดหวังกำไร/ขาดทุนเฉลี่ยต่อเทรด): {expectancy:+.3f}%")
    print(f"Profit Factor (กำไรรวม/ขาดทุนรวม): {profit_factor:.3f}"
          f"  {'✅ ดี (>1.5)' if profit_factor > 1.5 else '⚠️ ควรระวัง (<1.5)' if profit_factor >= 1 else '🔴 ขาดทุนสุทธิ'}")
    print()
    print(f"เทรดที่ไปถึง Partial TP ({CONFIG['PARTIAL_TP_R_MULTIPLE']}R) และขายไม้แรกสำเร็จ: "
          f"{n_partial_taken:,} เทรด ({n_partial_taken/n_total*100:.1f}%)")
    print(f"เทรดที่โดน Stop Loss ก่อนถึง Partial TP เลย (เสียเต็มไม้): "
          f"{n_never_reached_partial:,} เทรด ({n_never_reached_partial/n_total*100:.1f}%)")

    if "signal_type" in trades_df.columns:
        print("\nWin Rate แยกตามประเภทสัญญาณ:")
        for sig_type, label in [("both", "สัญญาณคู่ (MACD+EMA20)"), ("macd", "MACD Cross เดี่ยว"), ("ema20", "EMA20 Bounce เดี่ยว")]:
            subset = trades_df[trades_df["signal_type"] == sig_type]
            if len(subset) > 0:
                wr = (subset["pnl_pct"] > 0).mean() * 100
                print(f"  - {label}: {wr:.1f}% Win Rate ({len(subset)} เทรด)")

    print("\n💡 เทียบผลทั้ง 3 กลยุทธ์: รัน `backtest`, `backtest-trailing`, `backtest-hybrid` "
          "แล้วดู Expectancy/Profit Factor ควบคู่กัน")
    print("⚠️  หมายเหตุ: ผลทดสอบย้อนหลัง (Backtest) ไม่ได้การันตีผลในอนาคต ตลาดเปลี่ยนแปลงตลอดเวลา")
    print("=" * 78)


def run_historical_backtest_trailing():
    """
    รัน Historical Backtest แบบ Trailing Stop (ไม่มี TP ตายตัว) เทียบกับเวอร์ชันปกติ
    ใช้ข้อมูล/เงื่อนไขสัญญาณเดียวกันทุกอย่าง ต่างกันแค่วิธีออกจากเทรด
    """
    print("=" * 78)
    print("📈 HISTORICAL BACKTEST ENGINE (TRAILING STOP) — ทดสอบกลยุทธ์ Trailing Stop")
    print(f"ย้อนหลัง {CONFIG['HISTORICAL_BACKTEST_YEARS']} ปี | "
          f"Slippage {CONFIG['SLIPPAGE_PCT']*100:.2f}% | "
          f"Trail = {CONFIG['TRAILING_STOP_ATR_MULTIPLIER']}x ATR | "
          f"ถือสูงสุด {CONFIG['MAX_HOLD_DAYS']} วันเทรด")
    print("=" * 78)

    tickers = get_sp500_tickers()
    max_tickers = CONFIG["HISTORICAL_BACKTEST_MAX_TICKERS"]
    if max_tickers:
        tickers = tickers[:max_tickers]
        print(f"[INFO] จำกัดทดสอบแค่ {max_tickers} ตัวแรก (ปรับได้ที่ HISTORICAL_BACKTEST_MAX_TICKERS)")

    download_period = f"{CONFIG['HISTORICAL_BACKTEST_YEARS'] + 1}y"
    price_data = download_price_data(tickers, download_period, CONFIG["BATCH_DOWNLOAD_SIZE"])

    all_trades = []
    print(f"[INFO] กำลังจำลองเทรดย้อนหลัง (Trailing Stop) ของหุ้น {len(price_data)} ตัว ...")
    for ticker, df in price_data.items():
        try:
            df_ind = compute_indicators(df)
            trades = backtest_historical_signals_trailing(df_ind, ticker)
            all_trades.extend(trades)
        except Exception:
            continue

    if not all_trades:
        print("\n[RESULT] ไม่พบสัญญาณซื้อเลยในประวัติศาสตร์ช่วงที่ทดสอบ")
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)

    filename = f"historical_backtest_trailing_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    trades_df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] บันทึกเทรดทั้งหมด ({len(trades_df)} เทรด) ลงไฟล์: {filename}")

    print_historical_backtest_summary_trailing(trades_df)
    return trades_df


def print_historical_backtest_summary_trailing(trades_df: pd.DataFrame):
    """สรุปสถิติผลการทดสอบย้อนหลังแบบ Trailing Stop เทียบกับเวอร์ชัน TP ตายตัว"""
    n_total = len(trades_df)
    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] < 0]
    breakeven = trades_df[trades_df["pnl_pct"] == 0]

    win_rate = len(wins) / n_total * 100 if n_total else 0
    avg_win_pct = wins["pnl_pct"].mean() if not wins.empty else 0
    avg_loss_pct = losses["pnl_pct"].mean() if not losses.empty else 0
    expectancy = (win_rate / 100 * avg_win_pct) + ((1 - win_rate / 100) * avg_loss_pct)

    gross_profit = wins["pnl_pct"].sum()
    gross_loss = abs(losses["pnl_pct"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    n_gap_trail_sl = (trades_df["exit_reason"] == "GAP_TRAIL_SL").sum()
    n_trail_sl = (trades_df["exit_reason"] == "TRAIL_SL").sum()
    n_timeout = (trades_df["exit_reason"] == "TIMEOUT").sum()

    print("\n" + "=" * 78)
    print("📊 สรุปผล HISTORICAL BACKTEST (TRAILING STOP)")
    print("=" * 78)
    print(f"จำนวนเทรดทั้งหมด: {n_total:,} เทรด (จาก {trades_df['Ticker'].nunique()} หุ้น)")
    print(f"Win Rate: {win_rate:.1f}%  ({len(wins)} ชนะ / {len(losses)} แพ้ / {len(breakeven)} เสมอทุน)")
    print(f"กำไรเฉลี่ยต่อเทรดที่ชนะ: {avg_win_pct:+.2f}%")
    print(f"ขาดทุนเฉลี่ยต่อเทรดที่แพ้: {avg_loss_pct:+.2f}%")
    print(f"Expectancy (คาดหวังกำไร/ขาดทุนเฉลี่ยต่อเทรด): {expectancy:+.2f}%")
    print(f"Profit Factor (กำไรรวม/ขาดทุนรวม): {profit_factor:.2f}"
          f"  {'✅ ดี (>1.5)' if profit_factor > 1.5 else '⚠️ ควรระวัง (<1.5)' if profit_factor >= 1 else '🔴 ขาดทุนสุทธิ'}")
    print()
    print("สาเหตุการออกจากเทรด:")
    print(f"  - ชน Trailing Stop ปกติ:        {n_trail_sl:,} เทรด")
    print(f"  - ชน Trailing Stop แบบ Gap Down: {n_gap_trail_sl:,} เทรด "
          f"({n_gap_trail_sl/n_total*100:.1f}% ของทั้งหมด)")
    print(f"  - หมดเวลาถือ (Timeout):          {n_timeout:,} เทรด")

    if "signal_type" in trades_df.columns:
        print("\nWin Rate แยกตามประเภทสัญญาณ:")
        for sig_type, label in [("both", "สัญญาณคู่ (MACD+EMA20)"), ("macd", "MACD Cross เดี่ยว"), ("ema20", "EMA20 Bounce เดี่ยว")]:
            subset = trades_df[trades_df["signal_type"] == sig_type]
            if len(subset) > 0:
                wr = (subset["pnl_pct"] > 0).mean() * 100
                print(f"  - {label}: {wr:.1f}% Win Rate ({len(subset)} เทรด)")

    print("\n💡 เทียบกับเวอร์ชัน TP ตายตัว: รัน `python sp500_swing_scanner.py backtest` "
          "แล้วเทียบ Expectancy/Profit Factor สองแบบดูว่าแบบไหนดีกว่าจริงในข้อมูลของคุณ")
    print("⚠️  หมายเหตุ: ผลทดสอบย้อนหลัง (Backtest) ไม่ได้การันตีผลในอนาคต ตลาดเปลี่ยนแปลงตลอดเวลา")
    print("=" * 78)


def run_historical_backtest():
    """
    รัน Historical Backtest เต็มรูปแบบ: ดึงข้อมูลย้อนหลัง N ปีของหุ้นทั้งหมดใน S&P 500
    (หรือจำนวนที่จำกัดไว้ใน CONFIG) แล้วจำลองการเทรดทุกครั้งที่เคยเกิดสัญญาณจริงในอดีต
    สรุปสถิติ Win Rate / Expectancy / Profit Factor ที่มีนัยสำคัญทางสถิติ
    """
    print("=" * 78)
    print("📈 HISTORICAL BACKTEST ENGINE — ทดสอบย้อนหลังเต็มรูปแบบ")
    print(f"ย้อนหลัง {CONFIG['HISTORICAL_BACKTEST_YEARS']} ปี | "
          f"Slippage {CONFIG['SLIPPAGE_PCT']*100:.2f}% | "
          f"ถือสูงสุด {CONFIG['MAX_HOLD_DAYS']} วันเทรด")
    print("=" * 78)

    tickers = get_sp500_tickers()
    max_tickers = CONFIG["HISTORICAL_BACKTEST_MAX_TICKERS"]
    if max_tickers:
        tickers = tickers[:max_tickers]
        print(f"[INFO] จำกัดทดสอบแค่ {max_tickers} ตัวแรก (ปรับได้ที่ HISTORICAL_BACKTEST_MAX_TICKERS)")

    download_period = f"{CONFIG['HISTORICAL_BACKTEST_YEARS'] + 1}y"
    price_data = download_price_data(tickers, download_period, CONFIG["BATCH_DOWNLOAD_SIZE"])

    all_trades = []
    print(f"[INFO] กำลังจำลองเทรดย้อนหลังของหุ้น {len(price_data)} ตัว ...")
    for ticker, df in price_data.items():
        try:
            df_ind = compute_indicators(df)
            trades = backtest_historical_signals(df_ind, ticker)
            all_trades.extend(trades)
        except Exception:
            continue

    if not all_trades:
        print("\n[RESULT] ไม่พบสัญญาณซื้อเลยในประวัติศาสตร์ช่วงที่ทดสอบ")
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)

    filename = f"historical_backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    trades_df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] บันทึกเทรดทั้งหมด ({len(trades_df)} เทรด) ลงไฟล์: {filename}")

    print_historical_backtest_summary(trades_df)
    return trades_df


def print_historical_backtest_summary(trades_df: pd.DataFrame):
    """สรุปสถิติผลการทดสอบย้อนหลัง: Win Rate, Avg Win/Loss, Expectancy, Profit Factor"""
    n_total = len(trades_df)
    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] < 0]
    breakeven = trades_df[trades_df["pnl_pct"] == 0]

    win_rate = len(wins) / n_total * 100 if n_total else 0
    avg_win_pct = wins["pnl_pct"].mean() if not wins.empty else 0
    avg_loss_pct = losses["pnl_pct"].mean() if not losses.empty else 0
    expectancy = (win_rate / 100 * avg_win_pct) + ((1 - win_rate / 100) * avg_loss_pct)

    gross_profit = wins["pnl_pct"].sum()
    gross_loss = abs(losses["pnl_pct"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    n_gap_sl = (trades_df["exit_reason"] == "GAP_SL").sum()
    n_normal_sl = (trades_df["exit_reason"] == "SL").sum()
    n_tp = (trades_df["exit_reason"] == "TP").sum()
    n_timeout = (trades_df["exit_reason"] == "TIMEOUT").sum()

    print("\n" + "=" * 78)
    print("📊 สรุปผล HISTORICAL BACKTEST")
    print("=" * 78)
    print(f"จำนวนเทรดทั้งหมด: {n_total:,} เทรด (จาก {trades_df['Ticker'].nunique()} หุ้น)")
    print(f"Win Rate: {win_rate:.1f}%  ({len(wins)} ชนะ / {len(losses)} แพ้ / {len(breakeven)} เสมอทุน)")
    print(f"กำไรเฉลี่ยต่อเทรดที่ชนะ: {avg_win_pct:+.2f}%")
    print(f"ขาดทุนเฉลี่ยต่อเทรดที่แพ้: {avg_loss_pct:+.2f}%")
    print(f"Expectancy (คาดหวังกำไร/ขาดทุนเฉลี่ยต่อเทรด): {expectancy:+.2f}%")
    print(f"Profit Factor (กำไรรวม/ขาดทุนรวม): {profit_factor:.2f}"
          f"  {'✅ ดี (>1.5)' if profit_factor > 1.5 else '⚠️ ควรระวัง (<1.5)' if profit_factor >= 1 else '🔴 ขาดทุนสุทธิ'}")
    print()
    print("สาเหตุการออกจากเทรด:")
    print(f"  - ชน Take Profit ปกติ:        {n_tp:,} เทรด")
    print(f"  - ชน Stop Loss ปกติ:          {n_normal_sl:,} เทรด")
    print(f"  - ชน Stop Loss แบบ Gap Down:  {n_gap_sl:,} เทรด "
          f"({n_gap_sl/n_total*100:.1f}% ของทั้งหมด — นี่คือ 'ความเจ็บปวดจริง' ที่ Backtest แบบเดิมมองไม่เห็น)")
    print(f"  - หมดเวลาถือ (Timeout):        {n_timeout:,} เทรด")

    if "signal_type" in trades_df.columns:
        print("\nWin Rate แยกตามประเภทสัญญาณ:")
        for sig_type, label in [("both", "สัญญาณคู่ (MACD+EMA20)"), ("macd", "MACD Cross เดี่ยว"), ("ema20", "EMA20 Bounce เดี่ยว")]:
            subset = trades_df[trades_df["signal_type"] == sig_type]
            if len(subset) > 0:
                wr = (subset["pnl_pct"] > 0).mean() * 100
                print(f"  - {label}: {wr:.1f}% Win Rate ({len(subset)} เทรด)")

    print("\n⚠️  หมายเหตุ: ผลทดสอบย้อนหลัง (Backtest) ไม่ได้การันตีผลในอนาคต ตลาดเปลี่ยนแปลงตลอดเวลา")
    print("=" * 78)


# ==============================================================================
# 6g. Signal Quality Analyzer — ดึงสถิติ Win Rate จริงจาก Backtest มาช่วยจัดอันดับ/วิเคราะห์
# ==============================================================================

def load_latest_signal_type_stats():
    """
    ค้นหาไฟล์ผลการทดสอบย้อนหลัง (Historical Backtest) ที่ถูกสร้างล่าสุดในโฟลเดอร์เดียวกัน
    ไม่ว่าจะเป็นแบบ TP ตายตัว (`backtest`), Trailing Stop (`backtest-trailing`), หรือ
    Hybrid (`backtest-hybrid`) แล้วคำนวณสถิติ Win Rate / Profit Factor / กำไรเฉลี่ย
    แยกตามประเภทสัญญาณ (both/macd/ema20) เพื่อใช้เป็นข้อมูลจริงประกอบการจัดอันดับ Top N
    และเขียนบทวิเคราะห์ประกอบเหตุผล

    คืนค่า dict {"stats": {...}, "source_file": "..."} หรือ None ถ้ายังไม่เคยรัน backtest มาก่อนเลย
    """
    patterns = [
        "historical_backtest_trades_*.csv",
        "historical_backtest_trailing_trades_*.csv",
        "historical_backtest_hybrid_trades_*.csv",
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))

    if not candidates:
        return None

    latest_file = max(candidates, key=os.path.getmtime)

    try:
        df = pd.read_csv(latest_file)
    except Exception:
        return None

    if "signal_type" not in df.columns or "pnl_pct" not in df.columns:
        return None

    stats = {}
    for sig_type in ["both", "macd", "ema20"]:
        subset = df[df["signal_type"] == sig_type]
        if subset.empty:
            continue
        wins = subset[subset["pnl_pct"] > 0]
        losses = subset[subset["pnl_pct"] < 0]
        win_rate = len(wins) / len(subset) * 100
        gross_profit = wins["pnl_pct"].sum()
        gross_loss = abs(losses["pnl_pct"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        stats[sig_type] = {
            "win_rate": win_rate,
            "avg_pnl": subset["pnl_pct"].mean(),
            "profit_factor": profit_factor,
            "n_trades": len(subset),
        }

    if not stats:
        return None

    return {"stats": stats, "source_file": latest_file}


def print_investment_analysis(top_df: pd.DataFrame, signal_stats_result):
    """
    เขียนบทวิเคราะห์เจาะลึกของหุ้น Top N โดยอิงจากสถิติ Win Rate จริงจาก Historical Backtest
    (ถ้ามี) บอกชัดเจนว่าตัวไหนน่าเข้าเทรดที่สุดเป็นอันดับแรก เพราะอะไร
    """
    print("\n" + "=" * 78)
    print("🔍 บทวิเคราะห์: หุ้นตัวไหนใน Top N น่าเข้าเทรดที่สุด และเพราะอะไร")
    print("=" * 78)

    if signal_stats_result is None:
        print("[INFO] ยังไม่พบไฟล์ผล Historical Backtest ในโฟลเดอร์นี้ (backtest / backtest-trailing /")
        print("       backtest-hybrid) จึงยังไม่มีสถิติ Win Rate จริงมาช่วยวิเคราะห์เชิงลึก")
        print("       การจัดอันดับด้านบนตอนนี้อิงตามสภาพคล่อง (Volume) เพียงอย่างเดียว")
        print("       แนะนำให้รัน `python sp500_swing_scanner.py backtest` ก่อน แล้วสแกนใหม่อีกครั้ง")
        print("       จะได้บทวิเคราะห์ที่มีข้อมูลสถิติจริงประกอบเหตุผลครบถ้วน")
        print("=" * 78)
        return

    stats = signal_stats_result["stats"]
    source_file = signal_stats_result["source_file"]
    label_map = {"both": "สัญญาณคู่ (MACD+EMA20)", "macd": "สัญญาณเดี่ยว (MACD Cross)", "ema20": "สัญญาณเดี่ยว (EMA20 Bounce)"}

    print(f"[อ้างอิงจาก: {source_file}]\n")

    scored_rows = []
    for _, row in top_df.iterrows():
        sig_type = row.get("_signal_type")
        s = stats.get(sig_type)
        scored_rows.append((row, sig_type, s))

    for rank, (row, sig_type, s) in enumerate(scored_rows, start=1):
        ticker = row["Ticker"]
        label = label_map.get(sig_type, row["ระดับสัญญาณ"])
        print(f"[{rank}] {ticker}")
        if s is not None:
            quality = ("แข็งแรงกว่าค่าเฉลี่ยตลาด" if s["win_rate"] >= 50 and s["profit_factor"] >= 1.15
                        else "อยู่ในเกณฑ์ปานกลาง" if s["profit_factor"] >= 1.0
                        else "อ่อนแอกว่าค่าเฉลี่ย ควรระวังเป็นพิเศษ")
            print(f"    ประเภทสัญญาณ: {label}")
            print(f"    สถิติในอดีต: Win Rate {s['win_rate']:.1f}% | Profit Factor {s['profit_factor']:.2f} "
                  f"| กำไรเฉลี่ย {s['avg_pnl']:+.2f}% ต่อเทรด (จากตัวอย่าง {s['n_trades']:,} เทรด)")
            print(f"    การประเมิน: {quality}")
        else:
            print(f"    ประเภทสัญญาณ: {label} (ไม่มีข้อมูลสถิติเปรียบเทียบ)")
        print(f"    สภาพคล่อง (Volume เฉลี่ย 30 วัน): {row['_avg_vol30']:,.0f} หุ้น/วัน")
        print()

    # --- สรุปว่าตัวไหนน่าเข้าเทรดที่สุด ---
    ranked_with_score = []
    for row, sig_type, s in scored_rows:
        wr = s["win_rate"] if s else 50.0
        pf = s["profit_factor"] if s else 1.0
        ranked_with_score.append((row["Ticker"], sig_type, wr, pf))

    best = max(ranked_with_score, key=lambda x: (x[2], x[3]))
    worst = min(ranked_with_score, key=lambda x: (x[2], x[3]))

    print("-" * 78)
    print(f"🏆 สรุป: **{best[0]}** น่าเข้าเทรดเป็นอันดับแรกในกลุ่มนี้")
    if best[1] and stats.get(best[1]):
        bs = stats[best[1]]
        print(f"   เพราะประเภทสัญญาณ ({label_map.get(best[1], best[1])}) มีสถิติ Win Rate "
              f"{bs['win_rate']:.1f}% และ Profit Factor {bs['profit_factor']:.2f} "
              f"สูงสุดในกลุ่ม Top {len(ranked_with_score)} ตัวนี้")
    else:
        print("   (ไม่มีข้อมูลสถิติเปรียบเทียบสำหรับตัวนี้ ใช้สภาพคล่องเป็นเกณฑ์หลักแทน)")

    if worst[0] != best[0] and worst[1] and stats.get(worst[1]):
        ws = stats[worst[1]]
        if ws["profit_factor"] < 1.0 or ws["win_rate"] < 48:
            print(f"\n⚠️  ระวังเป็นพิเศษ: **{worst[0]}** แม้จะติด Top N (มักเพราะสภาพคล่องสูง) แต่ประเภทสัญญาณ "
                  f"({label_map.get(worst[1], worst[1])}) มีสถิติในอดีตค่อนข้างอ่อนแอ "
                  f"(Win Rate {ws['win_rate']:.1f}%, Profit Factor {ws['profit_factor']:.2f}) "
                  f"พิจารณาลดขนาดการเข้าซื้อ หรือข้ามตัวนี้ไปเลยก็ได้")

    print("\n💡 หมายเหตุ: สถิตินี้คำนวณจากผล Backtest ย้อนหลังเท่านั้น ไม่ได้การันตีผลในอนาคต")
    print("   ควรใช้ประกอบการตัดสินใจร่วมกับปัจจัยอื่น (ข่าว, ภาวะตลาดรวม, ความเสี่ยงเฉพาะตัวหุ้น)")
    print("=" * 78)


# ==============================================================================
# 7. Main Scanner
# ==============================================================================

def run_scanner():
    capital_usd = CONFIG["CAPITAL_THB"] / CONFIG["USD_THB_RATE"]
    print("=" * 78)
    print(f"S&P 500 SHORT-TERM SWING SCANNER")
    print(f"ทุนตั้งต้น: {CONFIG['CAPITAL_THB']:,.0f} บาท  ≈  ${capital_usd:,.2f} USD "
          f"(เรท {CONFIG['USD_THB_RATE']} บาท/USD)")
    print(f"เวลาสแกน: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    # --- [SANITY CHECK] เตือนถ้ารันระหว่างตลาดสหรัฐฯ ยังเปิดอยู่ ---
    market_hours_warning = check_market_hours_warning()
    if market_hours_warning:
        print(f"\n{market_hours_warning}\n")

    # --- [AUTO BACKTEST] เช็คผลจริงของ Top Picks รอบก่อนหน้า ก่อนเริ่มสแกนรอบใหม่ ---
    if CONFIG["ENABLE_AUTO_BACKTEST"]:
        prev_file = find_previous_top_picks_file()
        if prev_file:
            print(f"[INFO] พบผลสแกนรอบก่อนหน้า: {prev_file} — กำลังเช็คผลจริง (Backtest) ...")
            backtest_df = backtest_previous_picks(prev_file)
            if not backtest_df.empty:
                save_backtest_to_history(backtest_df, CONFIG["BACKTEST_HISTORY_FILE"])
                print_backtest_summary(backtest_df, CONFIG["BACKTEST_HISTORY_FILE"])
        else:
            print("[INFO] ยังไม่มีผลสแกนรอบก่อนหน้าให้ backtest (นี่อาจเป็นการรันครั้งแรก)")

    # --- [ACTIVE TRADES TRACKER] ติดตามหุ้นเก่าที่ยังถืออยู่ (OPEN) ทั้งหมดจากประวัติสะสม ---
    # ทำหลังบล็อก AUTO BACKTEST ด้านบน (ซึ่งจะ append แถว OPEN ของรอบล่าสุดเข้าไฟล์ก่อน)
    # และทำก่อนเริ่มพิมพ์ตารางสแกนหุ้นตัวใหม่ของวันนี้เสมอ ตามที่ต้องการ
    if CONFIG["ENABLE_ACTIVE_TRADE_TRACKER"]:
        track_active_portfolio_trades(CONFIG["BACKTEST_HISTORY_FILE"])

    # --- [MARKET REGIME] เช็คเทรนด์ SPY ก่อนสแกนหุ้นรายตัว ---
    regime_note = ""
    if CONFIG["ENABLE_MARKET_REGIME_FILTER"]:
        print("[INFO] กำลังเช็คเทรนด์ภาพรวมตลาด (SPY) ...")
        regime = check_market_regime()
        if regime["is_bullish"] is True:
            print(f"[INFO] ✅ Market Regime: SPY เป็นขาขึ้น — {regime['description']}")
        elif regime["is_bullish"] is False:
            print(f"[WARN] ⚠️  Market Regime: SPY เป็นขาลง — {regime['description']}")
            regime_note = "⚠️ ตลาดรวม (SPY) เป็นขาลง — สัญญาณนี้อาจเป็นแค่การเด้งเพื่อลงต่อ (Bull Trap) โปรดระวังเป็นพิเศษ"
            if CONFIG["STRICT_MARKET_REGIME_FILTER"]:
                print("\n[RESULT] ตั้งค่า STRICT_MARKET_REGIME_FILTER=True ไว้ "
                      "และตลาดรวมเป็นขาลง จึงงดแสดงผลสัญญาณซื้อรอบนี้ทั้งหมด")
                print("=" * 78)
                return pd.DataFrame()
        else:
            print(f"[WARN] {regime['description']}")

    tickers = get_sp500_tickers()
    price_data = download_price_data(
        tickers, CONFIG["HISTORY_PERIOD"], CONFIG["BATCH_DOWNLOAD_SIZE"]
    )

    results = []
    for ticker, df in price_data.items():
        try:
            df_ind = compute_indicators(df)
            signal = check_signal(df_ind)
            if not signal["pass"]:
                continue

            position = calculate_position(signal, capital_usd)
            if position is None:
                continue

            reason_text = signal["reason"]
            if regime_note:
                reason_text += f" | {regime_note}"

            results.append({
                "Ticker": ticker,
                "ราคาซื้อแนะนำ ($)": position["entry"],
                "ราคาเป้าหมาย ($)": position["take_profit"],
                "จุดขาย Stoploss ($)": position["stop_loss"],
                "จำนวนซื้อ ไม้แรก (50%)": position["shares_first_tranche"],
                # [ROBUSTNESS FIX] คอลัมน์ภาษาอังกฤษล้วนสำหรับให้โค้ดส่วนอื่น (เช่น
                # backtest_previous_picks / track_active_portfolio_trades) อ่านค่าจำนวนหุ้น
                # ได้แบบชัวร์ๆ โดยไม่ต้อง match ชื่อคอลัมน์ภาษาไทยตรงเป๊ะ ป้องกันโปรแกรมพัง
                # หากมีการแก้ label ภาษาไทยด้านบนในอนาคต
                "shares": position["shares_first_tranche"],
                "มูลค่าไม้แรก ($)": position["position_value_usd"],
                "ความเสี่ยง/เทรด ($)": position["risk_amount_usd"],
                # --- Trailing Stop แนะนำ สำหรับตั้งใน Webull (ใช้แทน TP ตายตัวได้) ---
                "Trailing Stop ($ Trail)": position["trailing_stop_amount"],
                "Trailing Stop (% Trail)": position["trailing_stop_pct"],
                "ระดับสัญญาณ": signal["signal_label"],
                "เหตุผลทำไมต้องซื้อ": reason_text,
                # --- คอลัมน์ช่วยจัดอันดับ (จะถูกลบออกก่อนแสดงผล) ---
                "_signal_strength": signal["signal_strength"],
                "_signal_type": signal["signal_type"],
                "_avg_vol30": signal["avg_vol30"],
            })
        except Exception as e:
            # ข้ามหุ้นที่มีปัญหาข้อมูล ไม่ให้สคริปต์ล้มทั้งหมด
            continue

    if not results:
        print("\n[RESULT] ไม่พบหุ้นที่ผ่านเงื่อนไขทั้งหมดในการสแกนรอบนี้")
        return pd.DataFrame()

    full_df = pd.DataFrame(results)

    # --- [EARNINGS BLACKOUT] ตัดหุ้นที่ใกล้ประกาศงบออกจากผลลัพธ์เลย (ก่อนจัดอันดับ Top N) ---
    # ป้องกันเคสแบบ Citigroup (C) ที่ราคาเปิดกระโดดชน Stop Loss เพราะประกาศงบกะทันหัน
    earnings_days_cache = {}
    if CONFIG["CHECK_EARNINGS_RISK"] and CONFIG["ENABLE_EARNINGS_BLACKOUT"] and not full_df.empty:
        print(f"[INFO] กำลังเช็ค Earnings Blackout (ตัดออกถ้าประกาศงบภายใน "
              f"{CONFIG['EARNINGS_BLACKOUT_DAYS']} วัน) ของหุ้นที่ผ่านเกณฑ์ทั้งหมด ({len(full_df)} ตัว) ...")
        keep_mask = []
        excluded = []
        for _, row in full_df.iterrows():
            ticker = row["Ticker"]
            days_until = get_days_until_earnings(ticker)
            earnings_days_cache[ticker] = days_until
            is_blackout = days_until is not None and days_until <= CONFIG["EARNINGS_BLACKOUT_DAYS"]
            keep_mask.append(not is_blackout)
            if is_blackout:
                excluded.append((ticker, days_until))

        full_df = full_df[keep_mask].reset_index(drop=True)

        if excluded:
            print(f"[INFO] 🚫 ตัดออก {len(excluded)} ตัว เพราะใกล้ประกาศงบเกินไป (Earnings Blackout):")
            for tk, d in excluded:
                day_label = "วันนี้" if d == 0 else f"อีก {d} วัน"
                print(f"       - {tk}: ประกาศงบ{day_label}")

    if full_df.empty:
        print("\n[RESULT] ไม่พบหุ้นที่ผ่านเงื่อนไขทั้งหมดในการสแกนรอบนี้ "
              "(อาจถูกตัดออกหมดเพราะ Earnings Blackout)")
        return pd.DataFrame()

    # --- โหลดสถิติ Win Rate จริงจากผล Historical Backtest ล่าสุด (ถ้ามี) มาช่วยจัดอันดับ ---
    signal_stats_result = load_latest_signal_type_stats()

    # --- จัดอันดับ: ใช้สถิติ Win Rate/Profit Factor จริงจาก Backtest เป็นเกณฑ์หลัก (ถ้ามีข้อมูล) ---
    # [ประวัติการแก้ไข] เดิมเคยให้น้ำหนักพิเศษกับ "สัญญาณคู่" โดยสมมติเอาเองว่าน่าจะแม่นกว่า
    # แต่ผลทดสอบย้อนหลังจริงพิสูจน์ว่าสัญญาณคู่แย่ที่สุดเสมอ (ยืนยันซ้ำ 3 รอบทั้ง TP ตายตัว,
    # Trailing Stop, Hybrid) จึงเปลี่ยนมาใช้ "สถิติจริง" จากไฟล์ backtest ล่าสุดเป็นเกณฑ์หลักแทน
    # การเดาเอาเอง ถ้ายังไม่เคยรัน backtest มาก่อนเลย (signal_stats_result is None) จะ fallback
    # เป็นสภาพคล่องอย่างเดียวเหมือนเดิม (คะแนนเท่ากันหมด ทำให้ _avg_vol30 เป็นตัวตัดสินจริงๆ)
    if signal_stats_result is not None:
        stats_lookup = signal_stats_result["stats"]
        full_df["_win_rate_score"] = full_df["_signal_type"].map(
            lambda st: stats_lookup.get(st, {}).get("win_rate", 50.0)
        )
        full_df["_profit_factor_score"] = full_df["_signal_type"].map(
            lambda st: stats_lookup.get(st, {}).get("profit_factor", 1.0)
        )
    else:
        full_df["_win_rate_score"] = 50.0
        full_df["_profit_factor_score"] = 1.0

    full_df = full_df.sort_values(
        by=["_win_rate_score", "_profit_factor_score", "_avg_vol30"], ascending=[False, False, False]
    ).reset_index(drop=True)
    full_df.insert(0, "อันดับ", range(1, len(full_df) + 1))

    # บันทึกผลลัพธ์ "ทั้งหมด" ที่ผ่านเงื่อนไข ไว้เป็นไฟล์สำรอง (เผื่ออยากดูตัวเลือกอื่นนอกเหนือ Top N)
    display_cols = [c for c in full_df.columns if not c.startswith("_")]
    full_filename = f"scan_result_full_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    full_df[display_cols].to_csv(full_filename, index=False, encoding="utf-8-sig")

    # --- ตัดเหลือ Top N ตัวที่น่าสนใจที่สุด ---
    top_n = CONFIG["TOP_N_RESULTS"]
    top_full = full_df.head(top_n).reset_index(drop=True)  # เก็บคอลัมน์ internal ไว้ให้บทวิเคราะห์ใช้
    top_df = top_full[display_cols]  # ตัวที่แสดงผล/บันทึกไฟล์จริง (ไม่มีคอลัมน์ internal)

    # --- [EARNINGS WARNING] สำหรับ Top N: เตือน (ไม่ตัดออก) ถ้าใกล้ประกาศงบในช่วงถัดจาก Blackout ---
    if CONFIG["CHECK_EARNINGS_RISK"] and not top_df.empty:
        reason_col = "เหตุผลทำไมต้องซื้อ"
        blackout_days = CONFIG["EARNINGS_BLACKOUT_DAYS"] if CONFIG["ENABLE_EARNINGS_BLACKOUT"] else -1
        for idx, row in top_df.iterrows():
            ticker = row["Ticker"]
            # ใช้ค่าที่เช็คไว้แล้วจากขั้นตอน Blackout ถ้ามี ไม่ต้องยิง API ซ้ำ
            days_until = earnings_days_cache.get(ticker, "MISSING")
            if days_until == "MISSING":
                days_until = get_days_until_earnings(ticker)
            if days_until is not None and blackout_days < days_until <= CONFIG["EARNINGS_WARNING_DAYS"]:
                warning = f"⚠️ ใกล้ประกาศงบใน {days_until} วัน เสี่ยง Gap ทะลุ Stoploss โปรดพิจารณาความเสี่ยงเพิ่มเติม"
                top_df.at[idx, reason_col] = f"{row[reason_col]} | {warning}"

    print(f"\n[RESULT] พบหุ้นที่ผ่านเงื่อนไขทั้งหมด {len(full_df)} ตัว "
          f"→ คัดเหลือ Top {min(top_n, len(full_df))} ตัวที่น่าสนใจที่สุด\n")

    _print_readable_table(top_df)

    # บันทึกผลลัพธ์ Top N เป็นไฟล์ CSV หลัก
    filename = f"scan_result_top{top_n}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    top_df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] บันทึกผลลัพธ์ Top {top_n} ลงไฟล์: {filename}")
    print(f"[INFO] บันทึกผลลัพธ์ทั้งหมด ({len(full_df)} ตัว) ลงไฟล์สำรอง: {full_filename}")

    print("\n" + "=" * 78)
    if signal_stats_result is not None:
        print(f"เกณฑ์การจัดอันดับ: 1) Win Rate จริงจาก Backtest ล่าสุด "
              f"({signal_stats_result['source_file']}) มาก่อน")
        print("                  2) Profit Factor สูงกว่า เป็นตัวตัดสินรอง")
        print("                  3) สภาพคล่อง (วอลุ่มเฉลี่ย 30 วัน) สูงกว่า เป็นตัวตัดสินสุดท้าย")
    else:
        print("เกณฑ์การจัดอันดับ: สภาพคล่อง (วอลุ่มเฉลี่ย 30 วัน) เพียงอย่างเดียว")
        print("(ยังไม่พบไฟล์ Historical Backtest ในโฟลเดอร์นี้ — รัน `python sp500_swing_scanner.py")
        print(" backtest` ก่อน จะได้จัดอันดับด้วยสถิติ Win Rate จริงในการสแกนครั้งถัดไป)")
    print("คำเตือน: ผลลัพธ์นี้เป็นการกรองตามเงื่อนไขทางเทคนิคเท่านั้น ไม่ใช่คำแนะนำการลงทุน")
    print("โปรดตรวจสอบข้อมูล ณ เวลาจริง (Pre-Market/Post-Market) ก่อนตัดสินใจเทรดทุกครั้ง")
    print("=" * 78)

    print_investment_analysis(top_full, signal_stats_result)

    return top_df


def _print_readable_table(df: pd.DataFrame):
    """
    แสดง DataFrame เป็นตารางที่อ่านง่าย โดยแยกคอลัมน์ 'เหตุผลทำไมต้องซื้อ'
    (ซึ่งเป็นข้อความยาว) ออกมาแสดงแยกใต้ตารางหลัก เพื่อไม่ให้ตารางล้นจอ
    """
    reason_col = "เหตุผลทำไมต้องซื้อ"
    hidden_cols = {reason_col, "shares"}  # "shares" คือคอลัมน์ช่วยสำหรับโค้ด ไม่ต้องโชว์ซ้ำกับ "จำนวนซื้อ ไม้แรก (50%)"
    main_cols = [c for c in df.columns if c not in hidden_cols]

    try:
        from tabulate import tabulate
        print(tabulate(df[main_cols], headers="keys", tablefmt="grid", showindex=False))
    except ImportError:
        print(df[main_cols].to_string(index=False))
        print("\n[TIP] ติดตั้ง `pip install tabulate` แล้วรันใหม่ จะได้ตารางที่มีเส้นกรอบสวยขึ้น")

    print("\n" + "-" * 78)
    print("เหตุผลประกอบการซื้อ (แยกตามอันดับ):")
    print("-" * 78)
    for _, row in df.iterrows():
        print(f"\n[{row['อันดับ']}] {row['Ticker']} — {row['ระดับสัญญาณ']}")
        print(f"    {row[reason_col]}")

    if CONFIG.get("ENABLE_TRAILING_STOP_SUGGESTION"):
        _print_webull_trailing_stop_guide(df)


def _print_webull_trailing_stop_guide(df: pd.DataFrame):
    """
    แสดงคำแนะนำวิธีตั้งคำสั่ง Trailing Stop ใน Webull สำหรับหุ้นแต่ละตัวในตาราง
    (ทางเลือกแทนการตั้ง Take Profit ตายตัว — ปล่อยให้กำไรวิ่งได้ไกลขึ้นตามผล Backtest)
    """
    print("\n" + "=" * 78)
    print("📱 วิธีตั้งคำสั่งใน Webull — Trailing Stop Order (ทางเลือกแทน TP ตายตัว)")
    print("=" * 78)
    print("ขั้นตอน: เปิดหน้าหุ้น > กด Trade > เลือกแท็บ Order > ประเภทคำสั่งเลือก")
    print("'Trailing Stop' (หรือ 'Trailing Stop Limit' ถ้าอยากจำกัดราคาขายขั้นต่ำด้วย)")
    print("แล้วเลือกโหมด Trail By: 'Amount' (ใส่เป็น $) หรือ 'Percentage' (ใส่เป็น %) อย่างใดอย่างหนึ่ง")
    print("-" * 78)

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        entry = row["ราคาซื้อแนะนำ ($)"]
        trail_amt = row["Trailing Stop ($ Trail)"]
        trail_pct = row["Trailing Stop (% Trail)"]
        initial_sl = row["จุดขาย Stoploss ($)"]
        print(f"\n[{row['อันดับ']}] {ticker}")
        print(f"    1) ตั้ง SL ป้องกันทุนไว้ก่อนที่ ${initial_sl:.2f} (Stop-Loss Order ปกติ แบบ GTC)")
        print(f"       — กันเงินจมถ้าราคาร่วงแรงตั้งแต่ต้น ก่อนที่ Trailing จะเริ่มขยับตาม")
        print(f"    2) เมื่อราคาเริ่มวิ่งเป็นบวก ให้ตั้ง Trailing Stop Order เพิ่ม (หรือแก้จาก SL เดิม):")
        print(f"       - Trail By Amount: ${trail_amt:.2f}")
        print(f"       - หรือ Trail By Percentage: {trail_pct:.2f}%")
        print(f"       (สองแบบนี้คือค่าเดียวกัน แค่คนละหน่วย เลือกใส่แบบใดแบบหนึ่งพอ)")
        print(f"    3) Time in Force ตั้งเป็น GTC (Trailing Stop ไม่รองรับ GTC ค้างข้ามคืนในบางโบรกเกอร์")
        print(f"       ต้องเช็ค/ตั้งใหม่ทุกเช้าถ้า Webull ไม่รองรับ GTC สำหรับคำสั่งนี้)")

    print("\n⚠️  หมายเหตุสำคัญ:")
    print("   - Trailing Stop เป็น Stop Order: เมื่อราคาแตะจุดที่เลื่อนมา จะกลายเป็น Market Order ทันที")
    print("     อาจได้ราคาแย่กว่าจุดที่ตั้งไว้เล็กน้อยถ้าตลาดผันผวนแรง (เหมือน Stop-Loss ปกติ)")
    print("   - ตัวเลขนี้คำนวณจาก ATR ณ วันที่สแกน ถ้าถือข้ามหลายวัน ควรพิจารณาคำนวณ ATR ใหม่")
    print("     เป็นระยะ เพราะความผันผวนของหุ้นอาจเปลี่ยนไปจากวันที่เข้าซื้อ")
    print("   - ยังไม่ได้ทดสอบ Trailing Stop นี้ด้วย Historical Backtest จริง (เป็นแค่คำแนะนำ")
    print("     จากค่า ATR เดียวกับ SL เดิม) แนะนำให้รอผลทดสอบ backtest-trailing ก่อนใช้เงินจริง")
    print("=" * 78)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("backtest", "--backtest", "-b"):
        run_historical_backtest()
    elif len(sys.argv) > 1 and sys.argv[1].lower() in ("backtest-trailing", "--backtest-trailing", "-bt"):
        run_historical_backtest_trailing()
    elif len(sys.argv) > 1 and sys.argv[1].lower() in ("backtest-hybrid", "--backtest-hybrid", "-bh"):
        run_historical_backtest_hybrid()
    else:
        run_scanner()
