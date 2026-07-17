import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz

# ดึงฟังก์ชันสแกนและเช็คตลาดจากสคริปต์หลักของคุณ
from sp500_swing_scanner import run_scanner, CONFIG, check_market_regime, compute_indicators

# --- 1. ตั้งค่าหน้าตาของ Web App (UX/UI Page Config) ---
st.set_page_config(
    page_title="S&P 500 Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ตั้งค่าโซนเวลาประเทศไทย ---
tz_bkk = pytz.timezone('Asia/Bangkok')
now_bkk = datetime.now(tz_bkk)

# --- 3. ส่วนหัวของหน้าเว็บ (Header Area) ---
st.title("📊 S&P500 Scanner By Optimodz")
st.caption(f"เวลาปัจจุบัน (BKK): {now_bkk.strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")

# --- 4. โซนควบคุมและแสดงสถานะตลาด (Control & Market Status Panel) ---
col_action, col_spy, col_cap = st.columns([1, 2, 1.5])

with col_action:
    run_button = st.button("🚀 เริ่มสแกนหุ้นวันนี้", use_container_width=True, type="primary")

with col_spy:
    regime = check_market_regime()
    if regime["is_bullish"] is True:
        st.success(f"🟩 **Market Regime:** SPY ขาขึ้น ({regime['description']})")
    elif regime["is_bullish"] is False:
        st.error(f"🟥 **Market Regime:** SPY ขาลง ({regime['description']})")
    else:
        st.warning(f"🟨 **Status:** {regime['description']}")

with col_cap:
    capital_usd = CONFIG["CAPITAL_THB"] / CONFIG["USD_THB_RATE"]
    st.info(f"💰 **ทุนรวม:** {CONFIG['CAPITAL_THB']:,} THB (≈ ${capital_usd:,.2f} USD)")

st.markdown("---")

# ==============================================================================
# ✨ โซนฟังก์ชันใหม่: ระบบวิเคราะห์หุ้นในพอร์ตรายตัว (UX/UI Manual Portfolio Analyzer)
# ==============================================================================
st.subheader("🔍 ระบบเช็คสถานะหุ้นรายตัว (ยังถือต่อได้ไหม?)")
st.markdown("💡 *พิมพ์ชื่อหุ้นและราคาที่คุณเข้าซื้อจริง เพื่อให้ระบบคำนวณหน้างานว่าควร ถือรอ หรือ ขายออก ทันที*")

# สร้างกล่องกรอกข้อมูล 4 ช่องเรียงกันสวยงาม
col_tk, col_ent, col_tp, col_sl = st.columns(4)

with col_tk:
    input_ticker = st.text_input("📝 ชื่อหุ้น (Ticker เช่น NVDA, UBER):", "").upper().strip()
with col_ent:
    input_entry = st.number_input("💵 ราคาที่คุณเข้าซื้อ ($):", min_value=0.0, value=0.0, step=0.01)
with col_tp:
    input_tp = st.number_input("🎯 ราคาเป้าหมาย TP ($):", min_value=0.0, value=0.0, step=0.01)
with col_sl:
    input_sl = st.number_input("🚨 จุดขายขาดทุน SL ($):", min_value=0.0, value=0.0, step=0.01)

# ปุ่มกดสำหรับเริ่มวิเคราะห์หุ้นตัวนี้
analyze_button = st.button("🧐 วิเคราะห์หุ้นตัวนี้เลย", use_container_width=True)

if analyze_button and input_ticker:
    if input_entry <= 0 or input_tp <= 0 or input_sl <= 0:
        st.warning("⚠️ กรุณากรอก ราคาเข้าซื้อ, ราคาเป้าหมาย และ จุดขายขาดทุน ให้ครบถ้วนและมากกว่า 0")
    else:
        with st.spinner(f"กำลังดึงข้อมูลราคาสดของ {input_ticker} จาก Yahoo Finance..."):
            try:
                # ดึงข้อมูลย้อนหลัง 1 ปี เพื่อคำนวณเทคนิคอลให้แม่นยำ (ตามตรรกะ Active Tracker เดิม)
                data = yf.download(tickers=input_ticker, period=CONFIG["ACTIVE_TRADE_LOOKBACK_PERIOD"], interval="1d", auto_adjust=True, progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data.dropna(subset=["Close"])
                
                if data.empty:
                    st.error(f"❌ ไม่พบข้อมูลราคาของหุ้น {input_ticker} กรุณาเช็คตัวสะกดอีกครั้ง")
                else:
                    last_close = float(data.iloc[-1]["Close"])
                    
                    # คำนวณ Indicators ตามตรรกะโค้ดหลัก
                    df_ind = compute_indicators(data.copy())
                    df_ind = df_ind.dropna(subset=["MACD", "MACD_Signal"])
                    
                    # 1. เช็คตรรกะโมเมนตัม (MACD Cross)[cite: 1]
                    macd_dead_cross = False
                    if not df_ind.empty:
                        last_ind = df_ind.iloc[-1]
                        if last_ind["MACD"] < last_ind["MACD_Signal"]:
                            macd_dead_cross = True # เกิดสัญญาณเตือนโมเมนตัมจบรอบ[cite: 1]
                    
                    # --- สรุปคำแนะนำเชิงกลยุทธ์ตามหลัก UX/UI ---
                    st.markdown(f"### 📊 ผลการวิเคราะห์หุ้น **{input_ticker}**")
                    
                    # คำนวณกำไร/ขาดทุนปัจจุบันแบบ Mark-to-Market
                    current_pnl_pct = ((last_close - input_entry) / input_entry) * 100
                    
                    col_res1, col_res2 = st.columns(2)
                    col_res1.metric("ราคาปิดล่าสุด", f"${last_close:.2f}")
                    col_res2.metric("กำไร/ขาดทุนปัจจุบัน", f"{current_pnl_pct:+.2f}%")
                    
                    # กล่องคำวิเคราะห์ขนาดใหญ่
                    if last_close <= input_sl:
                        st.error(f"🔴 **คำแนะนำ: ขายออกด่วน (ชน Stop Loss)!** ราคาล่าสุด (${last_close:.2f}) ต่ำกว่าจุดตัดขาดทุนที่คุณตั้งไว้ (${input_sl:.2f})")
                    elif last_close >= input_tp:
                        st.success(f"🟢 **คำแนะนำ: ขายทำกำไร (ชนเป้า TP)!** ราคาล่าสุด (${last_close:.2f}) ถึงหรือเลยเป้าหมายทำกำไรของคุณแล้ว (${input_tp:.2f})")
                    elif macd_dead_cross:
                        st.error(f"🚨 **คำแนะนำ: พิจารณาขายออก (Momentum Exit)!** ราคาปัจจุบันยังไม่ชน SL แต่เส้น MACD ได้ตัดหลุดต่ำกว่าเส้น Signal เรียบร้อยแล้วในกราฟรายวัน บ่งบอกว่าแรงส่งขาขึ้นหมดลงแล้ว[cite: 1]")
                    else:
                        st.info(f"🔵 **คำแนะนำ: ถือรอต่อได้ (HOLD)** ราคาหุ้นยังแกว่งตัวอยู่ระหว่างทาง และสัญญาณเทคนิคอลฝั่งโมเมนตัมยังไม่เสียทรง (MACD ยังอยู่เหนือ Signal) สามารถถือรอตามแผนเดิมได้ครับ")
                        
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")

st.markdown("---")

# --- 5. โซนผลลัพธ์การสแกนหุ้นใหม่ของวันนี้ ---
st.subheader("🏆 หุ้น Top Picks ที่ผ่านเงื่อนไขวันนี้")

if run_button:
    with st.spinner("กำลังดาวน์โหลดและวิเคราะห์ข้อมูลหุ้น S&P 500 ทั้งหมดจาก Yahoo Finance..."):
        top_df = run_scanner()
        if top_df is not None and not top_df.empty:
            st.success(f"สแกนเสร็จสิ้น! พบหุ้นที่น่าสนใจที่สุด {len(top_df)} อันดับแรก")
            reason_col = "เหตุผลทำไมต้องซื้อ"
            main_table_cols = [c for c in top_df.columns if c != reason_col]
            
            st.dataframe(top_df[main_table_cols], use_container_width=True, hide_index=True)
            
            st.markdown("### 🔍 รายละเอียดและเหตุผลประกอบการตัดสินใจ")
            for _, row in top_df.iterrows():
                rank_val = row.get("อันดับ", "-")
                ticker_val = row.get("Ticker", "Unknown")
                signal_val = row.get("ระดับสัญญาณ", "Unknown")
                
                with st.expander(f"อันดับ {rank_val} | **{ticker_val}** ({signal_val})"):
                    st.write(f"**💡 เหตุผลทางเทคนิค:** {row.get(reason_col, 'ไม่มีข้อมูลเหตุผล')}")
                    st.write(f"**📱 คำแนะนำตั้งคำสั่ง Webull (Trailing Stop):** Trail ราคาเป็นจำนวนที่ **${row.get('Trailing Stop ($ Trail)', 0.0)}** หรือคิดเป็น **{row.get('Trailing Stop (% Trail)', 0.0)}%**")
        else:
            st.warning("ไม่พบหุ้นที่ผ่านเกณฑ์ทางเทคนิคทั้งหมดในรอบการสแกนนี้")
else:
    st.info("💡 *หยิบมือถือขึ้นมากดปุ่ม 'เริ่มสแกนหุ้นวันนี้' ด้านบน เพื่อดูหน้าจอ Dashboard สรุปผลได้ทันที*")
