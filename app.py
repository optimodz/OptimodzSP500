import streamlit as st
import pandas as pd
import os
from datetime import datetime

# 1. ดึงฟังก์ชันและการตั้งค่าจากสคริปต์เดิมของคุณมาใช้งาน
from sp500_swing_scanner import run_scanner, track_active_portfolio_trades, CONFIG, check_market_regime

# --- 2. ตั้งค่าหน้าตาของ Web App (UX/UI Page Config) ---
st.set_page_config(
    page_title="S&P 500 Swing Scanner",
    page_icon="📊",
    layout="wide", # ปรับหน้าเว็บให้กว้างเต็มจอเพื่อให้ตารางไม่อึดอัด
    initial_sidebar_state="collapsed"
)

# --- 3. ส่วนหัวของหน้าเว็บ (Header Area) ---
st.title("📊 S&P 500 Short-Term Swing Scanner")
st.caption(f"อัปเดตข้อมูลล่าสุด: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")

# --- 4. โซนควบคุมและแสดงสถานะตลาด (Control & Market Status Panel) ---
col_action, col_spy, col_cap = st.columns([1, 2, 1.5])

with col_action:
    # ปุ่มกดขนาดใหญ่สีเด่นชัดสำหรับเริ่มรันระบบสแกนสดๆ (Call to Action Button)
    run_button = st.button("🚀 เริ่มสแกนหุ้นวันนี้", use_container_width=True, type="primary")

with col_spy:
    # เช็คเทรนด์ภาพรวมตลาดแม่ (SPY) มาแสดงผลทันทีแบบเข้าใจง่าย
    regime = check_market_regime()
    if regime["is_bullish"] is True:
        st.success(f"🟩 **Market Regime:** SPY ขาขึ้น ({regime['description']})")
    elif regime["is_bullish"] is False:
        st.error(f"🟥 **Market Regime:** SPY ขาลง ({regime['description']})")
    else:
        st.warning(f"🟨 **Status:** {regime['description']}")

with col_cap:
    # การคำนวณเงินทุนสรุปให้เห็นแบบชัดๆ ไม่ต้องคำนวณในใจ
    capital_usd = CONFIG["CAPITAL_THB"] / CONFIG["USD_THB_RATE"]
    st.info(f"💰 **ทุนรวม:** {CONFIG['CAPITAL_THB']:,} THB (≈ ${capital_usd:,.2f} USD)")

st.markdown("---")

# --- 5. โซนที่ 1: ตารางติดตามหุ้นเดิมที่ยังถืออยู่ (Active Trades Tracker) ---
st.subheader("📂 พอร์ตโฟลิโอ: หุ้นที่ยังถืออยู่ (Active Trades)")

history_file = CONFIG["BACKTEST_HISTORY_FILE"]
if os.path.exists(history_file):
    # รันระบบอัปเดตราคาและสถานะหุ้นเก่าอัตโนมัติทุกครั้งที่เปิดหน้าเว็บ
    updated_active_df = track_active_portfolio_trades(history_file)
    
    if not updated_active_df.empty:
        # ดึงเฉพาะรายการที่สถานะยังเปิดอยู่ (OPEN) มาแสดงผลบน Dashboard
        open_trades = updated_active_df[updated_active_df["outcome"] == "OPEN"]
        
        if not open_trades.empty:
            # คัดเลือกเฉพาะคอลัมน์สำคัญเพื่อให้ตารางกระชับและสะอาดตาที่สุด
            display_cols = ["Ticker", "entry_date", "entry_price", "take_profit", "stop_loss", "current_or_exit_price", "days_held", "pnl_pct", "pnl_usd"]
            st.dataframe(open_trades[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("ไม่มีออเดอร์ค้างอยู่ในพอร์ตโฟลิโอขณะนี้")
    else:
        st.info("ไม่มีออเดอร์ค้างอยู่ในพอร์ตโฟลิโอขณะนี้")
else:
    st.info("ยังไม่มีประวัติการซื้อขายในระบบ (ไฟล์ประวัติสะสมยังไม่ถูกสร้าง)")

st.markdown("---")

# --- 6. โซนที่ 2: ผลลัพธ์การสแกนหุ้นใหม่ของวันนี้ (Scan Results Area) ---
st.subheader("🏆 หุ้น Top Picks ที่ผ่านเงื่อนไขวันนี้")

# เมื่อผู้ใช้กดปุ่มสแกน
if run_button:
    # แสดงตัวโหลดอนิเมชันเพื่อบ่งบอกให้ผู้ใช้ทราบว่าระบบกำลังทำงานอยู่จริง (UX Feedback)
    with st.spinner("กำลังดาวน์โหลดและวิเคราะห์ข้อมูลหุ้น S&P 500 ทั้งหมดจาก Yahoo Finance..."):
        top_df = run_scanner()
        
        if not top_df.empty:
            st.success(f"สแกนเสร็จสิ้น! พบหุ้นที่น่าสนใจที่สุด {len(top_df)} อันดับแรก")
            
            # ซ่อนคอลัมน์คำอธิบายยาวๆ ออกไปก่อน เพื่อให้ตารางข้อมูลตัวเลขดูง่าย ไม่รกตา
            reason_col = "เหตุผลทำไมต้องซื้อ"
            main_table_cols = [c for c in top_df.columns if c != reason_col]
            
            # แสดงตารางผลลัพธ์หลัก (ผู้ใช้สามารถคลิกหัวตารางเพื่อจัดเรียงข้อมูลได้อิสระ)
            st.dataframe(top_df[main_table_cols], use_container_width=True, hide_index=True)
            
            # --- ปรับ UI แยกบทวิเคราะห์และคำอธิบายออกมาไว้ด้านล่างตารางหลัก ---
            st.markdown("### 🔍 รายละเอียดและเหตุผลประกอบการตัดสินใจ")
            for _, row in top_df.iterrows():
                # ใช้ระบบการ์ด (Expander) เพื่อซ่อน/ขยายข้อมูล ช่วยลดความหนาแน่นของตัวอักษรบนหน้าจอ
                with st.expander(f"อันดับ {row['อันดับ']} | **{row['Ticker']}** ({row['ระดับสัญญาณ']})"):
                    st.write(f"**💡 เหตุผลทางเทคนิค:** {row[reason_col]}")
                    st.write(f"**📱 คำแนะนำตั้งคำสั่ง Webull (Trailing Stop):** Trail ราคาเป็นจำนวนที่ **${row['Trailing Stop ($ Trail)']}** หรือคิดเป็น **{row['Trailing Stop (% Trail)']}%**")
        else:
            st.warning("ไม่พบหุ้นที่ผ่านเกณฑ์ทางเทคนิคทั้งหมดในรอบการสแกนนี้")
else:
    st.write("💡 *กดปุ่ม 'เริ่มสแกนหุ้นวันนี้' ด้านบน เพื่อดึงข้อมูลสัญญาณใหม่ล่าสุดแบบ Real-time*")