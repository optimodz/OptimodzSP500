import streamlit as st
import pandas as pd
from datetime import datetime

# ดึงฟังก์ชันสแกนและเช็คตลาดจากสคริปต์หลักของคุณ
from sp500_swing_scanner import run_scanner, CONFIG, check_market_regime

# --- 1. ตั้งค่าหน้าตาของ Web App (UX/UI Page Config) ---
st.set_page_config(
    page_title="S&P 500 Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ส่วนหัวของหน้าเว็บ (Header Area) ---
st.title("📊 S&P 500 Short-Term Swing Scanner")
st.caption(f"เวลาปัจจุบัน (BKK): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")

# --- 3. โซนควบคุมและแสดงสถานะตลาด (Control & Market Status Panel) ---
col_action, col_spy, col_cap = st.columns([1, 2, 1.5])

with col_action:
    # ปุ่มกดสแกนสดๆ Real-time (Call to Action)
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
    # สรุปเงินทุนสำหรับคำนวณขนาดไม้เทรด
    capital_usd = CONFIG["CAPITAL_THB"] / CONFIG["USD_THB_RATE"]
    st.info(f"💰 **ทุนรวม:** {CONFIG['CAPITAL_THB']:,} THB (≈ ${capital_usd:,.2f} USD)")

st.markdown("---")

# --- 4. โซนผลลัพธ์การสแกนหุ้นใหม่ของวันนี้ (Scan Results Area) ---
st.subheader("🏆 หุ้น Top Picks ที่ผ่านเงื่อนไขวันนี้")

if run_button:
    # แสดงตัวโหลดอนิเมชันให้ผู้ใช้รู้ว่าระบบกำลังดึงข้อมูลอยู่ (UX Feedback)
    with st.spinner("กำลังดาวน์โหลดและวิเคราะห์ข้อมูลหุ้น S&P 500 ทั้งหมดจาก Yahoo Finance..."):
        
        # รันฟังก์ชันสแกนหลัก (ตัวนี้จะแอบสร้าง CSV บน Cloud ชั่วคราวช่างมันครับ เราดึงตัวแปรมาใช้โชว์หน้าจอพอ)
        top_df = run_scanner()
        
        if top_df is not None and not top_df.empty:
            st.success(f"สแกนเสร็จสิ้น! พบหุ้นที่น่าสนใจที่สุด {len(top_df)} อันดับแรก")
            
            # ซ่อนคอลัมน์คำอธิบายยาวๆ ออกไปก่อน เพื่อให้ตารางข้อมูลตัวเลขดูง่ายบนจอมือถือ
            reason_col = "เหตุผลทำไมต้องซื้อ"
            main_table_cols = [c for c in top_df.columns if c != reason_col]
            
            # แสดงตารางผลลัพธ์หลักบนหน้าเว็บ
            st.dataframe(top_df[main_table_cols], use_container_width=True, hide_index=True)
            
            # --- ปรับ UI แยกบทวิเคราะห์และคำอธิบายออกมาไว้ด้านล่างตารางหลัก ---
            st.markdown("### 🔍 รายละเอียดและเหตุผลประกอบการตัดสินใจ")
            for _, row in top_df.iterrows():
                rank_val = row.get("อันดับ", "-")
                ticker_val = row.get("Ticker", "Unknown")
                signal_val = row.get("ระดับสัญญาณ", "Unknown")
                
                # ใช้ระบบการ์ด (Expander) พับเก็บข้อมูลยาวๆ ช่วยลดความล้าของสายตาเวลาดูในมือถือ
                with st.expander(f"อันดับ {rank_val} | **{ticker_val}** ({signal_val})"):
                    st.write(f"**💡 เหตุผลทางเทคนิค:** {row.get(reason_col, 'ไม่มีข้อมูลเหตุผล')}")
                    st.write(f"**📱 คำแนะนำตั้งคำสั่ง Webull (Trailing Stop):** Trail ราคาเป็นจำนวนที่ **${row.get('Trailing Stop ($ Trail)', 0.0)}** หรือคิดเป็น **{row.get('Trailing Stop (% Trail)', 0.0)}%**")
        else:
            st.warning("ไม่พบหุ้นที่ผ่านเกณฑ์ทางเทคนิคทั้งหมดในรอบการสแกนนี้")
else:
    st.info("💡 *หยิบมือถือขึ้นมากดปุ่ม 'เริ่มสแกนหุ้นวันนี้' ด้านบน เพื่อดูหน้าจอ Dashboard สรุปผลได้ทันที*")
