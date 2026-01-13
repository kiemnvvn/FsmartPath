import streamlit as st
from streamlit_gsheets import GSheetsConnection
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import time

# --- 1. CẤU HÌNH HỆ THỐNG & GIAO DIỆN ---
st.set_page_config(page_title="F-SmartPath Tin học 12", layout="wide", page_icon="🎓")

st.markdown("""
    <style>
        .block-container { padding-top: 2.5rem !important; }
        .fsmart-header { 
            background-color: #F8F9FA; padding: 15px 20px; border-radius: 12px; 
            border-left: 10px solid #FF4B4B; margin-bottom: 25px; 
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        }
        .topic-list { line-height: 1.1; margin-bottom: 0px; padding-bottom: 2px; font-size: 0.95rem; }
        .ai-feedback { 
            background-color: #E8F0FE; padding: 15px; border-radius: 10px; 
            border-left: 5px solid #1A73E8; margin-top: 10px; margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

API_KEY = "AIzaSyAAUyXZ_zc8Ja2DP2kQrovU1CZq0DjI-30"
genai.configure(api_key=API_KEY)

def get_available_model():
    try:
        priority_models = ['gemini-1.5-flash', 'gemini-pro']
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for target in priority_models:
            full_name = f"models/{target}"
            if full_name in models: return full_name
        return models[0] if models else "models/gemini-pro"
    except: return "models/gemini-pro"

if 'quiz_data' not in st.session_state:
    st.session_state.update({
        'quiz_data': [], 'current_idx': 0, 'score': 0, 
        'history': {}, 'chat_history': [], 'is_started': False,
        'model_name': get_available_model(), 'is_finished': False, 'ai_comment': ""
    })

# --- 2. KẾT NỐI DỮ LIỆU ---
@st.cache_data(ttl=60)
def load_data(url):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(spreadsheet=url)
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except: return pd.DataFrame()

SHEET_URL = "https://docs.google.com/spreadsheets/d/1VRjMKfEFRieebvA6WGsvBSvwHbjttTsh6he74CRPO4M/edit?usp=sharing"
all_df = load_data(SHEET_URL)

# --- 3. HÀM AI ĐƯA RA LỜI KHUYÊN ---
def get_ai_feedback(score, history_list):
    wrong_topics = list(set([h['topic'] for h in history_list if not h['is_correct']]))
    prompt = f"""
    Bạn là giáo viên dạy Tin học 12. Học sinh vừa hoàn thành bài tập với kết quả {score}/100. 
    Các chủ đề học sinh làm sai: {', '.join(wrong_topics) if wrong_topics else 'Không có (Làm đúng hết)'}.
    Hãy đưa ra lời nhận xét ngắn gọn (tối đa 3 câu), xúc tích, mang tính động viên và định hướng. 
    Xưng hô Thầy - Em.
    """
    try:
        model = genai.GenerativeModel(st.session_state.model_name)
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Thầy khen em đã hoàn thành bài luyện tập. Hãy rà soát lại các câu sai để nắm vững kiến thức tốt hơn nhé!"

# --- 4. HÀM TẠO FRAME XÁC NHẬN ĐỘC LẬP ---
@st.dialog("Xác nhận kết thúc bài làm")
def show_confirm_dialog():
    st.write("❓ **Bạn có chắc chắn kết thúc bài làm không?**")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("OK", use_container_width=True, type="primary"):
            st.session_state.score = sum(10 for h in st.session_state.history.values() if h['is_correct'])
            st.write("⚙️ **AI đang phân tích bài làm của bạn....**")
            # Gọi AI lấy lời khuyên trước khi chuyển màn hình
            st.session_state.ai_comment = get_ai_feedback(st.session_state.score, list(st.session_state.history.values()))
            st.session_state.is_finished = True
            st.rerun()

# --- SIDEBAR (Giữ nguyên) ---
with st.sidebar:
    st.title("⚙️ Cấu hình")
    if not all_df.empty:
        topics = ["Tất cả"] + sorted(all_df['topic'].dropna().unique().tolist())
        sel_topic = st.selectbox("Chọn nội dung học tập:", topics)
        if st.button("🚀 Bắt đầu luyện đề", type="primary", use_container_width=True):
            f_df = all_df if sel_topic == "Tất cả" else all_df[all_df['topic'] == sel_topic]
            st.session_state.quiz_data = f_df.sample(n=min(10, len(f_df))).to_dict('records')
            st.session_state.current_idx = 0; st.session_state.score = 0; st.session_state.history = {}
            st.session_state.is_started = True; st.session_state.is_finished = False; st.session_state.ai_comment = ""
            st.rerun()
    st.divider()
    st.write(f"📞 Thầy Kiểm: 0905 89 39 59")

# --- GIAO DIỆN CHÍNH ---
col_main, col_chat = st.columns([1.6, 1], gap="large")

with col_main:
    st.markdown('<div class="fsmart-header"><h1><span style="color:#FF4B4B;">F</span>SmartPath</h1><p>Hệ thống luyện tập Tin học 12 thông minh</p></div>', unsafe_allow_html=True)
    
    if not st.session_state.is_started:
        st.info("👋 Thầy Kiểm chào bạn! Hãy chọn chủ đề bên trái để bắt đầu.")
    
    elif not st.session_state.is_finished:
        q_idx = st.session_state.current_idx
        q = st.session_state.quiz_data[q_idx]
        st.write(f"**Câu hỏi {q_idx + 1} / {len(st.session_state.quiz_data)}**")
        st.progress((q_idx + 1) / len(st.session_state.quiz_data))
        
        with st.container(border=True):
            st.markdown(f"**{q.get('content')}**")
            opts = [str(q.get(f'option {i}', '')).strip() for i in ['a','b','c','d']]
            opts = [o for o in opts if o and o.lower() != 'nan']
            
            old_ans = st.session_state.history.get(q_idx, {}).get('user_ans', None)
            default_idx = opts.index(old_ans) if old_ans in opts else 0
            ans = st.radio("Chọn câu trả lời:", opts, index=default_idx, key=f"q_{q_idx}")
            
            c_back, c_next = st.columns(2)
            with c_back:
                if st.button("⬅️ Câu trước", use_container_width=True) and q_idx > 0:
                    st.session_state.current_idx -= 1; st.rerun()
            
            with c_next:
                is_last = q_idx == len(st.session_state.quiz_data) - 1
                label = "✅ Kết thúc làm bài" if is_last else "Tiếp theo ➡️"
                if st.button(label, use_container_width=True, type="primary"):
                    st.session_state.history[q_idx] = {
                        "topic": q.get('topic', 'Chung'), "is_correct": str(ans).strip() == str(q.get('answer', '')).strip(),
                        "user_ans": ans, "correct_ans": q.get('answer', ''), "content": q.get('content'), "opts": opts
                    }
                    if is_last: show_confirm_dialog()
                    else: st.session_state.current_idx += 1; st.rerun()
    else:
        # --- MÀN HÌNH KẾT QUẢ ---
        st.success(f"🎊 Hoàn thành! Điểm của bạn: {st.session_state.score}/100")
        
        # HIỂN THỊ LỜI KHUYÊN AI
        st.markdown(f'<div class="ai-feedback"><b>👨‍🏫 Lời khuyên:</b><br>{st.session_state.ai_comment}</div>', unsafe_allow_html=True)

        h_list = list(st.session_state.history.values())
        df_res = pd.DataFrame(h_list)
        ratio = (len(df_res[df_res['is_correct']]) / len(st.session_state.quiz_data)) * 100

        # --- DÒNG 190: THÊM TẢI BÀI LÀM (PDF/TXT) ---
        report_text = f"BÁO CÁO KẾT QUẢ F-SMARTPATH\nĐiểm: {st.session_state.score}/100\nLời khuyên: {st.session_state.ai_comment}\n\nCHI TIẾT BÀI LÀM:\n"
        for i, h in enumerate(h_list):
            report_text += f"Câu {i+1}: {'Đúng' if h['is_correct'] else 'Sai'}\n- Nội dung: {h['content']}\n- Bạn chọn: {h['user_ans']}\n- Đáp án đúng: {h['correct_ans']}\n\n"
        st.download_button(label="📥 Tải về báo cáo kết quả bài làm", data=report_text, file_name=f"KetQua_FSmartPath.txt", mime="text/plain", use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(df_res, names='is_correct', hole=0.4, height=300, color='is_correct', 
                         color_discrete_map={True:'#2ecc71', False:'#e74c3c'}, labels={True:'Đúng', False:'Sai'}), use_container_width=True)
        with c2:
            st.write("💡 **Nhận xét năng lực**")
            if ratio >= 90: st.success(f"🚀 **Năng lực: XUẤT SẮC**")
            elif ratio >= 80: st.success(f"🌟 **Năng lực: GIỎI**")
            elif ratio >= 70: st.info(f"📈 **Năng lực: TRUNG BÌNH KHÁ**")
            else: st.error(f"⚠️ **Năng lực: CHƯA ĐẠT**")

            wrong_topics = df_res[df_res['is_correct'] == False]['topic'].unique().tolist()
            if wrong_topics:
                st.write("---")
                st.warning("📍 **Các chủ đề cần chú trọng:**")
                for topic in wrong_topics:
                    st.markdown(f'<p class="topic-list">- {topic}</p>', unsafe_allow_html=True)

        st.divider()
        st.subheader("🔍 BẢNG CHI TIẾT BÀI LÀM")
        view_mode = st.radio("Lọc hiển thị:", ["Tất cả", "Câu Đúng", "Câu Sai"], horizontal=True)
        for i, h in enumerate(h_list):
            res_str = "Đúng" if h['is_correct'] else "Sai"
            if view_mode == "Tất cả" or res_str == view_mode.replace("Câu ", ""):
                with st.expander(f"Câu {i+1}: {h['content'][:60]}... ({res_str})", expanded=(res_str=="Sai")):
                    st.write(f"**{h['content']}**")
                    for o in h['opts']:
                        if o == h['correct_ans']: st.write(f"✅ **{o}** (Đúng)")
                        elif o == h['user_ans'] and not h['is_correct']: st.write(f"❌ **{o}** (Bạn chọn)")
                        else: st.write(f"⚪ {o}")

        
        if st.button("🔄 Làm bài mới", use_container_width=True):
            st.session_state.is_started = False; st.rerun()


# --- 6. CỘT TRỢ GIẢNG AI (CẬP NHẬT HIỆU ỨNG CHỜ) ---
with col_chat:
    st.markdown('<div class="assistant-header">🤖 AI Mentor</div>', unsafe_allow_html=True)
    chat_box = st.container(height=450, border=True)
    
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
    if prompt := st.chat_input("Hỏi AI..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with chat_box:
            st.chat_message("user").markdown(prompt)
            
            # --- HIỆU ỨNG AI ĐANG SUY NGHĨ ---
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                status_text = "AI đang suy nghĩ"
                
                # Bắt đầu gọi API và chạy hiệu ứng chữ
                try:
                    model = genai.GenerativeModel(st.session_state.model_name)
                    
                    # Sử dụng stream=True để tạo cảm giác phản hồi nhanh hoặc giả lập bằng vòng lặp
                    # Ở đây ta dùng vòng lặp đơn giản để hiện dấu chấm động trong lúc API xử lý
                    with st.spinner(''):
                        # Hiệu ứng lặp ký tự dấu chấm
                        for i in range(3):
                            for dots in [".", "..", "..."]:
                                thinking_placeholder.markdown(f"*{status_text}{dots}*")
                                time.sleep(0.3)
                        
                        # Gọi kết quả thực tế
                        response = model.generate_content(f"Giải thích ngắn gọn cho học sinh 12: {prompt}")
                        thinking_placeholder.markdown(response.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                except:
                    thinking_placeholder.error("Dịch vụ AI đang bận, thầy hãy thử lại nhé!")