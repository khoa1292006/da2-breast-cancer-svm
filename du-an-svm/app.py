import hashlib
import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# 1. Cấu hình giao diện Web Y khoa
st.set_page_config(
    page_title="MedVision AI | Hệ Thống Chẩn Đoán Ung Thư Vú",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS phong cách phòng xét nghiệm / bệnh viện hiện đại
st.markdown("""
<style>
    /* Nền tổng thể phong cách Clinical Dark */
    .stApp {
        background: linear-gradient(180deg, #0b132b 0%, #1c2541 100%);
        color: #e2e8f0;
    }
    
    /* Khối thẻ thông tin lâm sàng */
    .med-card {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    .med-card-danger {
        background: rgba(45, 17, 24, 0.9);
        border-left: 6px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.2);
    }
    
    .med-card-success {
        background: rgba(14, 43, 35, 0.9);
        border-left: 6px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.2);
    }

    /* Huy hiệu y tế */
    .badge-danger {
        background-color: #ef4444;
        color: white;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        letter-spacing: 0.05em;
    }
    
    .badge-success {
        background-color: #10b981;
        color: white;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        letter-spacing: 0.05em;
    }

    /* Tiêu đề y khoa */
    .med-header {
        color: #38bdf8;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
</style>
""", unsafe_allow_html=True)

# 2. Hệ thống quản lý tài khoản bác sĩ bền vững (Persistent Storage)
USER_DB_FILE = "users.json"

def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USER_DB_FILE):
        default_db = {"bacsi": hash_pw("123456")}
        with open(USER_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(default_db, f)
        return default_db
    with open(USER_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_user(username, password):
    users = load_users()
    users[username] = hash_pw(password)
    with open(USER_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

# 3. Màn hình Đăng nhập / Đăng ký
if not st.session_state.authenticated:
    _, col_login, _ = st.columns([1, 1.6, 1])
    with col_login:
        st.markdown("""
        <div style="text-align: center; margin-top: 50px; margin-bottom: 25px;">
            <h1 style="color: #38bdf8; margin-bottom: 0;">🩺 HOSPITAL MEDVISION AI</h1>
            <p style="color: #94a3b8; font-size: 0.95rem;">Hệ Thống Trợ Lý Chẩn Đoán Ung Thư Tuyến Vú FNA</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab_log, tab_reg = st.tabs(["🔐 Bác Sĩ Đăng Nhập", "📝 Cấp Tài Khoản Mới"])
        users_db = load_users()
        
        with tab_log:
            with st.form("form_login"):
                u_in = st.text_input("Mã định danh Bác sĩ", placeholder="bacsi")
                p_in = st.text_input("Mật khẩu truy cập", type="password", placeholder="••••••")
                if st.form_submit_button("Xác thực chứng chỉ lâm sàng", use_container_width=True):
                    if u_in in users_db and users_db[u_in] == hash_pw(p_in):
                        st.session_state.authenticated = True
                        st.session_state.username = u_in
                        st.rerun()
                    else:
                        st.error("Tài khoản hoặc mật khẩu không đúng.")
            st.caption("Tài khoản mặc định: `bacsi` | Mật khẩu: `123456`")
            
        with tab_reg:
            with st.form("form_reg"):
                new_u = st.text_input("Tạo mã Bác sĩ mới")
                new_p = st.text_input("Mật khẩu bảo mật", type="password")
                new_cp = st.text_input("Xác nhận lại mật khẩu", type="password")
                if st.form_submit_button("Lưu hồ sơ vào hệ thống", use_container_width=True):
                    if not new_u or not new_p:
                        st.warning("Vui lòng nhập đầy đủ thông tin.")
                    elif new_u in users_db:
                        st.error("Mã tài khoản này đã tồn tại trong cơ sở dữ liệu.")
                    elif new_p != new_cp:
                        st.error("Mật khẩu xác nhận không khớp.")
                    else:
                        save_user(new_u, new_p)
                        st.success(f"Tài khoản `{new_u}` đã được tạo thành công! Bạn có thể chuyển qua đăng nhập.")
    st.stop()

# 4. Tải Model SVM & Dữ liệu
@st.cache_resource
def load_all_assets():
    payload = joblib.load('svm_breast_cancer_model.joblib')
    dataset = load_breast_cancer(as_frame=True)
    return payload, dataset

try:
    payload, dataset = load_all_assets()
    model = payload['model']
    feature_names = payload['feature_names']
    target_names = payload['target_names']
    df_raw = dataset.data
    y_raw = dataset.target
except Exception as e:
    st.error(f"Lỗi hệ thống: {e}. Vui lòng kiểm tra lại file `svm_breast_cancer_model.joblib`.")
    st.stop()

# Sidebar: Bảng thông tin Bác sĩ phụ trách
with st.sidebar:
    st.markdown(f"""
    <div style="background: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #475569; margin-bottom: 15px;">
        <span style="font-size: 0.8rem; color: #38bdf8; text-transform: uppercase; font-weight: bold;">Bác sĩ phụ trách</span>
        <h3 style="color: #f8fafc; margin: 4px 0 0 0;">BS. {st.session_state.username.upper()}</h3>
        <p style="color: #10b981; font-size: 0.85rem; margin: 0;">● Trực tuyến (Phòng Giải phẫu bệnh)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### ⚙️ Thông số Mô hình Học máy")
    st.markdown("""
    - **Mô hình:** Support Vector Machine (SVM)
    - **Hàm nhân (Kernel):** RBF
    - **Dữ liệu chuẩn:** Wisconsin Diagnostic (WDBC)
    - **Số đặc trưng:** 30 chỉ số tế bào FNA
    """)
    st.divider()
    if st.button("🚪 Đăng xuất khỏi hệ thống", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

# 5. Nội dung làm việc chính
st.markdown("<h2 class='med-header'>🩺 BẢNG HỖ TRỢ RA QUYẾT ĐỊNH LÂM SÀNG (CDSS)</h2>", unsafe_allow_html=True)
st.caption("Ứng dụng phân tích chỉ số sinh thiết tế bào học chọc hút kim nhỏ (FNA) tuyến vú.")

main_tab1, main_tab2 = st.tabs(["🔬 Khảo Sát Bệnh Án Cá Nhân", "📊 Phân Tích Danh Sách Bệnh Án (.CSV)"])

with main_tab1:
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    load_mal = col_ctrl1.button("🚨 Nạp mẫu: Ca ÁC TÍNH (Malignant)", use_container_width=True)
    load_ben = col_ctrl2.button("🟢 Nạp mẫu: Ca LÀNH TÍNH (Benign)", use_container_width=True)
    load_rst = col_ctrl3.button("🔄 Đặt lại giá trị trung bình", use_container_width=True)

    if "current_record" not in st.session_state or load_rst:
        st.session_state.current_record = df_raw.mean().to_dict()
    elif load_mal:
        st.session_state.current_record = df_raw[y_raw == 0].iloc[0].to_dict()
    elif load_ben:
        st.session_state.current_record = df_raw[y_raw == 1].iloc[0].to_dict()

    st.markdown("#### 1. Nhập thông số hình thái nhân tế bào")
    
    mean_cols = [f for f in feature_names if "mean" in f]
    se_cols = [f for f in feature_names if "error" in f]
    worst_cols = [f for f in feature_names if "worst" in f]

    user_values = {}
    with st.expander("📍 Phân nhóm 1: Kích thước & Cấu trúc trung bình (Mean Values)", expanded=True):
        c1, c2, c3 = st.columns(3)
        for i, col in enumerate(mean_cols):
            with [c1, c2, c3][i % 3]:
                user_values[col] = st.number_input(
                    col,
                    value=float(st.session_state.current_record.get(col, 0.0)),
                    format="%.4f"
                )

    with st.expander("📍 Phân nhóm 2: Độ bất đối xứng & Sai số tiêu bản (Standard Error)", expanded=False):
        c1, c2, c3 = st.columns(3)
        for i, col in enumerate(se_cols):
            with [c1, c2, c3][i % 3]:
                user_values[col] = st.number_input(
                    col,
                    value=float(st.session_state.current_record.get(col, 0.0)),
                    format="%.4f"
                )

    with st.expander("📍 Phân nhóm 3: Các giá trị dị dạng tồi tệ nhất (Worst Values)", expanded=False):
        c1, c2, c3 = st.columns(3)
        for i, col in enumerate(worst_cols):
            with [c1, c2, c3][i % 3]:
                user_values[col] = st.number_input(
                    col,
                    value=float(st.session_state.current_record.get(col, 0.0)),
                    format="%.4f"
                )

    st.markdown("<br>", unsafe_allow_html=True)
    run_diagnosis = st.button("🚀 BẮT ĐẦU CHẨN ĐOÁN LÂM SÀNG", type="primary", use_container_width=True)

    if run_diagnosis:
        input_df = pd.DataFrame([user_values])
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        result_label = target_names[prediction]

        prob_malignant = prob[0] * 100
        prob_benign = prob[1] * 100

        st.markdown("### 2. Kết Quả Hội Chẩn Lâm Sàng")

        # THẺ KẾT QUẢ PHÂN TÍCH Y KHOA
        if result_label == 'malignant':
            st.markdown(f"""
            <div class="med-card-danger">
                <span class="badge-danger">MÃ NGUY HIỂM: ÁC TÍNH (MALIGNANT)</span>
                <h2 style="color: #ef4444; margin-top: 10px; margin-bottom: 5px;">Xác suất Khối u Ác tính: {prob_malignant:.2f}%</h2>
                <p style="color: #fca5a5; font-size: 1.05rem;">
                    <b>Nhận định tế bào học:</b> Nhân tế bào biến dạng kích thước lớn, màng nhân gồ ghề bất đối xứng và mật độ concavity cao. 
                    Mô hình SVM nhận diện mẫu tế bào này thuộc nhóm có nguy cơ xâm lấn mô đệm rất cao.
                </p>
                <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; margin-top: 10px;">
                    <b>🩺 Khuyến nghị của Bác sĩ chuyên khoa:</b> Tiến hành sinh thiết lõi kim (Core Needle Biopsy) khẩn cấp và chụp MRI tuyến vú đối chứng.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="med-card-success">
                <span class="badge-success">KẾT QUẢ AN TOÀN: LÀNH TÍNH (BENIGN)</span>
                <h2 style="color: #10b981; margin-top: 10px; margin-bottom: 5px;">Xác suất Khối u Lành tính: {prob_benign:.2f}%</h2>
                <p style="color: #a7f3d0; font-size: 1.05rem;">
                    <b>Nhận định tế bào học:</b> Quần thể tế bào phân bố đồng đều, nhân tròn đều nhẵn, không ghi nhận các điểm phân bào bất thường. 
                    Mô hình SVM xếp mẫu vào vùng an toàn sinh học.
                </p>
                <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; margin-top: 10px;">
                    <b>🩺 Khuyến nghị của Bác sĩ chuyên khoa:</b> U xơ/nang tuyến vú lành tính. Chỉ định theo dõi định kỳ sau 6 tháng.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 3. Mô Hình Trực Quan Hóa AI (AI Explainability)")

        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            st.markdown("""<div class="med-card"><b>📊 Biểu đồ Radar: 6 Đặc trưng Nhân Tế Bào Quan Trọng</b>""", unsafe_allow_html=True)
            radar_cols = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean concavity']
            
            norm_vals = []
            for rc in radar_cols:
                v_min = df_raw[rc].min()
                v_max = df_raw[rc].max()
                scaled = (user_values[rc] - v_min) / (v_max - v_min) * 100
                norm_vals.append(np.clip(scaled, 0, 100))

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_vals,
                theta=radar_cols,
                fill='toself',
                name='Bệnh nhân hiện tại',
                line_color='#ef4444' if result_label == 'malignant' else '#10b981',
                fillcolor='rgba(239, 68, 68, 0.3)' if result_label == 'malignant' else 'rgba(16, 185, 129, 0.3)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color="#94a3b8"),
                    angularaxis=dict(color="#cbd5e1")
                ),
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1'),
                margin=dict(l=40, r=40, t=30, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_c2:
            st.markdown("""<div class="med-card"><b>📍 Không Gian Phân Tách Siêu Phẳng SVM (Decision Space)</b>""", unsafe_allow_html=True)
            fig_scatter, ax = plt.subplots(figsize=(6, 4.3))
            fig_scatter.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#0f172a')

            f1, f2 = 'worst radius', 'worst concave points'
            ax.scatter(df_raw[y_raw == 1][f1], df_raw[y_raw == 1][f2], color='#10b981', alpha=0.35, label='Lành tính (Nền)', s=18)
            ax.scatter(df_raw[y_raw == 0][f1], df_raw[y_raw == 0][f2], color='#ef4444', alpha=0.35, label='Ác tính (Nền)', s=18)

            p_f1 = user_values[f1]
            p_f2 = user_values[f2]
            ax.scatter([p_f1], [p_f2], color='#38bdf8', s=200, marker='*', edgecolor='white', linewidth=1.5, label='CA ĐANG KHÁM')

            ax.set_xlabel(f1, color='#94a3b8')
            ax.set_ylabel(f2, color='#94a3b8')
            ax.tick_params(colors='#64748b')
            for spine in ax.spines.values():
                spine.set_color('#334155')
            ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.2, color='#64748b')

            st.pyplot(fig_scatter)
            st.markdown("</div>", unsafe_allow_html=True)

with main_tab2:
    st.markdown("""<div class="med-card">""", unsafe_allow_html=True)
    st.subheader("Kiểm Tra Bệnh Án Hàng Loạt (Batch Diagnosis)")
    up_csv = st.file_uploader("Tải lên danh sách mẫu mô học (.csv)", type=["csv"])
    if up_csv is not None:
        batch_df = pd.read_csv(up_csv)
        if all(f in batch_df.columns for f in feature_names):
            preds = model.predict(batch_df[feature_names])
            probs = model.predict_proba(batch_df[feature_names])
            
            batch_df["KẾT LUẬN"] = [target_names[p].upper() for p in preds]
            batch_df["Nguy cơ Ác tính (%)"] = (probs[:, 0] * 100).round(2)
            batch_df["Tỷ lệ Lành tính (%)"] = (probs[:, 1] * 100).round(2)
            
            st.dataframe(batch_df, use_container_width=True)
            csv_export = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Tải Báo Cáo Chẩn Đoán Tổng Hợp (.csv)",
                data=csv_export,
                file_name="Bao_cao_chan_doan_ung_thu_vu.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.error("File CSV không đúng cấu trúc 30 cột đặc tính chuẩn của tập Wisconsin.")
    st.markdown("</div>", unsafe_allow_html=True)