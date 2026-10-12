import hashlib
import json
import os
import secrets
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import streamlit as st

# ==============================================================================
# 1. APPLICATION CONFIGURATION & DESIGN SYSTEM (CSS3 CLINICAL DARK)
# ==============================================================================
st.set_page_config(
    page_title="MedVision AI | Decision Support System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b1120;
        background-image: 
            radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(15, 23, 42, 0.8) 0px, transparent 50%);
        color: #f1f5f9;
    }

    /* Cards */
    .med-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(51, 65, 85, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }

    .med-card-danger {
        background: linear-gradient(135deg, rgba(69, 10, 10, 0.85) 0%, rgba(24, 24, 27, 0.9) 100%);
        border: 1px solid #dc2626;
        border-left: 6px solid #ef4444;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
    }

    .med-card-success {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.85) 0%, rgba(24, 24, 27, 0.9) 100%);
        border: 1px solid #059669;
        border-left: 6px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    }

    /* Badges */
    .badge-clinical {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-danger { background: #ef4444; color: #ffffff; }
    .badge-success { background: #10b981; color: #ffffff; }
    
    /* Metrics Header */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# 2. AUTHENTICATION & SECURITY SUBSYSTEM (PBKDF2-HMAC-SHA256)
# ==============================================================================
AUTH_DB_FILE = "doctor_credentials.json"

def secure_hash(password: str, salt_hex: str = None) -> tuple[str, str]:
    """Tạo mã băm bảo mật công nghệ PBKDF2 với Salt chống tấn công giải mã trước."""
    if not salt_hex:
        salt_hex = secrets.token_hex(16)
    salt_bytes = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt_bytes, 120_000)
    return key.hex(), salt_hex

def init_auth_db():
    if not os.path.exists(AUTH_DB_FILE):
        default_pwd_hash, default_salt = secure_hash("MedVision@2026")
        initial_users = {
            "dr.nguyen": {
                "name": "BS. CKI Nguyễn Văn Hoàng",
                "dept": "Khoa Giải phẫu bệnh & Sinh thiết",
                "hash": default_pwd_hash,
                "salt": default_salt,
                "created_at": str(datetime.now().date())
            }
        }
        with open(AUTH_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(initial_users, f, indent=4)

def get_users_db() -> dict:
    init_auth_db()
    with open(AUTH_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def authenticate_doctor(username: str, password_raw: str) -> dict | None:
    db = get_users_db()
    user = db.get(username)
    if not user:
        return None
    test_hash, _ = secure_hash(password_raw, user["salt"])
    return user if test_hash == user["hash"] else None

def register_doctor(username: str, full_name: str, dept: str, password_raw: str) -> tuple[bool, str]:
    db = get_users_db()
    if username in db:
        return False, "Mã Bác sĩ / Tên đăng nhập này đã tồn tại trên cơ sở dữ liệu."
    pwd_hash, salt = secure_hash(password_raw)
    db[username] = {
        "name": full_name,
        "dept": dept,
        "hash": pwd_hash,
        "salt": salt,
        "created_at": str(datetime.now().date())
    }
    with open(AUTH_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4)
    return True, "Cấp tài khoản chuyên môn thành công!"

# Quản lý Trạng thái Session
if "auth" not in st.session_state:
    st.session_state.auth = {
        "logged_in": False,
        "user_id": None,
        "name": None,
        "dept": None
    }
if "history" not in st.session_state:
    st.session_state.history = []

# Màn hình Đăng nhập / Cấp tài khoản
if not st.session_state.auth["logged_in"]:
    _, col_auth, _ = st.columns([1, 1.4, 1])
    with col_auth:
        st.markdown("""
        <div style="text-align: center; margin-top: 40px; margin-bottom: 25px;">
            <div style="font-size: 3rem;">🧬</div>
            <h2 style="color: #38bdf8; margin: 0; font-weight: 800;">MEDVISION ARCHITECTURE</h2>
            <p style="color: #94a3b8; font-size: 0.9rem;">Hệ thống CDSS Phân Tích Hình Thái Tế Bào FNA Vú</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab_login, tab_register = st.tabs(["🔐 Bác Sĩ Đăng Nhập", "📝 Đăng Ký Tài Khoản Y Khoa"])
        
        with tab_login:
            with st.form("form_login_secure"):
                uid = st.text_input("Mã danh tính Bác sĩ (Username)", placeholder="dr.nguyen")
                pwd = st.text_input("Mật khẩu bảo mật", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button("Xác thực thông tin lâm sàng", use_container_width=True)
                
                if submit_login:
                    doc_data = authenticate_doctor(uid.strip(), pwd)
                    if doc_data:
                        st.session_state.auth["logged_in"] = True
                        st.session_state.auth["user_id"] = uid.strip()
                        st.session_state.auth["name"] = doc_data["name"]
                        st.session_state.auth["dept"] = doc_data["dept"]
                        st.rerun()
                    else:
                        st.error("Thông tin xác thực không chính xác hoặc tài khoản chưa được phân quyền.")
            st.info("💡 Tài khoản kiểm thử: `dr.nguyen` | Mật khẩu: `MedVision@2026`")
            
        with tab_register:
            with st.form("form_register_secure"):
                reg_uid = st.text_input("Mã định danh (Username viết liền, không dấu)")
                reg_name = st.text_input("Họ và tên Bác sĩ", placeholder="BS. CKI Trần Mai")
                reg_dept = st.selectbox("Khoa công tác", ["Giải phẫu bệnh", "Ung bướu", "Chẩn đoán hình ảnh", "Ngoại Vú"])
                reg_pwd = st.text_input("Mật khẩu", type="password")
                reg_pwd_confirm = st.text_input("Nhập lại mật khẩu", type="password")
                submit_reg = st.form_submit_button("Tạo tài khoản bác sĩ", use_container_width=True)
                
                if submit_reg:
                    if not (reg_uid and reg_name and reg_pwd):
                        st.warning("Vui lòng nhập đầy đủ các trường thông tin.")
                    elif reg_pwd != reg_pwd_confirm:
                        st.error("Mật khẩu xác nhận không khớp.")
                    elif len(reg_pwd) < 6:
                        st.error("Mật khẩu phải từ 6 ký tự trở lên.")
                    else:
                        ok, msg = register_doctor(reg_uid.strip(), reg_name.strip(), reg_dept, reg_pwd)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
    st.stop()

# ==============================================================================
# 3. MACHINE LEARNING ENGINE (PIPELINE + DEFENSIVE BOOTSTRAP)
# ==============================================================================
MODEL_FILE = 'svm_breast_cancer_model.joblib'

@st.cache_resource
def load_or_train_pipeline():
    """Tải Pipeline chuẩn hoặc tự động huấn luyện Pipeline SVM RBF tối ưu hóa."""
    dataset = load_breast_cancer(as_frame=True)
    df_features = dataset.data
    target = dataset.target
    
    if os.path.exists(MODEL_FILE):
        try:
            payload = joblib.load(MODEL_FILE)
            pipeline = payload['model']
            return pipeline, df_features, target, dataset.target_names
        except Exception:
            pass  # Nếu file hỏng, kích hoạt fallback bên dưới

    # Bootstrap Pipeline: Chuẩn hóa StandardScaler + SVM RBF Probability
    X_train, X_test, y_train, y_test = train_test_split(df_features, target, test_size=0.2, random_state=42, stratify=target)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    joblib.dump({
        'model': pipeline,
        'feature_names': list(df_features.columns),
        'target_names': dataset.target_names
    }, MODEL_FILE)
    
    return pipeline, df_features, target, dataset.target_names

pipeline, df_raw, y_raw, target_names = load_or_train_pipeline()
feature_names = list(df_raw.columns)

# ==============================================================================
# 4. SIDEBAR CONSOLE
# ==============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.7); padding: 18px; border-radius: 12px; border: 1px solid #334155;">
        <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 700; text-transform: uppercase;">Phiên làm việc lâm sàng</div>
        <h4 style="margin: 6px 0 2px 0; color: #f8fafc;">{st.session_state.auth['name']}</h4>
        <div style="font-size: 0.8rem; color: #94a3b8;">{st.session_state.auth['dept']}</div>
        <div style="display: flex; align-items: center; gap: 6px; margin-top: 10px; font-size: 0.8rem; color: #34d399;">
            <span style="height: 8px; width: 8px; background: #10b981; border-radius: 50%;"></span> Hệ thống trực tuyến
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### ⚙️ Cấu Hình Mô Hình")
    st.markdown("""
    - **Kiến trúc:** Support Vector Classifier (RBF)
    - **Chuẩn hóa:** Z-Score Scaler tích hợp
    - **Dataset:** Wisconsin Diagnostic (WDBC)
    - **Số chỉ số sinh thiết:** 30 đặc trưng nhân tế bào
    """)
    
    st.markdown("---")
    if st.button("🚪 Kết thúc phiên làm việc", use_container_width=True):
        st.session_state.auth = {"logged_in": False, "user_id": None, "name": None, "dept": None}
        st.rerun()

# ==============================================================================
# 5. CLINICAL WORKSPACE
# ==============================================================================
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <div>
        <h1 style="color: #38bdf8; margin: 0; font-size: 2rem; font-weight: 800;">CLINICAL DECISION SUPPORT SYSTEM</h1>
        <p style="color: #94a3b8; margin: 0; font-size: 0.95rem;">Hệ Thống Trợ Lý Chẩn Đoán Ung Thư Tuyến Vú FNA Qua Hình Thái Nhân Tế Bào</p>
    </div>
</div>
""", unsafe_allow_html=True)

tab_single, tab_batch, tab_audit = st.tabs([
    "🔬 Khảo Sát Bệnh Án Đơn Lẻ", 
    "📁 Phân Tích Danh Sách Batch (.CSV)", 
    "📋 Nhật Ký Hội Chẩn Phiên Này"
])

# ------------------------------------------------------------------------------
# TAB 1: KHẢO SÁT BỆNH ÁN ĐƠN LẺ
# ------------------------------------------------------------------------------
with tab_single:
    col_preset1, col_preset2, col_preset3 = st.columns(3)
    load_mal = col_preset1.button("🚨 Nạp mẫu: Ca Ác Tính Điển Hình", use_container_width=True)
    load_ben = col_preset2.button("🟢 Nạp mẫu: Ca Lành Tính Điển Hình", use_container_width=True)
    load_avg = col_preset3.button("🔄 Đặt về Giá Trị Trung Bình Quần Thể", use_container_width=True)

    if "patient_data" not in st.session_state or load_avg:
        st.session_state.patient_data = df_raw.mean().to_dict()
    elif load_mal:
        st.session_state.patient_data = df_raw[y_raw == 0].iloc[0].to_dict()
    elif load_ben:
        st.session_state.patient_data = df_raw[y_raw == 1].iloc[0].to_dict()

    with st.form("clinical_input_form"):
        st.markdown("##### 1. Chỉ số sinh thiết nhân tế bào học (FNA)")
        
        mean_features = [f for f in feature_names if "mean" in f]
        se_features = [f for f in feature_names if "error" in f]
        worst_features = [f for f in feature_names if "worst" in f]
        
        current_inputs = {}

        with st.expander("📌 Phân nhóm 1: Kích thước & Hình thái Trung bình (Mean Features)", expanded=True):
            cols = st.columns(3)
            for idx, feature in enumerate(mean_features):
                with cols[idx % 3]:
                    current_inputs[feature] = st.number_input(
                        label=feature,
                        value=float(st.session_state.patient_data.get(feature, 0.0)),
                        format="%.4f"
                    )

        with st.expander("📌 Phân nhóm 2: Độ lệch chuẩn & Sai số đo đạc (Standard Errors)", expanded=False):
            cols = st.columns(3)
            for idx, feature in enumerate(se_features):
                with cols[idx % 3]:
                    current_inputs[feature] = st.number_input(
                        label=feature,
                        value=float(st.session_state.patient_data.get(feature, 0.0)),
                        format="%.4f"
                    )

        with st.expander("📌 Phân nhóm 3: Các chỉ số dị dạng cực đại (Worst Features)", expanded=False):
            cols = st.columns(3)
            for idx, feature in enumerate(worst_features):
                with cols[idx % 3]:
                    current_inputs[feature] = st.number_input(
                        label=feature,
                        value=float(st.session_state.patient_data.get(feature, 0.0)),
                        format="%.4f"
                    )

        execute_inference = st.form_submit_button("🚀 TIẾN HÀNH PHÂN TÍCH LÂM SÀNG", use_container_width=True, type="primary")

    if execute_inference:
        patient_df = pd.DataFrame([current_inputs])
        prediction = pipeline.predict(patient_df)[0]
        probabilities = pipeline.predict_proba(patient_df)[0]
        
        # Trong dataset gốc: class 0 = malignant (ác tính), class 1 = benign (lành tính)
        prob_malignant = probabilities[0] * 100
        prob_benign = probabilities[1] * 100
        is_malignant = (prediction == 0)

        # Lưu vào lịch sử phiên làm việc
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "doctor": st.session_state.auth["name"],
            "diagnosis": "ÁC TÍNH" if is_malignant else "LÀNH TÍNH",
            "malignant_risk": f"{prob_malignant:.2f}%",
            "worst_radius": current_inputs.get("worst radius", 0.0)
        })

        st.markdown("---")
        st.markdown("### 2. Kết Quả Chẩn Đoán & Khuyến Nghị Chuyên Khoa")

        if is_malignant:
            st.markdown(f"""
            <div class="med-card-danger">
                <span class="badge-clinical badge-danger">CẢNH BÁO NGUY CƠ: ÁC TÍNH (MALIGNANT)</span>
                <div class="metric-value" style="color: #ef4444; margin-top: 8px;">
                    {prob_malignant:.2f}% Xác suất Ác tính
                </div>
                <p style="color: #fecaca; margin-top: 10px; font-size: 0.95rem; line-height: 1.6;">
                    <b>Nhận định hình thái:</b> Nhân tế bào biến dạng kích thước lớn, bờ đa cung không đều và tỷ lệ lõm màng nhân cao. 
                    Mô hình SVM xếp cấu trúc mẫu này vào vùng có nguy cơ xâm lấn tế bào học cao.
                </p>
                <div style="background: rgba(0,0,0,0.3); border-left: 4px solid #ef4444; padding: 12px 16px; border-radius: 6px; margin-top: 15px;">
                    <b>🩺 Chỉ định lâm sàng đề xuất:</b> Tiến hành sinh thiết lõi kim (CNB), chỉ định thêm MRI vú có cản quang và hội chẩn đa chuyên khoa trước phẫu thuật.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="med-card-success">
                <span class="badge-clinical badge-success">KẾT QUẢ AN TOÀN: LÀNH TÍNH (BENIGN)</span>
                <div class="metric-value" style="color: #10b981; margin-top: 8px;">
                    {prob_benign:.2f}% Xác suất Lành tính
                </div>
                <p style="color: #a7f3d0; margin-top: 10px; font-size: 0.95rem; line-height: 1.6;">
                    <b>Nhận định hình thái:</b> Quần thể tế bào kích thước đồng dạng, bờ nhân đều đặn, không ghi nhận các điểm phân bào đa dạng bất thường.
                </p>
                <div style="background: rgba(0,0,0,0.3); border-left: 4px solid #10b981; padding: 12px 16px; border-radius: 6px; margin-top: 15px;">
                    <b>🩺 Chỉ định lâm sàng đề xuất:</b> Sang thương lành tính (phù hợp u xơ/thay đổi sợi bọc). Hẹn siêu âm theo dõi sau 6 tháng (Phân nhóm BI-RADS 2).
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 3. Trực Quan Hóa Đa Chiều (AI Explainability)")
        vis_col1, vis_col2 = st.columns(2)

        with vis_col1:
            st.markdown('<div class="med-card"><b style="color: #38bdf8;">📊 Radar Phân Tích 6 Chiều Hình Thái Tế Bào</b>', unsafe_allow_html=True)
            radar_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean concavity']
            
            # Tính Percentile so với phân bố toàn bộ dữ liệu mẫu
            patient_scaled = [
                float(np.clip((current_inputs[rf] - df_raw[rf].min()) / (df_raw[rf].max() - df_raw[rf].min()) * 100, 0, 100))
                for rf in radar_features
            ]
            benign_mean = [
                float(((df_raw[y_raw == 1][rf].mean() - df_raw[rf].min()) / (df_raw[rf].max() - df_raw[rf].min())) * 100)
                for rf in radar_features
            ]
            malignant_mean = [
                float(((df_raw[y_raw == 0][rf].mean() - df_raw[rf].min()) / (df_raw[rf].max() - df_raw[rf].min())) * 100)
                for rf in radar_features
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=patient_scaled, theta=radar_features, fill='toself',
                name='Bệnh nhân hiện tại',
                line_color='#ef4444' if is_malignant else '#10b981'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=benign_mean, theta=radar_features, fill='toself',
                name='Trung bình Lành tính',
                line=dict(color='#10b981', dash='dot'), opacity=0.3
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=malignant_mean, theta=radar_features, fill='toself',
                name='Trung bình Ác tính',
                line=dict(color='#ef4444', dash='dot'), opacity=0.3
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color="#64748b"),
                    angularaxis=dict(color="#94a3b8")
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color="#cbd5e1")),
                margin=dict(l=30, r=30, t=30, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with vis_col2:
            st.markdown('<div class="med-card"><b style="color: #38bdf8;">📍 Không Gian Chiếu Siêu Phẳng Quyết Định (Interactive)</b>', unsafe_allow_html=True)
            
            f_x, f_y = 'worst radius', 'worst concave points'
            scatter_df = df_raw[[f_x, f_y]].copy()
            scatter_df['Phân loại'] = ['Lành tính' if y == 1 else 'Ác tính' for y in y_raw]
            
            fig_scatter = px.scatter(
                scatter_df, x=f_x, y=f_y, color='Phân loại',
                color_discrete_map={'Lành tính': '#10b981', 'Ác tính': '#ef4444'},
                opacity=0.45,
                template="plotly_dark"
            )
            
            # Đánh dấu vị trí bệnh nhân hiện tại
            fig_scatter.add_trace(go.Scatter(
                x=[current_inputs[f_x]],
                y=[current_inputs[f_y]],
                mode='markers+text',
                marker=dict(symbol='star', size=18, color='#38bdf8', line=dict(width=2, color='white')),
                text=["CA ĐANG KHÁM"],
                textposition="top center",
                name="Ca Đang Khám"
            ))
            
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.6)',
                margin=dict(l=30, r=30, t=30, b=30),
                legend=dict(font=dict(color="#cbd5e1"))
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: PHÂN TÍCH THEO DANH SÁCH (BATCH BATCH PROCESSING)
# ------------------------------------------------------------------------------
with tab_batch:
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown("#### 📁 Tải Lên Tập Dữ Liệu FNA Tuyến Vú")
    uploaded_file = st.file_uploader("Chọn file CSV chứa danh sách mẫu mô học", type=["csv"])
    
    if uploaded_file is not None:
        batch_input_df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in feature_names if c not in batch_input_df.columns]
        
        if not missing_cols:
            batch_preds = pipeline.predict(batch_input_df[feature_names])
            batch_probs = pipeline.predict_proba(batch_input_df[feature_names])
            
            result_batch_df = batch_input_df.copy()
            result_batch_df["KẾT LUẬN CDSS"] = ["ÁC TÍNH" if p == 0 else "LÀNH TÍNH" for p in batch_preds]
            result_batch_df["Xác suất Ác tính (%)"] = (batch_probs[:, 0] * 100).round(2)
            result_batch_df["Xác suất Lành tính (%)"] = (batch_probs[:, 1] * 100).round(2)
            
            c_stat1, c_stat2, c_stat3 = st.columns(3)
            total_cases = len(result_batch_df)
            mal_cases = (batch_preds == 0).sum()
            c_stat1.metric("Tổng số mẫu", f"{total_cases} ca")
            c_stat2.metric("Số mẫu nghi ngờ Ác tính", f"{mal_cases} ca", delta=f"{(mal_cases/total_cases)*100:.1f}%", delta_color="inverse")
            c_stat3.metric("Số mẫu Lành tính", f"{total_cases - mal_cases} ca")
            
            st.dataframe(result_batch_df, use_container_width=True)
            
            csv_data = result_batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Xuất Báo Cáo Chẩn Đoán Chi Tiết (.CSV)",
                data=csv_data,
                file_name=f"MedVision_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.error(f"Tập dữ liệu thiếu {len(missing_cols)} cột chỉ số bắt buộc so với chuẩn Wisconsin: {missing_cols[:5]}...")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: NHẬT KÝ HỘI CHẨN TRONG PHIÊN (AUDIT LOG)
# ------------------------------------------------------------------------------
with tab_audit:
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown("#### 📋 Lịch Sử Các Ca Khám Đã Đánh Giá Trong Phiên")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        if st.button("Xóa nhật ký phiên"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Chưa có ca bệnh nào được ghi nhận trong phiên làm việc này.")
    st.markdown('</div>', unsafe_allow_html=True)