import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Predictor | Nicholas Juari",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Muat model .pkl yang sudah dilatih dari pipeline.py"""
    clf = joblib.load("best_clf_pipeline.pkl")
    reg = joblib.load("best_reg_pipeline.pkl")
    return clf, reg

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        padding-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #5a7094;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2980b9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-label { font-size: 0.9rem; opacity: 0.8; }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .placed   { background: linear-gradient(135deg, #27ae60, #2ecc71); }
    .notplaced{ background: linear-gradient(135deg, #c0392b, #e74c3c); }
    .salary   { background: linear-gradient(135deg, #8e44ad, #9b59b6); }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ────────────────────────────────────────────────────
st.markdown('<p class="main-header">🎓 Student Placement Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">DTSC6012001 — Model Deployment | Nicholas Juari | 2802413064</p>', unsafe_allow_html=True)
st.divider()

# ─── SIDEBAR — INPUT FORM ───────────────────────────────────────
st.sidebar.header("📋 Input Data Mahasiswa")

with st.sidebar:
    st.subheader("🎓 Akademik")
    ssc_percentage        = st.slider("SSC Percentage (%)",      40, 100, 70)
    hsc_percentage        = st.slider("HSC Percentage (%)",      40, 100, 70)
    degree_percentage     = st.slider("Degree Percentage (%)",   40, 100, 70)
    cgpa                  = st.slider("CGPA",                    4.0, 10.0, 7.5, step=0.1)
    entrance_exam_score   = st.slider("Entrance Exam Score",     40, 100, 65)
    attendance_percentage = st.slider("Attendance (%)",          50, 100, 80)
    backlogs              = st.number_input("Backlogs",           0, 10, 0)

    st.subheader("💻 Technical & Soft Skills")
    technical_skill_score = st.slider("Technical Skill Score",  40, 100, 65)
    soft_skill_score      = st.slider("Soft Skill Score",       40, 100, 65)
    certifications        = st.number_input("Certifications",    0, 10, 2)

    st.subheader("🌱 Pengalaman & Lifestyle")
    internship_count        = st.number_input("Internship Count",     0, 5, 1)
    live_projects           = st.number_input("Live Projects",        0, 10, 2)
    work_experience_months  = st.number_input("Work Experience (Months)", 0, 60, 6)
    gender                  = st.selectbox("Gender",               ["Male", "Female"])
    extracurricular         = st.selectbox("Extracurricular Activities", ["Yes", "No"])

    predict_btn = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

# ─── MAIN CONTENT ───────────────────────────────────────────────
col_info, col_result = st.columns([1, 1], gap="large")

with col_info:
    st.subheader("📊 Ringkasan Input")
    input_data = {
        "gender":                   gender,
        "ssc_percentage":           ssc_percentage,
        "hsc_percentage":           hsc_percentage,
        "degree_percentage":        degree_percentage,
        "cgpa":                     cgpa,
        "entrance_exam_score":      entrance_exam_score,
        "technical_skill_score":    technical_skill_score,
        "soft_skill_score":         soft_skill_score,
        "internship_count":         internship_count,
        "live_projects":            live_projects,
        "work_experience_months":   work_experience_months,
        "certifications":           certifications,
        "attendance_percentage":    attendance_percentage,
        "backlogs":                 backlogs,
        "extracurricular_activities": extracurricular,
    }
    df_input = pd.DataFrame([input_data])
    st.dataframe(df_input.T.rename(columns={0: "Nilai"}), use_container_width=True)

    # Radar chart skill profile
    st.subheader("🕸️ Skill Profile")
    categories   = ["Academic", "Technical", "Soft Skill", "Attendance", "CGPA"]
    values_radar = [
        (ssc_percentage + hsc_percentage + degree_percentage) / 3,
        technical_skill_score,
        soft_skill_score,
        attendance_percentage,
        cgpa * 10,
    ]
    fig_radar = go.Figure(go.Scatterpolar(
        r=values_radar + [values_radar[0]],
        theta=categories + [categories[0]],
        fill="toself",
        line_color="#2980b9",
        fillcolor="rgba(41,128,185,0.25)",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=300,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_result:
    st.subheader("🎯 Hasil Prediksi")

    if predict_btn:
        try:
            clf_model, reg_model = load_models()
            input_df = pd.DataFrame([input_data])

            # ── KLASIFIKASI ──
            clf_pred = clf_model.predict(input_df)[0]
            clf_proba = clf_model.predict_proba(input_df)[0]

            if clf_pred == 1:
                st.markdown(
                    '<div class="result-card placed">'
                    '<div class="metric-label">Status Penempatan</div>'
                    '<div class="metric-value">✅ DITEMPATKAN</div>'
                    '</div>', unsafe_allow_html=True
                )
                prob_placed = clf_proba[1] * 100
                st.progress(int(prob_placed))
                st.caption(f"Probabilitas ditempatkan: **{prob_placed:.1f}%**")

                # ── REGRESI (hanya jika placed) ──
                reg_pred = reg_model.predict(input_df)[0]
                reg_pred = max(0, reg_pred)  # clamp negatif
                st.markdown(
                    f'<div class="result-card salary">'
                    f'<div class="metric-label">Prediksi Salary</div>'
                    f'<div class="metric-value">₹ {reg_pred:.2f} LPA</div>'
                    f'</div>', unsafe_allow_html=True
                )

                # Gauge chart salary
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=reg_pred,
                    delta={"reference": 5.0, "increasing": {"color": "green"}},
                    gauge={
                        "axis": {"range": [0, 15]},
                        "bar":  {"color": "#8e44ad"},
                        "steps": [
                            {"range": [0,  5], "color": "#f8c3c3"},
                            {"range": [5,  9], "color": "#fde9a2"},
                            {"range": [9, 15], "color": "#b6e2b6"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 3}, "value": 5},
                    },
                    title={"text": "Salary Prediction (LPA)"},
                    number={"suffix": " LPA"},
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            else:
                st.markdown(
                    '<div class="result-card notplaced">'
                    '<div class="metric-label">Status Penempatan</div>'
                    '<div class="metric-value">❌ TIDAK DITEMPATKAN</div>'
                    '</div>', unsafe_allow_html=True
                )
                prob_not = clf_proba[0] * 100
                st.progress(int(prob_not))
                st.caption(f"Probabilitas tidak ditempatkan: **{prob_not:.1f}%**")
                st.info("💡 Tingkatkan CGPA, technical skills, dan jumlah internship untuk meningkatkan peluang.")

            # ── Probability Bar ──
            st.subheader("📊 Distribusi Probabilitas")
            fig_bar = px.bar(
                x=["Tidak Ditempatkan", "Ditempatkan"],
                y=[clf_proba[0]*100, clf_proba[1]*100],
                color=["Tidak Ditempatkan", "Ditempatkan"],
                color_discrete_map={"Tidak Ditempatkan": "#e74c3c", "Ditempatkan": "#2ecc71"},
                labels={"x": "Status", "y": "Probabilitas (%)"},
                text_auto=".1f",
            )
            fig_bar.update_layout(height=250, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

        except FileNotFoundError:
            st.error("❌ Model belum ditemukan! Jalankan `pipeline.py` terlebih dahulu untuk melatih dan menyimpan model.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.info("👈 Isi form di sidebar lalu klik **Prediksi Sekarang**")
        st.image("https://img.icons8.com/fluency/200/graduation-cap.png", width=150)

# ─── FOOTER ─────────────────────────────────────────────────────
st.divider()
st.caption("🎓 Nicholas Juari · 2802413064 · DTSC6012001 Model Deployment · BINUS University 2025/2026")
