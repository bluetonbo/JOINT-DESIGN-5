import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor  # 추가
from sklearn.metrics import r2_score                # 추가
import xgboost as xgb                               # 추가
import io
import sqlite3
import json
import os
import google.generativeai as genai
from datetime import datetime
import time

# 1. 페이지 설정
st.set_page_config(
    layout="wide", 
    page_title="JOINT AI - Process Optimization Suite",
    page_icon="⚡"
)

# 2. 콘솔 스타일 CSS (스크롤 박스 CSS 추가)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #090d16 !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f1524 !important;
        border-right: 1px solid #1e293b;
        min-width: 360px !important;
    }
    
    /* 스크롤 가능한 박스 스타일 추가 */
    .scrollable-box {
        max-height: 400px;
        overflow-y: auto;
        padding: 15px;
        background-color: #0f1524;
        border: 1px solid #223154;
        border-radius: 6px;
        color: #e2e8f0;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    
    .glass-card {
        background: #131b2e;
        border: 1px solid #223154;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    .glass-card-title {
        color: #38bdf8;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #1e293b;
    }

    .stButton>button, .stDownloadButton>button {
        height: 2.8rem !important;
        font-size: 0.9rem !important;
        border-radius: 4px !important;
        background: #10b981 !important;
        color: #ffffff !important;
        font-weight: 600;
        border: none !important;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    /* [추가] 스크롤 박스 스타일 */
    .scrollable-box {
        max-height: 400px;
        overflow-y: auto;
        padding: 15px;
        background-color: #0f1524;
        border: 1px solid #223154;
        border-radius: 6px;
        color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# 3. 다국어 사전 사전 정의
LANG_DICT = {
    "KO": {
        "title": "JOINT 프로세스 인텔리전스",
        "console": "제어 콘솔",
        "upload_title": "마스터 데이터 스트림",
        "upload_help": "신규 로그 파일 업로드 (CSV, XLSX, DB)",
        "upload_hist_help": "기존 누적 DB 파일 업로드 (선택)",
        "init_btn": "엔진 초기화 및 데이터 통합 실행",
        "tab1": "품질 역최적화 타겟팅",
        "tab2": "실시간 가상 시뮬레이터",
        "tab3": "팩토리 데이터레이크 로그",
        "bound_title": "경계 조건 최적화 도구",
        "bound_mode": "안전 경계 제한 모드",
        "kpi_title": "목표 품질 KPI 범위 설정",
        "run_opt": "역추론 최적화 탐색 실행",
        "pred_title": "예측 성능 분석",
        "rec_title": "추천 공정 스펙 조건 (34개 입력 변수)",
        "live_input": "실시간 가상 타겟 범위 설정 (What-If)",
        "run_sim": "가상 역최적화 파라미터 도출",
        "sim_title": "시뮬레이션 역산 도출 결과 (34개 공정 변수)",
        "sim_pred_title": "도출 조건 기준 최종 예측 품질",
        "engine_inactive": "코어 엔진 비활성화: 사이드바를 통해 로그 데이터를 업로드하십시오.",
        "best_algo": "최적 선택 알고리즘",
        "opt_conf": "목표 최적화 신뢰도",
        "dl_format": "내보내기 파일 포맷 선택",
        "dl_btn_spec": "추천 스펙 데이터 다운로드",
        "dl_btn_pred": "예측 성능 데이터 다운로드",
        "db_export_title": "💾 데이터베이스 외부 내보내기",
        "db_prepare_btn": "⚙️ DB 스냅샷 생성 및 서버 저장",
        "db_current_latest": "✨ 최신 데이터 상태가 파일에 이미 반영되어 있습니다.",
        "db_prepared_msg": "준비된 파일: ",
        "db_pc_download": "📥 내보낸 DB 파일 PC로 직접 다운로드",
        "db_save_empty": "저장할 데이터가 없습니다. 먼저 데이터 업로드 후 엔진 초기화를 완료해 주세요.",
        "ai_title": "🤖 JOINT AI 공정 인사이트 가이드",
        "ai_btn": "LLM 기반 공정 가이드라인 생성",
        "ai_loading": "최적화 변수와 품질 KPI 데이터를 분석하여 팩토리 가이드를 생성 중입니다..."
    },
    "EN": {
        "title": "JOINT PROCESS INTELLIGENCE",
        "console": "CONTROL CONSOLE",
        "upload_title": "Master Data Stream",
        "upload_help": "Upload New Log File (CSV, XLSX, DB)",
        "upload_hist_help": "Upload Existing History DB File (Optional)",
        "init_btn": "RUN ENGINE INIT & DATA MERGE",
        "tab1": "QUALITY INVERSE TARGETING",
        "tab2": "REAL-TIME WHAT-IF SIMULATOR",
        "tab3": "FACTORY DATALAKE LOGS",
        "bound_title": "Boundary Condition Optimizer",
        "bound_mode": "Safety Bound Limit Mode",
        "kpi_title": "Target Quality KPIs Range Configurator",
        "run_opt": "RUN INVERSE INFERENCE SEARCH",
        "pred_title": "Predicted Performance Analysis",
        "rec_title": "Recommended Process Specifications (34 Variables)",
        "live_input": "Real-time Virtual Target Range Configurator (What-If)",
        "run_sim": "EXECUTE VIRTUAL INVERSE OPTIMIZATION",
        "sim_title": "Inversed Simulation Results (34 Process Variables)",
        "sim_pred_title": "Final Predicted Quality via Inversed Specs",
        "engine_inactive": "CORE ENGINE INACTIVE: Please upload log data via sidebar.",
        "best_algo": "Selected Best Algorithm",
        "opt_conf": "Target Optimization Confidence",
        "dl_format": "Select Export File Format",
        "dl_btn_spec": "DOWNLOAD RECOMMENDED SPECS",
        "dl_btn_pred": "DOWNLOAD PREDICTED PERFORMANCE",
        "db_export_title": "💾 External Database Export",
        "db_prepare_btn": "⚙️ Generate & Save DB Snapshot",
        "db_current_latest": "✨ The file contains the latest data state.",
        "db_prepared_msg": "Prepared File: ",
        "db_pc_download": "📥 Download Saved DB File to PC Directly",
        "db_save_empty": "No data available to save. Please run Engine Initialization first.",
        "ai_title": "🤖 JOINT AI Process Insight Guidance",
        "ai_btn": "Generate LLM-based Process Guidelines",
        "ai_loading": "Analyzing optimized variables and quality KPI data to generate factory guidance..."
    }
}

if "lang" not in st.session_state:
    st.session_state["lang"] = "KO"

# 4. 인증 시스템
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        st.markdown("<br><br><div class='glass-card' style='text-align: center; padding: 40px;'><h2 style='color: #10b981;'>JOINT AI SYSTEM</h2></div>", unsafe_allow_html=True)
        pwd = st.text_input("Enter Password", type="password")
        if st.button("AUTHENTICATE SYSTEM"):
            if pwd == "admin1234":
                st.session_state.authenticated = True
                st.rerun()
            else: st.error("Invalid credentials.")
    st.stop()

col_title, col_lang = st.columns([8, 1])
with col_lang:
    lang_choice = st.selectbox("🌐 Lang", ["KO", "EN"], index=0 if st.session_state["lang"] == "KO" else 1, label_visibility="collapsed")
    if lang_choice != st.session_state["lang"]:
        st.session_state["lang"] = lang_choice
        st.rerun()

L_G = LANG_DICT[st.session_state["lang"]]

# 5. 입력 변수 및 타겟 정의
X_list = [
    'BD', 'CID', 'BH', 'CIH', 'CITH', 'COHB', 'CD1', 'CD2', 'CD3', 'CD4', 'CD5', 
    'CH1', 'CH2', 'CH3', 'CH4', 'CGW', 'CGD', 'CD6', 'CR', 'CF', 'SH1', 'SR', 
    'SH2', 'SH3', 'SH4', 'SH5', 'SH6', 'SID', 'SOD', 'SH7', 'SH8', 'SH9', 'SR2', 'COHA'
]
# 1. 먼저 정의해야 합니다 (이 코드가 루프보다 위에 있어야 함)
target_vars = ["BT", "RT", "AGB", "RGB", "AGA", "RGA", "AGI", "RGI"]

spec_limits = {
    "BT": (5.0, 7.0),
    "RT": (2.0, 4.0),
    "AGB": (0.0, 0.2),
    "RGB": (0.0, 0.4),
    "AGA": (0.0, 1.0),
    "RGA": (0.0, 1.0),
    "AGI": (0.0, 1.0),
    "RGI": (0.0, 1.0)
}

# 2. 그 다음에 루프를 실행해야 합니다
for tgt in target_vars:
    # 이제 spec_limits가 정의되어 있으므로 오류가 발생하지 않습니다
    min_val, max_val = spec_limits.get(tgt, (0.0, 1.0))
    st.session_state[f'opt_{tgt.lower()}'] = st.slider(
        f"Target {tgt} Range",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        key=f"slider_{tgt}"
    )
    

# [추가] 원본 학습데이터(VOLVO_SPA12_CABJ_TRAIN_DATA) 헤더 기반 변수명 약자 -> 실제 의미 매핑 사전
# 이 앱이 사용하는 변수명은 볼스터드(Ball Stud) 조인트 스웨이징(swaging) 조립 공정 검사 데이터의 약자임
VAR_GLOSSARY = {
    'BD': 'ballstud_diameter_mm (볼스터드 직경)',
    'CID': 'case_inner_diameter_mm (케이스 내경)',
    'BH': 'bearing_Height_mm (베어링 높이)',
    'CIH': 'case_inner_height_mm (케이스 내측 높이)',
    'CITH': 'case_inner_taper_height_mm (케이스 내측 테이퍼 높이)',
    'COHB': 'case_outer_height_before_mm (케이스 외측 높이, 스웨이징 전)',
    'COHA': 'case_outer_height_after_mm (케이스 외측 높이, 스웨이징 후)',
    'CD1': 'case_d1_mm (케이스 치수 D1)',
    'CD2': 'case_d2_mm (케이스 치수 D2)',
    'CD3': 'case_d3_mm (케이스 치수 D3)',
    'CD4': 'case_d4_mm (케이스 치수 D4)',
    'CD5': 'case_d5_mm (케이스 치수 D5)',
    'CD6': 'case_d6_mm (케이스 치수 D6)',
    'CH1': 'case_h1_mm (케이스 높이 H1)',
    'CH2': 'case_h2_mm (케이스 높이 H2)',
    'CH3': 'case_h3_mm (케이스 높이 H3)',
    'CH4': 'case_h4_mm (케이스 높이 H4)',
    'CGW': 'case_groove_width_mm (케이스 그루브 폭)',
    'CGD': 'case_groove_depth_mm (케이스 그루브 깊이)',
    'CR': 'case_roundness_mm (케이스 진원도)',
    'CF': 'case_flatness_mm (케이스 평면도)',
    'SH1': 'seat_h1_mm (시트 높이 H1)',
    'SR': 'seat_R_mm (시트 R 치수)',
    'SH2': 'seat_h2_mm (시트 높이 H2)',
    'SH3': 'seat_h3_mm (시트 높이 H3)',
    'SH4': 'seat_h4_mm (시트 높이 H4)',
    'SH5': 'seat_h5_mm (시트 높이 H5)',
    'SH6': 'seat_h6_mm (시트 높이 H6)',
    'SID': 'seat_inner_d_mm (시트 내경)',
    'SOD': 'seat_outer_d_mm (시트 외경)',
    'SH7': 'seat_h7_mm (시트 높이 H7)',
    'SH8': 'seat_h8_mm (시트 높이 H8)',
    'SH9': 'seat_h9_mm (시트 높이 H9)',
    'SR2': 'seat_R2_mm (시트 R2 치수)',
}

TARGET_GLOSSARY = {
    'BT': 'breakaway_torque_Nm (분리 토크 / 초기 회전 토크)',
    'RT': 'running_torque_Nm (회전 토크)',
    'AGB': 'axial_gap_before_mm (축방향 유격, 내구 시험 전)',
    'RGB': 'radial_gap_before_mm (반경방향 유격, 내구 시험 전)',
    'AGA': 'axial_gap_after_mm (축방향 유격, 내구 시험 후)',
    'RGA': 'radial_gap_after_mm (반경방향 유격, 내구 시험 후)',
    'AGI': 'axial_gap_increase_mm (축방향 유격 증가량, 시험 전후 차)',
    'RGI': 'radial_gap_increase_mm (반경방향 유격 증가량, 시험 전후 차)',
}

def on_slider_change(prefix):
    val_tuple = st.session_state[f'{prefix}_s_val']
    st.session_state[f'{prefix}_n_min'] = val_tuple[0]
    st.session_state[f'{prefix}_n_max'] = val_tuple[1]

def on_min_change(prefix):
    current_slider = st.session_state[f'{prefix}_s_val']
    new_min = st.session_state[f'{prefix}_n_min']
    if new_min > current_slider[1]:
        new_min = current_slider[1]
        st.session_state[f'{prefix}_n_min'] = new_min
    st.session_state[f'{prefix}_s_val'] = (new_min, current_slider[1])

def on_max_change(prefix):
    current_slider = st.session_state[f'{prefix}_s_val']
    new_max = st.session_state[f'{prefix}_n_max']
    if new_max < current_slider[0]:
        new_max = current_slider[0]
        st.session_state[f'{prefix}_n_max'] = new_max
    st.session_state[f'{prefix}_s_val'] = (current_slider[0], new_max)

def on_sim_slider_change(prefix):
    val_tuple = st.session_state[f'sim_tgt_{prefix}_s_val']
    st.session_state[f'sim_tgt_{prefix}_n_min'] = val_tuple[0]
    st.session_state[f'sim_tgt_{prefix}_n_max'] = val_tuple[1]

def on_sim_min_change(prefix):
    current_slider = st.session_state[f'sim_tgt_{prefix}_s_val']
    new_min = st.session_state[f'sim_tgt_{prefix}_n_min']
    if new_min > current_slider[1]:
        new_min = current_slider[1]
        st.session_state[f'sim_tgt_{prefix}_n_min'] = new_min
    st.session_state[f'sim_tgt_{prefix}_s_val'] = (new_min, current_slider[1])

def on_sim_max_change(prefix):
    current_slider = st.session_state[f'sim_tgt_{prefix}_s_val']
    new_max = st.session_state[f'sim_tgt_{prefix}_n_max']
    if new_max < current_slider[0]:
        new_max = current_slider[0]
        st.session_state[f'sim_tgt_{prefix}_n_max'] = new_max
    st.session_state[f'sim_tgt_{prefix}_s_val'] = (current_slider[0], new_max)

# [최종 수정된 AI 함수: 404 및 429 오류 완벽 방지 + 실제 변수 의미 기반 컨텍스트 고정(grounding)]
def generate_ai_guidance(process_specs, predicted_kpis, mode="Optimization"):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ API 키가 환경변수에 설정되지 않았습니다."

    genai.configure(api_key=api_key)

    # 변수 약자를 실제 의미(VAR_GLOSSARY/TARGET_GLOSSARY)와 함께 표기하여 AI가 정확히 이해하도록 함
    specs_str = "\n".join([
        f"  - {k} [{VAR_GLOSSARY.get(k, '정의되지 않은 변수')}]: {v:.3f}"
        for k, v in process_specs.items()
    ])
    kpis_lines = []
    for k, v in predicted_kpis.items():
        spec_range = SPEC_GUIDE.get(k, "정의되지 않음")
        glossary = TARGET_GLOSSARY.get(k, '정의되지 않은 KPI')
        kpis_lines.append(f"  - {k} [{glossary}]: {v:.3f}  (정상 스펙 범위: {spec_range})")
    kpis_str = "\n".join(kpis_lines)

    mode_desc = {
        "Optimization": "사용자가 설정한 목표 품질(KPI) 범위를 만족시키기 위해 역최적화 알고리즘으로 도출한 '추천 공정 스펙' 결과",
        "Simulation": "사용자가 What-If 시뮬레이터에서 가상으로 설정한 목표 품질 범위에 대해 역최적화로 도출한 '가상 시뮬레이션' 결과"
    }.get(mode, "공정 최적화 결과")

    # 시스템 지시문: VOLVO SPA1/2 CABJ 볼스터드 조인트 스웨이징 공정 데이터를 학습한 시스템임을 명확히 고정
    system_instruction = (
        "당신은 'JOINT AI - Process Optimization Suite'에 내장된 공정 엔지니어링 전문 AI 어시스턴트입니다. "
        "이 시스템은 VOLVO SPA1/2 플랫폼 CABJ(볼스터드 조인트, Ball Stud Joint) 부품의 스웨이징(swaging) 조립 "
        "공정 검사 데이터를 학습하여, 목표 품질 KPI(분리 토크, 회전 토크, 축/반경방향 유격 등)를 만족하는 "
        "34개 단품/조립 치수 공정 변수(볼스터드 직경, 케이스 내경, 시트 높이 등) 조합을 역최적화로 도출합니다.\n\n"
        "다음 규칙을 반드시 지키세요:\n"
        "1. 답변은 오직 아래 사용자 메시지로 주어지는 '공정 변수 값'과 '예측 KPI 값' 데이터에만 근거해야 합니다.\n"
        "2. 각 변수명 뒤 대괄호 안의 실제 의미(예: case_inner_diameter_mm)를 참고하여 볼스터드 조인트 조립 공정 관점에서 해석하세요.\n"
        "3. 이 데이터와 무관한 일반 지식, 다른 산업, 다른 부품, 다른 주제로 답변을 확장하지 마세요.\n"
        "4. 변수명이나 수치가 불명확하더라도 임의로 새로운 정보를 지어내지 말고, 주어진 수치 범위 내에서만 해석하세요.\n"
        "5. 답변은 한국어로, 공정 엔지니어가 현장에서 바로 참고할 수 있는 실무형 보고서 형식으로 작성하세요."
    )

    prompt = f"""아래는 JOINT 공정 최적화 시스템(VOLVO SPA1/2 CABJ 볼스터드 조인트 스웨이징 공정)에서 도출된 결과 데이터입니다. 이 데이터를 분석하여 공정 가이드라인을 작성해 주세요.

[분석 모드]
{mode} - {mode_desc}

[도출된 공정 변수 스펙 (34개 단품/조립 치수 변수)]
{specs_str}

[해당 조건에서의 예측 품질 KPI]
{kpis_str}

[요청 사항 - 아래 형식에 맞춰 작성]
1. 결과 요약: 위 공정 조건과 예측 KPI가 의미하는 바를 핵심만 요약
2. KPI 적합성 평가: 각 KPI가 정상 스펙 범위 내에 있는지, 범위를 벗어났다면 어느 정도 벗어났는지 평가
3. 주의가 필요한 공정 변수: 위 34개 변수 중 볼스터드 조인트 스웨이징 공정 특성상 특이하거나 KPI에 영향을 줄 수 있는 변수가 있다면 언급 (없으면 '특이사항 없음'으로 명시)
4. 현장 적용 권장사항: 이 공정 조건을 실제 라인에 적용할 때 점검해야 할 사항을 3가지 이내로 제시

반드시 위에 제공된 수치 데이터만을 근거로 작성하고, 데이터에 없는 내용은 추측하지 마세요."""

    try:
        # 1. 현재 계정에서 호출 가능한 모델 리스트 자동 추출 (404 방지)
        available_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        if not available_models:
            return "❌ 사용 가능한 AI 모델을 찾을 수 없습니다."

        # 2. 우선순위 모델 설정 (flash를 우선 사용)
        priority = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash']
        target_model = next((m for m in priority if m in available_models), available_models[0])

        # 3. 모델 생성 (system_instruction으로 도메인/역할 고정) 및 호출
        model = genai.GenerativeModel(target_model, system_instruction=system_instruction)
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return "⏳ API 사용량이 많습니다. 잠시 후 다시 시도해 주세요."
        return f"❌ AI 생성 오류: {error_msg}"

# (이하 기존 코드의 나머지 로직을 동일하게 유지)

if 'scaler' not in st.session_state:
    init_dict = {
        'scaler': None, 'df_caulking': pd.DataFrame(),
        'process_vars': X_list, 'target_vars': target_vars, 'active_x_list': X_list,
        'optimizer_status': "STANDBY", 'opt_result_x': None, 'confidence_score': None, 'sim_confidence': None,
        'best_algorithm_used': "SLSQP", 'sim_result_x': None,
        'prepared_db_file': None, 'data_changed_since_save': False,
        'ai_analysis_result': None,
        'valid_target_vars': target_vars
    }
    for tgt in target_vars:
        init_dict[f'model_{tgt.lower()}'] = None
        init_dict[f'opt_pred_{tgt.lower()}'] = None
        init_dict[f'sim_pred_{tgt.lower()}'] = None
    for var in X_list:
        init_dict[f'm_{var.lower()}_min'] = 0.0
        init_dict[f'm_{var.lower()}_max'] = 100.0
        init_dict[f'sim_{var.lower()}'] = 0.0
        
    for tgt in target_vars:
        t_low = tgt.lower()
        if tgt == 'BT': range_val = (0.0, 8.0)
        elif tgt == 'RT': range_val = (0.0, 4.0)
        elif tgt == 'AGB': range_val = (0.0, 0.3)
        else: range_val = (0.0, 1.0)
        
        init_dict[f'{t_low}_s_val'] = range_val
        init_dict[f'{t_low}_n_min'] = range_val[0]
        init_dict[f'{t_low}_n_max'] = range_val[1]
        
        init_dict[f'sim_tgt_{t_low}_s_val'] = range_val
        init_dict[f'sim_tgt_{t_low}_n_min'] = range_val[0]
        init_dict[f'sim_tgt_{t_low}_n_max'] = range_val[1]
        
    st.session_state.update(init_dict)

# 6. 사이드바 제어반
with st.sidebar:
    st.markdown(f"<h2 style='color: #ffffff; font-size:1.15rem; margin-bottom: 20px;'>{L_G['console']}</h2>", unsafe_allow_html=True)
    with st.expander(L_G['upload_title'], expanded=True):
        u_input = st.file_uploader(L_G['upload_help'], type=['csv','xlsx','db'], key="new_data_file")
        db_input = st.file_uploader(L_G['upload_hist_help'], type=['db'], key="history_db_file")

    if st.button(L_G['init_btn'], type="primary"):
        if u_input or db_input:
            data_frames = []
            
            if u_input:
                try:
                    if u_input.name.endswith('.db'):
                        temp_db = "temp_uploaded_joint.db"
                        with open(temp_db, "wb") as t:
                            t.write(u_input.getvalue())
                        conn = sqlite3.connect(temp_db)
                        df_temp = pd.read_sql_query("SELECT vars FROM production_log", conn)
                        conn.close()
                        if os.path.exists(temp_db):
                            os.remove(temp_db)
                        df_new = pd.json_normalize([json.loads(x) for x in df_temp['vars']])
                    elif u_input.name.endswith('csv'):
                        df_new = pd.read_csv(u_input, na_values=['N/A', 'n/a', 'NA', 'N/A ', ' N/A', '-', 'null'])
                    else:
                        df_new = pd.read_excel(u_input, na_values=['N/A', 'n/a', 'NA', 'N/A ', ' N/A', '-', 'null'])
                    data_frames.append(df_new)
                except Exception as e:
                    st.sidebar.error(f"신규 파일 로드 오류: {e}")

            if db_input:
                try:
                    temp_db_hist = "temp_uploaded_hist.db"
                    with open(temp_db_hist, "wb") as t:
                        t.write(db_input.getvalue())
                    conn = sqlite3.connect(temp_db_hist)
                    df_temp_hist = pd.read_sql_query("SELECT vars FROM production_log", conn)
                    conn.close()
                    if os.path.exists(temp_db_hist):
                        os.remove(temp_db_hist)
                    df_hist = pd.json_normalize([json.loads(x) for x in df_temp_hist['vars']])
                    data_frames.append(df_hist)
                except Exception as e:
                    st.sidebar.error(f"기존 DB 파일 로드 오류: {e}")

            df_comb = None
            if data_frames:
                df_comb = pd.concat(data_frames, ignore_index=True)
                if 'vars' in df_comb.columns:
                    df_comb = df_comb.drop(columns=['vars'], errors='ignore')

            if df_comb is not None:
                df_comb.columns = [c.strip() for c in df_comb.columns]
                
                valid_targets = []
                for tgt in target_vars:
                    if tgt in df_comb.columns:
                        converted = pd.to_numeric(df_comb[tgt], errors='coerce')
                        if not converted.isna().all():
                            valid_targets.append(tgt)
                
                if not valid_targets:
                    valid_targets = ['BT']
                
                for col in X_list + target_vars:
                    if col in df_comb.columns:
                        df_comb[col] = pd.to_numeric(df_comb[col], errors='coerce')
                    else:
                        df_comb[col] = np.nan
                
                df_imputed = df_comb.copy()
                for var in X_list:
                    df_imputed[var] = df_imputed[var].fillna(df_imputed[var].median()) if not df_imputed[var].isna().all() else 0.0
                for tgt in target_vars:
                    df_imputed[tgt] = df_imputed[tgt].fillna(df_imputed[tgt].median()) if not df_imputed[tgt].isna().all() else 0.0
                
                scaler = MinMaxScaler().fit(df_imputed[X_list])
                
               # [수정된 모델 자동 선택 엔진 - 들여쓰기 교정본]
                models = {}
                model_metadata = {} 
                
                train_progress_bar = st.sidebar.progress(0, text="모델 학습 준비 중...")
                total_targets_n = len(target_vars)
                
                for t_idx, target in enumerate(target_vars):
                    train_progress_bar.progress(
                        t_idx / total_targets_n,
                        text=f"⚙️ KPI 모델 학습 중 ({t_idx+1}/{total_targets_n}): {target}"
                    )
                    X_scaled_t = scaler.transform(df_imputed[X_list])
                    y_t = df_imputed[target]
                    
                    # 후보 모델 생성
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror')
                    lr_model = LinearRegression()
                    
                    # 학습
                    rf_model.fit(X_scaled_t, y_t)
                    xgb_model.fit(X_scaled_t, y_t)
                    lr_model.fit(X_scaled_t, y_t)
                    
                    # R2 Score 성능 평가
                    scores = {
                        "RandomForest": r2_score(y_t, rf_model.predict(X_scaled_t)),
                        "XGBoost": r2_score(y_t, xgb_model.predict(X_scaled_t)),
                        "Linear": r2_score(y_t, lr_model.predict(X_scaled_t))
                    }
                    
                    # 최고 성능 모델 자동 채택
                    best_algo_name = max(scores, key=scores.get)
                    models[f'model_{target.lower()}'] = rf_model if best_algo_name == "RandomForest" else (xgb_model if best_algo_name == "XGBoost" else lr_model)
                    model_metadata[f'algo_{target.lower()}'] = best_algo_name
                
                train_progress_bar.progress(1.0, text="✅ 모든 KPI 모델 학습 완료")
                st.session_state['model_metadata'] = model_metadata
                
                data_bounds = {}
                update_data = {
                    'scaler': scaler, 'df_caulking': df_comb, 'optimizer_status': "ENGINE READY",
                    'prepared_db_file': None, 'data_changed_since_save': True,
                    'valid_target_vars': valid_targets
                }
                for tgt in target_vars:
                    update_data[f'model_{tgt.lower()}'] = models[f'model_{tgt.lower()}']
                
                for var in X_list:
                    v_min = float(df_imputed[var].min())
                    v_max = float(df_imputed[var].max())
                    if v_min == v_max: v_max += 1.0
                    data_bounds[var] = (v_min, v_max)
                    update_data[f'm_{var.lower()}_min'] = v_min
                    update_data[f'm_{var.lower()}_max'] = v_max
                    update_data[f'sim_{var.lower()}'] = float(df_imputed[var].median())
                    
                update_data['data_bounds'] = data_bounds
                
                for tgt in target_vars:
                    t_low = tgt.lower()
                    if tgt == 'BT': range_val = (0.0, 8.0)
                    elif tgt == 'RT': range_val = (0.0, 4.0)
                    elif tgt == 'AGB': range_val = (0.0, 0.3)
                    else: range_val = (0.0, 1.0)
                    
                    update_data[f'{t_low}_s_val'] = range_val
                    update_data[f'{t_low}_n_min'] = range_val[0]
                    update_data[f'{t_low}_n_max'] = range_val[1]
                    update_data[f'sim_tgt_{t_low}_s_val'] = range_val
                    update_data[f'sim_tgt_{t_low}_n_min'] = range_val[0]
                    update_data[f'sim_tgt_{t_low}_n_max'] = range_val[1]
                    
                st.session_state.update(update_data)
                st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<h3 style='color:#00e5ff; font-size:1.1rem;'>{L_G['db_export_title']}</h3>", unsafe_allow_html=True)
    
    if not st.session_state['df_caulking'].empty:
        if st.sidebar.button(L_G['db_prepare_btn'], key="btn_create_db_snapshot"):
            today_str = datetime.now().strftime("%Y%m%d")
            
            if st.session_state.get('data_changed_since_save', True) or st.session_state['prepared_db_file'] is None:
                idx = 1
                while True:
                    candidate = f"joint-{today_str}-{idx}.db"
                    if not os.path.exists(candidate):
                        final_filename = candidate
                        break
                    idx += 1
                st.session_state['prepared_db_file'] = final_filename
            else:
                final_filename = st.session_state['prepared_db_file']
            
            try:
                existing_df = pd.DataFrame()
                if os.path.exists(final_filename):
                    try:
                        conn_old = sqlite3.connect(final_filename)
                        df_old_raw = pd.read_sql_query("SELECT vars FROM production_log", conn_old)
                        conn_old.close()
                        existing_df = pd.json_normalize([json.loads(x) for x in df_old_raw['vars']])
                    except Exception:
                        existing_df = pd.DataFrame()

                df_to_save = st.session_state['df_caulking'].copy()
                if 'vars' in df_to_save.columns:
                    df_to_save = df_to_save.drop(columns=['vars'], errors='ignore')
                
                if not existing_df.empty:
                    df_to_save = pd.concat([existing_df, df_to_save], ignore_index=True)

                conn = sqlite3.connect(final_filename)
                df_to_save['vars'] = df_to_save.apply(lambda row: json.dumps(row.to_dict()), axis=1)
                df_to_save[['vars']].to_sql("production_log", conn, if_exists="replace", index=False)
                conn.close()
                
                st.session_state['data_changed_since_save'] = False
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
        
        if st.session_state.get('prepared_db_file'):
            target_file = st.session_state['prepared_db_file']
            if os.path.exists(target_file):
                try:
                    with open(target_file, "rb") as f:
                        db_bytes = f.read()
                    
                    if not st.session_state.get('data_changed_since_save', True):
                        st.sidebar.markdown(f"<span style='color:#a3e635; font-size:0.85rem;'>{L_G['db_current_latest']}</span>", unsafe_allow_html=True)
                    
                    st.sidebar.markdown(f"✅ {L_G['db_prepared_msg']} `{target_file}`")
                    st.sidebar.download_button(
                        label=L_G['db_pc_download'],
                        data=db_bytes,
                        file_name=target_file,
                        mime="application/x-sqlite3",
                        key="db_final_download_action"
                    )
                except Exception as e:
                    st.sidebar.error(f"File Load Error: {e}")
    else:
        st.sidebar.warning(L_G['db_save_empty'])

# 7. 메인 뷰포트
if st.session_state['scaler'] is not None:
    db = st.session_state['data_bounds']
    valid_tgts = st.session_state['valid_target_vars']
    
    st.markdown(f"<h1 style='margin-bottom:20px; font-size:1.8rem;'>{L_G['title']}</h1>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs([L_G['tab1'], L_G['tab2'], L_G['tab3']])

    with tab1:
        layout_l, layout_r = st.columns([1.3, 1.2], gap="large")
        with layout_l:
            st.markdown(f"<div class='glass-card'><div class='glass-card-title'>{L_G['bound_title']}</div>", unsafe_allow_html=True)
            bound_mode = st.radio(L_G['bound_mode'], options=["Auto Mode", "Manual Expert Tuning"], index=0, horizontal=True, label_visibility="collapsed")
            
            chosen_bounds = {}
            if "Auto Mode" in bound_mode:
                bound_text = ""
                for v in X_list:
                    bound_text += f"• {v}: {db[v][0]:.3f}~{db[v][1]:.3f}<br>"
                st.markdown(f"<div style='background:#0f172a; padding:15px; border-radius:6px; border:1px solid #1e293b; font-size:0.85rem; line-height:1.5; max-height:200px; overflow-y:auto;'><span style='color:#38bdf8; font-weight:600;'>[34 Variables Mapping Status]</span><br>{bound_text}</div>", unsafe_allow_html=True)
                for v in X_list:
                    chosen_bounds[v] = db[v]
            else:
                st.markdown("<div style='max-height:300px; overflow-y:auto; padding-right:10px;'>", unsafe_allow_html=True)
                for v in X_list:
                    v_clean = v.lower()
                    st.session_state[f'm_{v_clean}_min'], st.session_state[f'm_{v_clean}_max'] = st.slider(
                        f"{v} Range", float(db[v][0]*0.5), float(db[v][1]*1.5), (st.session_state[f'm_{v_clean}_min'], st.session_state[f'm_{v_clean}_max']), 0.005
                    )
                    chosen_bounds[v] = (st.session_state[f'm_{v_clean}_min'], st.session_state[f'm_{v_clean}_max'])
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown(f"</div><div class='glass-card'><div class='glass-card-title'>{L_G['kpi_title']}</div>", unsafe_allow_html=True)
            
            for idx, tgt in enumerate(valid_tgts):
                t_low = tgt.lower()
                spec_txt = SPEC_GUIDE.get(tgt, "0.0 ~ 1.0")
                max_slider_val = 20.0 if tgt in ['BT', 'RT'] else (0.3 if tgt == 'AGB' else 5.0)
                
                st.markdown(f"<p style='font-size:0.85rem; font-weight:600; color:#38bdf8; margin-bottom:5px;'>{idx+1}. Target {tgt} Range (Spec: {spec_txt})</p>", unsafe_allow_html=True)
                col_c1, col_c2 = st.columns([1.8, 1.2])
                with col_c1:
                    st.slider(f"{tgt} Slider UI", 0.0, float(max_slider_val), step=0.001 if tgt=='AGB' else 0.05, label_visibility="collapsed", key=f"{t_low}_s_val", on_change=on_slider_change, args=(t_low,))
                with col_c2:
                    num_c1, num_c2 = st.columns(2)
                    num_c1.number_input("Min", step=0.001 if tgt=='AGB' else 0.1, key=f"{t_low}_n_min", on_change=on_min_change, args=(t_low,))
                    num_c2.number_input("Max", step=0.001 if tgt=='AGB' else 0.1, key=f"{t_low}_n_max", on_change=on_max_change, args=(t_low,))
                st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)

            undefined_tgts = [t for t in target_vars if t not in valid_tgts]
            if undefined_tgts:
                with st.expander(f"Undefined Quality Targets (미정 항목 {len(undefined_tgts)}종 제어단)"):
                    for tgt in undefined_tgts:
                        t_low = tgt.lower()
                        st.markdown(f"<p style='font-size:0.8rem; margin:2px 0; color:#94a3b8;'>• {tgt} Range</p>", unsafe_allow_html=True)
                        cx1, cx2 = st.columns([1.8, 1.2])
                        with cx1:
                            st.slider(f"{tgt} S", -0.5, 5.0, step=0.01, label_visibility="collapsed", key=f"{t_low}_s_val", on_change=on_slider_change, args=(t_low,))
                        with cx2:
                            sub_nc1, sub_nc2 = st.columns(2)
                            sub_nc1.number_input("Min", step=0.01, key=f"{t_low}_n_min", on_change=on_min_change, args=(t_low,))
                            sub_nc2.number_input("Max", step=0.01, key=f"{t_low}_n_max", on_change=on_max_change, args=(t_low,))
                
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button(L_G['run_opt'], type="secondary"):
                def target_loss(x):
                    df_x = pd.DataFrame([x], columns=X_list)
                    q = st.session_state['scaler'].transform(df_x)
                    
                    total_loss = 0.0
                    for tgt in target_vars:
                        model_key = f'model_{tgt.lower()}'
                        if st.session_state[model_key] is not None:
                            pred = st.session_state[model_key].predict(q)[0]
                            t_range = st.session_state[f'{tgt.lower()}_s_val']
                            total_loss += max(0, t_range[0] - pred)**2 + max(0, pred - t_range[1])**2
                    return total_loss
                
                init_x = [(db[v][0] + db[v][1]) / 2 for v in X_list]
                bands = [db[v] for v in X_list]
                
                algorithms = ['L-BFGS-B', 'SLSQP', 'Powell', 'Nelder-Mead']
                best_loss = float('inf')
                best_res = None
                selected_algo = 'SLSQP'
                
                opt_progress_bar = st.progress(0, text="역추론 최적화 탐색 준비 중...")
                total_algos_n = len(algorithms)
                
                for a_idx, algo in enumerate(algorithms):
                    opt_progress_bar.progress(
                        a_idx / total_algos_n,
                        text=f"🔍 알고리즘 탐색 중 ({a_idx+1}/{total_algos_n}): {algo}"
                    )
                    try:
                        if algo in ['L-BFGS-B', 'SLSQP']:
                            res_temp = minimize(target_loss, init_x, method=algo, bounds=bands)
                        else:
                            res_temp = minimize(target_loss, init_x, method=algo)
                        
                        final_x = np.clip(res_temp.x, [b[0] for b in bands], [b[1] for b in bands])
                        current_score_loss = target_loss(final_x)
                        
                        if current_score_loss < best_loss:
                            best_loss = current_score_loss
                            best_res = res_temp
                            best_res.x = final_x
                            selected_algo = algo
                    except Exception as e:
                        continue
                
                opt_progress_bar.progress(1.0, text=f"✅ 최적화 완료 (선택된 알고리즘: {selected_algo})")

                q_opt = st.session_state['scaler'].transform(pd.DataFrame([best_res.x], columns=X_list))
                
                update_opt_dict = {
                    'opt_result_x': best_res.x, 
                    'confidence_score': round(max(50.0, 100.0 - (best_loss * 10)), 1),
                    'best_algorithm_used': selected_algo,
                    'ai_analysis_result': None
                }
                for tgt in target_vars:
                    model_key = f'model_{tgt.lower()}'
                    if st.session_state[model_key] is not None:
                        update_opt_dict[f'opt_pred_{tgt.lower()}'] = float(st.session_state[model_key].predict(q_opt)[0])
                    else:
                        update_opt_dict[f'opt_pred_{tgt.lower()}'] = 0.0
                        
                st.session_state.update(update_opt_dict)
                st.rerun()

            if st.session_state['opt_result_x'] is not None:
                st.markdown(f"<div class='glass-card'><div class='glass-card-title' style='color:#a3e635;'>{L_G['ai_title']}</div>", unsafe_allow_html=True)
                if st.button(L_G['ai_btn'], key="ai_btn_tab1"):
                    with st.spinner(L_G['ai_loading']):
                        current_p_specs = {X_list[i]: st.session_state['opt_result_x'][i] for i in range(len(X_list))}
                        current_kpis = {tgt: st.session_state[f'opt_pred_{tgt.lower()}'] for tgt in target_vars}
                        st.session_state['ai_analysis_result'] = generate_ai_guidance(current_p_specs, current_kpis, mode="Optimization")
                
                if st.session_state.get('ai_analysis_result') is not None:
                   with st.expander("AI Report", expanded=True):
                    st.markdown(f"<div class='scrollable-box'>{st.session_state['ai_analysis_result']}</div>", unsafe_allow_html=True)
                    st.download_button(
                     label="📥 리포트 다운로드",
                      data=st.session_state['ai_analysis_result'],
                     file_name="AI_Report.txt",
                      mime="text/plain"
                     )
        with layout_r:
            if st.session_state['opt_result_x'] is not None:
                st.markdown(f"<div class='glass-card'><div class='glass-card-title' style='color:#3b82f6;'>{L_G['pred_title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='padding: 10px; background: #1e293b; border-left: 4px solid #10b981; margin-bottom: 15px; font-size: 0.88rem;'>🎯 <strong>{L_G['best_algo']}:</strong> <span style='color:#10b981; font-family:\"JetBrains Mono\"; font-weight:700;'>{st.session_state['best_algorithm_used']}</span></div>", unsafe_allow_html=True)
                # 추가된 부분 시작
                st.markdown("<div style='margin: -10px 0 15px 0; border-top: 1px solid #1e293b;'></div>", unsafe_allow_html=True)
                st.markdown("<span style='font-size:0.75rem; color:#94a3b8;'>Auto-selected Algorithms:</span>", unsafe_allow_html=True)
                algo_list = [f"{tgt}: {st.session_state.get('model_metadata', {}).get(f'algo_{tgt.lower()}', 'N/A')}" for tgt in target_vars]
                st.markdown(f"<div style='font-size:0.75rem; color:#38bdf8; margin-bottom: 15px;'>{' | '.join(algo_list)}</div>", unsafe_allow_html=True)
                # 추가된 부분 끝
                pred_dict = {tgt: [st.session_state[f'opt_pred_{tgt.lower()}']] for tgt in target_vars}
                df_pred_export = pd.DataFrame(pred_dict)
                
                col_pred_sel, col_pred_trigger = st.columns([1, 1])
                with col_pred_sel:
                    file_format_pred = st.selectbox(L_G['dl_format'], ["Excel (.xlsx)", "Database (.db)"], key="fmt_pred", label_visibility="collapsed")
                
                with col_pred_trigger:
                    if "Excel" in file_format_pred:
                        buffer_p = io.BytesIO()
                        with pd.ExcelWriter(buffer_p) as writer:
                            df_pred_export.to_excel(writer, index=False, sheet_name='Predicted_Performance')
                        st.download_button(label=L_G['dl_btn_pred'], data=buffer_p.getvalue(), file_name="predicted_performance.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_btn_pred_trigger")
                    else:
                        conn_p = sqlite3.connect(":memory:")
                        df_pred_export.to_sql("predicted_performance", conn_p, index=False, if_exists="replace")
                        backup_conn_p = sqlite3.connect("temp_pred.db")
                        conn_p.backup(backup_conn_p)
                        backup_conn_p.close()
                        conn_p.close()
                        with open("temp_pred.db", "rb") as f: db_bytes_p = f.read()
                        st.download_button(label=L_G['dl_btn_pred'], data=db_bytes_p, file_name="predicted_performance.db", mime="application/x-sqlite3", key="dl_btn_pred_db_trigger")
                
                st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
                
                cols_p_card = st.columns(3)
                for idx, tgt in enumerate(target_vars):
                    p_val = st.session_state[f'opt_pred_{tgt.lower()}']
                    val_display = f"{p_val:.3f}" if isinstance(p_val, float) else "0.000"
                    cols_p_card[idx % 3].markdown(f"<div style='padding:8px; background:#1e293b; border-radius:4px; margin-bottom:6px;'><span style='color:#94a3b8; font-size:0.72rem;'>Predicted {tgt}</span><br><strong style='font-size:1.05rem; color:#ffffff;'>{val_display}</strong></div>", unsafe_allow_html=True)
                
                st.metric(L_G['opt_conf'], f"{st.session_state['confidence_score']}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='glass-card'><div class='glass-card-title' style='color:#10b981;'>{L_G['rec_title']}</div>", unsafe_allow_html=True)
                ox = st.session_state['opt_result_x']
                df_export = pd.DataFrame([ox], columns=X_list)
                
                col_dl_sel, col_dl_trigger = st.columns([1, 1])
                with col_dl_sel:
                    file_format = st.selectbox(L_G['dl_format'], ["Excel (.xlsx)", "Database (.db)"], key="fmt_spec", label_visibility="collapsed")
                
                with col_dl_trigger:
                    if "Excel" in file_format:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer) as writer: df_export.to_excel(writer, index=False, sheet_name='Optimized_Specs')
                        st.download_button(label=L_G['dl_btn_spec'], data=buffer.getvalue(), file_name="recommended_process_spec.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_btn_spec_trigger")
                    else:
                        conn = sqlite3.connect(":memory:")
                        df_export.to_sql("recommended_spec", conn, index=False, if_exists="replace")
                        backup_conn = sqlite3.connect("temp_spec.db")
                        conn.backup(backup_conn)
                        backup_conn.close()
                        conn.close()
                        with open("temp_spec.db", "rb") as f: db_bytes = f.read()
                        st.download_button(label=L_G['dl_btn_spec'], data=db_bytes, file_name="recommended_process_spec.db", mime="application/x-sqlite3", key="dl_btn_spec_db_trigger")
                
                st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
                cols = st.columns(3)
                for idx, v_name in enumerate(X_list):
                    val_display = f"{ox[idx]:.3f}"
                    cols[idx % 3].markdown(f"<div style='padding:8px; background:#1e293b; border-radius:4px; margin-bottom:6px;'><span style='color:#94a3b8; font-size:0.72rem;'>{v_name}</span><br><strong style='font-size:1.05rem; color:#ffffff;'>{val_display}</strong></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        sim_l, sim_r = st.columns([1.3, 1.2], gap="large")
        
        with sim_l:
            st.markdown(f"<div class='glass-card'><div class='glass-card-title'>{L_G['live_input']}</div>", unsafe_allow_html=True)
            
            for idx, tgt in enumerate(valid_tgts):
                t_low = tgt.lower()
                max_slider_val = 20.0 if tgt in ['BT', 'RT'] else (0.3 if tgt == 'AGB' else 5.0)
                
                st.markdown(f"<p style='font-size:0.85rem; font-weight:600; color:#00e5ff; margin-bottom:5px;'>{idx+1}. Sim Target {tgt} Range</p>", unsafe_allow_html=True)
                col_sb1, col_sb2 = st.columns([1.8, 1.2])
                with col_sb1:
                    st.slider(f"Sim {tgt} Slider UI", 0.0, float(max_slider_val), step=0.001 if tgt=='AGB' else 0.05, label_visibility="collapsed", key=f"sim_tgt_{t_low}_s_val", on_change=on_sim_slider_change, args=(t_low,))
                with col_sb2:
                    sb_num_c1, sb_num_c2 = st.columns(2)
                    sb_num_c1.number_input("Sim Min", step=0.001 if tgt=='AGB' else 0.1, key=f"sim_tgt_{t_low}_n_min", on_change=on_sim_min_change, args=(t_low,))
                    sb_num_c2.number_input("Sim Max", step=0.001 if tgt=='AGB' else 0.1, key=f"sim_tgt_{t_low}_n_max", on_change=on_sim_max_change, args=(t_low,))
                st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)

            if undefined_tgts:
                with st.expander(f"Undefined Quality Targets (미정 항목 {len(undefined_tgts)}종 제어단)"):
                    for tgt in undefined_tgts:
                        t_low = tgt.lower()
                        st.markdown(f"<p style='font-size:0.8rem; margin:2px 0; color:#94a3b8;'>• {tgt} Range</p>", unsafe_allow_html=True)
                        cx1, cx2 = st.columns([1.8, 1.2])
                        with cx1:
                            st.slider(f"Sim {tgt} S", -0.5, 5.0, step=0.01, label_visibility="collapsed", key=f"sim_tgt_{t_low}_s_val", on_change=on_sim_slider_change, args=(t_low,))
                        with cx2:
                            sub_nc1, sub_nc2 = st.columns(2)
                            sub_nc1.number_input("Sim Min", step=0.01, key=f"sim_tgt_{t_low}_n_min", on_change=on_sim_min_change, args=(t_low,))
                            sub_nc2.number_input("Sim Max", step=0.01, key=f"sim_tgt_{t_low}_n_max", on_change=on_sim_max_change, args=(t_low,))

            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button(L_G['run_sim'], type="secondary"):
                def sim_target_loss(x):
                    df_x = pd.DataFrame([x], columns=X_list)
                    q = st.session_state['scaler'].transform(df_x)
                    
                    total_loss = 0.0
                    for tgt in target_vars:
                        model_key = f'model_{tgt.lower()}'
                        if st.session_state[model_key] is not None:
                            pred = st.session_state[model_key].predict(q)[0]
                            t_range = st.session_state[f'sim_tgt_{tgt.lower()}_s_val']
                            total_loss += max(0, t_range[0] - pred)**2 + max(0, pred - t_range[1])**2
                    return total_loss
                
                init_x = [(db[v][0] + db[v][1]) / 2 for v in X_list]
                bands = [db[v] for v in X_list]
                
                with st.spinner("🔍 가상 역최적화 파라미터 탐색 중..."):
                    res_sim = minimize(sim_target_loss, init_x, method='SLSQP', bounds=bands)
                    final_sim_x = np.clip(res_sim.x, [b[0] for b in bands], [b[1] for b in bands])
                    sim_loss_val = sim_target_loss(final_sim_x)
                
                q_sim_opt = st.session_state['scaler'].transform(pd.DataFrame([final_sim_x], columns=X_list))
                
                update_sim_dict = {
                    'sim_result_x': final_sim_x,
                    'sim_confidence': round(max(50.0, 100.0 - (sim_loss_val * 10)), 1),
                    'ai_analysis_result': None
                }
                for tgt in target_vars:
                    model_key = f'model_{tgt.lower()}'
                    if st.session_state[model_key] is not None:
                        update_sim_dict[f'sim_pred_{tgt.lower()}'] = float(st.session_state[model_key].predict(q_sim_opt)[0])
                    else:
                        update_sim_dict[f'sim_pred_{tgt.lower()}'] = 0.0
                        
                st.session_state.update(update_sim_dict)
                st.rerun()

            if st.session_state['sim_result_x'] is not None:
                st.markdown(f"<div class='glass-card'><div class='glass-card-title' style='color:#a3e635;'>{L_G['ai_title']}</div>", unsafe_allow_html=True)
                if st.button(L_G['ai_btn'], key="ai_btn_tab2"):
                    with st.spinner(L_G['ai_loading']):
                        current_p_specs = {X_list[i]: st.session_state['sim_result_x'][i] for i in range(len(X_list))}
                        current_kpis = {tgt: st.session_state[f'sim_pred_{tgt.lower()}'] for tgt in target_vars}
                        st.session_state['ai_analysis_result'] = generate_ai_guidance(current_p_specs, current_kpis, mode="Simulation")
                
                if st.session_state['ai_analysis_result'] is not None:
                    with st.expander("AI Report", expanded=True):
                        st.markdown(st.session_state['ai_analysis_result'])
                st.markdown("</div>", unsafe_allow_html=True)

        with sim_r:
            if st.session_state['sim_result_x'] is not None:
                st.markdown(f"<div class='glass-card'><div class='glass-card-title' style='color:#38bdf8;'>{L_G['sim_pred_title']}</div>", unsafe_allow_html=True)
                
                cols_s_card = st.columns(3)
                for idx, tgt in enumerate(target_vars):
                    s_val = st.session_state[f'sim_pred_{tgt.lower()}']
                    val_display = f"{s_val:.3f}" if isinstance(s_val, float) else "0.000"
                    cols_s_card[idx % 3].markdown(f"<div style='padding:8px; background:#1e293b; border-radius:4px; margin-bottom:6px;'><span style='color:#94a3b8; font-size:0.72rem;'>Est. {tgt}</span><br><strong style='font-size:1.05rem; color:#38bdf8;'>{val_display}</strong></div>", unsafe_allow_html=True)
                
                st.metric("Prediction Safe Index", f"{st.session_state['sim_confidence']}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='glass-card'><div class='glass-card-title' style='color:#10b981;'>{L_G['sim_title']}</div>", unsafe_allow_html=True)
                sx = st.session_state['sim_result_x']
                
                cols_s = st.columns(3)
                for idx, v_name in enumerate(X_list):
                    val_display = f"{sx[idx]:.3f}"
                    cols_s[idx % 3].markdown(f"<div style='padding:8px; background:#1e293b; border-radius:4px; margin-bottom:6px;'><span style='color:#94a3b8; font-size:0.72rem;'>{v_name}</span><br><strong style='font-size:1.05rem; color:#ffffff;'>{val_display}</strong></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.dataframe(st.session_state['df_caulking'], use_container_width=True)

else:
    st.info(L_G['engine_inactive'])
