"""VulcanTech Smart Metals — Painel Executivo de IA de Borda, MLOps e CVOps.

Sistema integrado de monitoramento preditivo, cálculo dinâmico de OEE,
atuação em malha fechada e inspeção óptica/termográfica por visão computacional
para as células robotizadas CEL-201 a CEL-204 da VulcanTech Smart Metals S.A.

Ligas Processadas: Aço Carbono 316L, Alumínio 7075 e Titânio Ti-6Al-4V.
Responsável Técnico: André Matteussi
Curso: Cibersistemas para Automação — CentroWEG / SENAI
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Configuração de PYTHONPATH para importação direta de módulos internos
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ml.oee_calculator import IndustrialOEECalculator
from src.vision.thermal_haz_inspector import ThermalHAZInspector
from src.vision.weld_inspector import WeldGeometricInspector

# Configuração visual padrão do Matplotlib
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# Mapeamentos e constantes do domínio industrial VulcanTech
MAPA_DIAGNOSTICO: Dict[int, str] = {
    0: "Classe 0 — Operação Normal (Nominal)",
    1: "Classe 1 — Desgaste Severo de Ferramenta / Fuso",
    2: "Classe 2 — Obstrução no Sistema de Refrigeração",
    3: "Classe 3 — Desalinhamento da Matriz de Prensa",
}

MEDIAS_HISTORICAS_LIGA: Dict[str, Dict[str, float]] = {
    "Aco_Carbono_316L": {
        "torque_fuso_nm": 117.44,
        "corrente_estator_amp": 30.10,
        "vibracao_triaxial_rms_mms": 2.22,
        "temperatura_fuso_c": 55.05,
        "indice_esforco_mecanico": 11.95,
    },
    "Aluminio_7075": {
        "torque_fuso_nm": 93.32,
        "corrente_estator_amp": 23.85,
        "vibracao_triaxial_rms_mms": 1.75,
        "temperatura_fuso_c": 43.99,
        "indice_esforco_mecanico": 7.29,
    },
    "Liga_Titanio_Ti6Al4V": {
        "torque_fuso_nm": 150.89,
        "corrente_estator_amp": 38.70,
        "vibracao_triaxial_rms_mms": 2.75,
        "temperatura_fuso_c": 70.88,
        "indice_esforco_mecanico": 19.01,
    },
}

MODULOS_PROJETO: Dict[str, Dict[str, str]] = {
    "Visão Geral & Planta": {
        "modulo_py": "src/dashboard/app.py",
        "etapa": "Visão Executiva & Células Industriais",
        "icone": "bi-grid-1x2-fill",
        "tags": "visao geral celulas vulcantech kpis distribuicao ligas oee",
    },
    "Saneamento & Engenharia (Etapa 1)": {
        "modulo_py": "src/data_pipeline/preprocessor.py & feature_engineering.py",
        "etapa": "Etapa 1 — Saneamento, IEM & Séries Temporais",
        "icone": "bi-sliders",
        "tags": "saneamento eda iem esforco mecanico normalizacao ligas rolling",
    },
    "Modelagem Preditiva & MLOps (Etapa 2)": {
        "modulo_py": "src/ml/train.py & evaluation.py",
        "etapa": "Etapa 2 — Random Forest, SMOTE & Isolation Forest",
        "icone": "bi-cpu-fill",
        "tags": "mlops random forest smote matriz confusao recall feature importance",
    },
    "Simulador Edge em Malha Fechada (Etapa 4)": {
        "modulo_py": "src/api/app.py & schemas.py",
        "etapa": "Etapa 4 — Inferência Edge, OEE & CLP",
        "icone": "bi-broadcast-pin",
        "tags": "simulador borda tempo real oee malha fechada clp pydantic",
    },
    "Visão Computacional CVOps (Etapa 3)": {
        "modulo_py": "src/vision/weld_inspector.py & thermal_haz_inspector.py",
        "etapa": "Etapa 3 — Canny Solda MIG/MAG & Termografia HSV",
        "icone": "bi-camera-video-fill",
        "tags": "visao computacional canny poros solda termografia hsv haz calor",
    },
    "Governança, ESG & Documentação (Etapa 5)": {
        "modulo_py": "docs/*.pdf & README.md",
        "etapa": "Etapa 5 — Parecer Regulatório, Drift & ESG",
        "icone": "bi-journal-code",
        "tags": "governanca parecer eu ai act lgpd purdue drift esg titanio",
    },
}


# =============================================================================
# 1. ESTILIZAÇÃO CSS INDUSTRIAL CORPORATIVA VULCANTECH
# =============================================================================
def aplicar_estilo_industrial() -> None:
    """Aplica folha de estilos CSS industrial com paleta corporativa VulcanTech."""
    css_bootstrap = (
        '<link rel="stylesheet" '
        'href="https://cdn.jsdelivr.net/npm/'
        'bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">'
    )
    st.markdown(css_bootstrap, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
            :root {
                --primary-color: #0A192F !important;
                --accent-color: #0090D4 !important;
                accent-color: #0090D4 !important;
            }
            input[type="radio"] {
                accent-color: #0090D4 !important;
            }
            .stApp {
                background-color: #F8FAFC;
                color: #0F172A;
            }

            /* Botões de Ação Industriais */
            div.stButton > button {
                background: linear-gradient(
                    135deg, #0A192F 0%, #1E3A8A 100%
                ) !important;
                color: #FFFFFF !important;
                border: 1px solid #0F172A !important;
                border-bottom: 3px solid #0090D4 !important;
                border-radius: 6px !important;
                padding: 0.55rem 1rem !important;
                font-weight: 600 !important;
                font-size: 0.86rem !important;
                letter-spacing: 0.3px !important;
                transition: all 0.2s ease-in-out !important;
                box-shadow: 0 2px 4px rgba(10, 25, 47, 0.12) !important;
            }
            div.stButton > button:hover {
                background: linear-gradient(
                    135deg, #1E3A8A 0%, #0284C7 100%
                ) !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 4px 8px rgba(0, 144, 212, 0.25) !important;
            }

            /* Sidebar Radio Buttons em Blocos Corporativos */
            section[data-testid="stSidebar"] div[role="radiogroup"] {
                gap: 6px !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label {
                background-color: #FFFFFF !important;
                border: 1px solid #CBD5E1 !important;
                border-left: 4px solid #94A3B8 !important;
                border-radius: 6px !important;
                padding: 10px 14px !important;
                margin-bottom: 4px !important;
                transition: all 0.18s ease !important;
                cursor: pointer !important;
                width: 100% !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
                border-left: 4px solid #0090D4 !important;
                background-color: #F0F9FF !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
                background-color: #0A192F !important;
                border-color: #0A192F !important;
                border-left: 4px solid #00D2FF !important;
                color: #FFFFFF !important;
                font-weight: 600 !important;
                box-shadow: 0 2px 6px rgba(10, 25, 47, 0.22) !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
                color: #FFFFFF !important;
                font-weight: 600 !important;
            }

            /* Topbar Executiva VulcanTech */
            .vulcantech-header {
                background: linear-gradient(
                    90deg, #0A192F 0%, #1E293B 60%, #0369A1 100%
                );
                color: #FFFFFF;
                padding: 16px 24px;
                border-radius: 8px;
                border-bottom: 3px solid #00D2FF;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 12px;
                margin-bottom: 20px;
                box-shadow: 0 4px 10px rgba(10, 25, 47, 0.15);
            }
            .vulcantech-title {
                font-size: 1.18rem;
                font-weight: 700;
                letter-spacing: 0.6px;
                color: #FFFFFF;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .vulcantech-badge {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 0.80rem;
                color: #E0F2FE;
                background-color: rgba(0, 144, 212, 0.25);
                border: 1px solid rgba(0, 210, 255, 0.4);
                padding: 4px 10px;
                border-radius: 4px;
            }
            .vulcantech-status-bar {
                font-family: 'Consolas', monospace;
                font-size: 0.80rem;
                color: #CBD5E1;
                display: flex;
                gap: 16px;
                align-items: center;
                flex-wrap: wrap;
            }

            /* Cartões e Painéis Estruturais */
            .corp-card {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-top: 3px solid #0090D4;
                border-radius: 8px;
                padding: 18px 20px;
                margin-bottom: 16px;
                box-shadow: 0 2px 4px rgba(15, 23, 42, 0.04);
            }
            .corp-card-title {
                font-size: 0.95rem;
                font-weight: 700;
                color: #0A192F;
                margin-bottom: 12px;
                padding-bottom: 8px;
                border-bottom: 1px solid #F1F5F9;
                display: flex;
                align-items: center;
                gap: 8px;
                text-transform: uppercase;
                letter-spacing: 0.4px;
            }
            .corp-kpi-box {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-left: 4px solid #0090D4;
                border-radius: 6px;
                padding: 14px 16px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            }
            .corp-kpi-label {
                font-size: 0.74rem;
                font-weight: 700;
                color: #64748B;
                text-transform: uppercase;
                letter-spacing: 0.4px;
            }
            .corp-kpi-value {
                font-size: 1.48rem;
                font-weight: 700;
                color: #0A192F;
                margin-top: 4px;
            }
            .corp-kpi-sub {
                font-size: 0.74rem;
                color: #0090D4;
                margin-top: 3px;
                font-family: 'Consolas', monospace;
            }

            /* Teoria e Fundamentação */
            .theory-box {
                background-color: #F0F9FF;
                border: 1px solid #BAE6FD;
                border-left: 4px solid #0284C7;
                border-radius: 6px;
                padding: 14px 18px;
                margin-bottom: 16px;
                color: #0C4A6E;
                font-size: 0.90rem;
                line-height: 1.6;
            }
            .math-pill {
                font-family: 'Cambria Math', serif;
                font-weight: 700;
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                padding: 2px 8px;
                border-radius: 4px;
                color: #0A192F;
            }

            /* Badges de Status Operacional */
            .status-badge-ok {
                background-color: #DEF7EC;
                color: #03543F;
                border: 1px solid #84E1BC;
                border-left: 4px solid #0E9F6E;
                padding: 8px 14px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 0.88rem;
                display: block;
            }
            .status-badge-warn {
                background-color: #FEF08A;
                color: #713F12;
                border: 1px solid #FDE047;
                border-left: 4px solid #EAB308;
                padding: 8px 14px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 0.88rem;
                display: block;
            }
            .status-badge-crit {
                background-color: #FEE2E2;
                color: #991B1B;
                border: 1px solid #FCA5A5;
                border-left: 4px solid #DC2626;
                padding: 8px 14px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 0.88rem;
                display: block;
            }

            /* Rodapé VulcanTech */
            .vulcantech-footer {
                margin-top: 36px;
                background-color: #0A192F;
                color: #94A3B8;
                padding: 22px 26px;
                border-radius: 8px;
                border-top: 3px solid #00D2FF;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 20px;
                font-size: 0.82rem;
                line-height: 1.6;
            }
            .footer-col-title {
                color: #F8FAFC;
                font-size: 0.86rem;
                font-weight: 700;
                text-transform: uppercase;
                margin-bottom: 8px;
                letter-spacing: 0.5px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def renderizar_kpi(coluna, titulo: str, valor: str, subtitulo: str, icone: str) -> None:
    """Renderiza um cartão métrico padronizado com ícone lateral."""
    with coluna:
        st.markdown(
            f"""
            <div class="corp-kpi-box">
                <div class="corp-kpi-label"><i class="bi {icone}"></i> {titulo}</div>
                <div class="corp-kpi-value">{valor}</div>
                <div class="corp-kpi-sub">{subtitulo}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def renderizar_cabecalho_sistema(meta_modulo: Dict[str, str]) -> None:
    """Renderiza o cabeçalho executivo superior do sistema."""
    horario_local = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    mod_str = meta_modulo["modulo_py"]
    st.markdown(
        f"""
        <div class="vulcantech-header">
            <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 12px;">
                <span class="vulcantech-title">
                    <i class="bi bi-shield-check" style="color: #00D2FF;"></i>
                    VULCANTECH PREDICTIVE & EDGE AI COCKPIT
                </span>
                <span class="vulcantech-badge">
                    <i class="bi bi-file-earmark-code" style="margin-right: 4px;"></i>
                    Módulo: {mod_str}
                </span>
            </div>
            <div class="vulcantech-status-bar">
                <span><i class="bi bi-buildings" style="color: #00D2FF;"></i> VulcanTech</span>
                <span><i class="bi bi-hdd-network" style="color: #10B981;"></i> CEL-201..204</span>
                <span><i class="bi bi-clock" style="color: #94A3B8;"></i> {horario_local}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def renderizar_rodape_sistema() -> None:
    """Renderiza o rodapé institucional em 3 colunas simétricas."""
    st.markdown(
        """
        <div class="vulcantech-footer">
            <div>
                <div class="footer-col-title">
                    <i class="bi bi-hdd-rack"></i> ECOSSISTEMA INDUSTRIAL
                </div>
                <strong>VulcanTech Predictive & Edge AI System (v1.0.0-edge)</strong><br>
                <span>VulcanTech Smart Metals S.A. — Manufatura Avançada</span><br>
                <span>Monitoramento das Células Robóticas CEL-201 a CEL-204</span>
            </div>
            <div>
                <div class="footer-col-title">
                    <i class="bi bi-shield-lock"></i> GOVERNANÇA & CIBERSEGURANÇA
                </div>
                <span><strong>Segurança:</strong> Isolamento DMZ e Purdue Nível 2/3</span><br>
                <span><strong>Regulatório:</strong> EU AI Act (Alto Risco) & LGPD</span><br>
                <span><strong>MLOps Drift:</strong> Testes KS bicaudais e métrica PSI</span>
            </div>
            <div>
                <div class="footer-col-title">
                    <i class="bi bi-mortarboard"></i> ENGENHARIA & AUTORIA
                </div>
                <span><strong>Responsável Técnico:</strong> André Matteussi</span><br>
                <span><strong>Curso:</strong> Cibersistemas para Automação — CentroWEG</span><br>
                <span><strong>Orientação:</strong> Prof. Fabiano da Silva Luiz</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 2. CARREGAMENTO EM CACHE DE ARTEFATOS E DADOS
# =============================================================================
@st.cache_data
def carregar_dados() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega dados brutos e dataset processado com tratamento de exceções."""
    caminho_bruto = ROOT_DIR / "data" / "raw" / "dados_sensores_vulcantech.csv"
    caminho_proc = ROOT_DIR / "data" / "processed" / "features_eng.parquet"

    df_raw = pd.read_csv(caminho_bruto) if caminho_bruto.exists() else pd.DataFrame()
    df_proc = pd.read_parquet(caminho_proc) if caminho_proc.exists() else pd.DataFrame()
    return df_raw, df_proc


@st.cache_resource
def carregar_artefatos_ml() -> Dict[str, Any]:
    """Carrega classificadores, escaladores e modelo de anomalia serializados."""
    dir_mod = ROOT_DIR / "models"
    artefatos: Dict[str, Any] = {}
    try:
        artefatos["classifier"] = joblib.load(dir_mod / "classifier_model.joblib")
        artefatos["iso_forest"] = joblib.load(dir_mod / "isolation_forest.joblib")
        artefatos["scaler"] = joblib.load(dir_mod / "scaler.joblib")
        artefatos["features"] = joblib.load(dir_mod / "feature_names.joblib")
        artefatos["oee_calc"] = IndustrialOEECalculator()
    except Exception as exc:
        st.error(f"Erro ao carregar modelos serializados em {dir_mod}: {exc}")
    return artefatos


def alinhar_features_para_modelo(
    df: pd.DataFrame, feature_names: List[str]
) -> pd.DataFrame:
    """Garante que todas as features esperadas pelo modelo estejam presentes."""
    df_copy = df.copy()
    for col in feature_names:
        if col not in df_copy.columns:
            df_copy[col] = 0.0
    return df_copy[feature_names]


# =============================================================================
# 3. MÓDULOS OPERACIONAIS (TELAS / ABAS)
# =============================================================================
def tela_visao_geral(
    df_raw: pd.DataFrame, df_proc: pd.DataFrame, artefatos: Dict[str, Any]
) -> None:
    """Tela 01: Visão Executiva Geral, Saúde das Células e Arquitetura."""
    taxa_falhas = (
        (df_proc["tipo_falha"] != 0).mean() * 100
        if not df_proc.empty
        else 0.0
    )

    k1, k2, k3, k4 = st.columns(4)
    renderizar_kpi(
        k1,
        "Células Instrumentadas",
        "04 Células",
        "CEL-201 a CEL-204 (Ciclo 10 min)",
        "bi-hdd-rack-fill",
    )
    renderizar_kpi(
        k2,
        "Telemetria Saneada",
        f"{len(df_proc):,}".replace(",", "."),
        f"Base Bruta: {len(df_raw):,} registros",
        "bi-database-fill-check",
    )
    renderizar_kpi(
        k3,
        "Features de Engenharia",
        f"{len(artefatos.get('features', []))} Atributos",
        "IEM + Normalização por Liga + Rolling",
        "bi-layers-fill",
    )
    renderizar_kpi(
        k4,
        "Taxa de Falhas Bruta",
        f"{taxa_falhas:.1f}%",
        "Balanceamento de Treino via SMOTE",
        "bi-exclamation-octagon-fill",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_esq, col_dir = st.columns([1.5, 1.0])
    with col_esq:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-activity"></i>
                    DIAGNÓSTICO CONSOLIDADO POR CÉLULA INDUSTRIAL
                </div>
            """,
            unsafe_allow_html=True,
        )
        if not df_proc.empty and "id_celula" in df_proc.columns:
            linhas = []
            for celula, grp in df_proc.groupby("id_celula"):
                ult_status = int(grp.iloc[-1]["tipo_falha"])
                falhas_qtd = int((grp["tipo_falha"] != 0).sum())
                desc_status = MAPA_DIAGNOSTICO.get(
                    ult_status, f"Classe {ult_status}"
                )
                linhas.append({
                    "Célula": celula,
                    "Total Leituras": len(grp),
                    "Temp. Fuso (°C)": round(
                        grp["temperatura_fuso_c"].mean(), 2
                    ),
                    "Vibração (mm/s)": round(
                        grp["vibracao_triaxial_rms_mms"].mean(), 2
                    ),
                    "Torque Fuso (Nm)": round(
                        grp["torque_fuso_nm"].mean(), 2
                    ),
                    "Vazão Refrig. (L/min)": round(
                        grp["vazao_refrigerante_lmin"].mean(), 2
                    ),
                    "Total Anomalias": falhas_qtd,
                    "Último Status": desc_status,
                })
            st.dataframe(
                pd.DataFrame(linhas),
                width='stretch',
                hide_index=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_dir:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-folder2-open"></i> ESTRUTURA MODULAR DO PROJETO
                </div>
                <ul style="font-size: 0.86rem; line-height: 1.65; padding-left: 18px; margin-bottom: 0;">
                    <li><code>src/data_pipeline/</code>: Saneamento físico EMI e extração de IEM.</li>
                    <li><code>src/ml/</code>: Random Forest multiclasse, SMOTE e Isolation Forest.</li>
                    <li><code>src/vision/</code>: Inspeção de solda MIG/MAG e termografia HSV.</li>
                    <li><code>src/api/</code>: Microsserviço RESTful FastAPI em malha fechada.</li>
                    <li><code>docs/</code>: Parecer regulatório, plano de Drift e laudo técnico.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Gráficos de Distribuição por Liga Metálica e Modos de Falha
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-pie-chart-fill"></i>
                    DISTRIBUIÇÃO DE PRODUÇÃO POR LIGA METÁLICA
                </div>
            """,
            unsafe_allow_html=True,
        )
        if not df_proc.empty and "liga_metalica" in df_proc.columns:
            contagem_ligas = df_proc["liga_metalica"].value_counts()
            fig_p, ax_p = plt.subplots(figsize=(6, 3.8))
            cores = ["#00579D", "#0090D4", "#FF6B35"]
            ax_p.pie(
                contagem_ligas.values,
                labels=contagem_ligas.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=cores,
                wedgeprops={"edgecolor": "#FFFFFF", "linewidth": 1.5},
            )
            ax_p.axis("equal")
            st.pyplot(fig_p)
            plt.close(fig_p)
        st.markdown("</div>", unsafe_allow_html=True)

    with g2:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-bar-chart-steps"></i>
                    TAXA DE FALHAS POR CLASSE OPERACIONAL
                </div>
            """,
            unsafe_allow_html=True,
        )
        if not df_proc.empty and "tipo_falha" in df_proc.columns:
            contagem_falhas = df_proc["tipo_falha"].value_counts().sort_index()
            fig_b, ax_b = plt.subplots(figsize=(6, 3.8))
            labels_falhas = [f"Classe {c}" for c in contagem_falhas.index]
            barras = ax_b.bar(
                labels_falhas,
                contagem_falhas.values,
                color=["#10B981", "#F59E0B", "#EF4444", "#8B5CF6"],
                edgecolor="#CBD5E1",
            )
            ax_b.set_ylabel("Quantidade de Leituras")
            ax_b.spines["top"].set_visible(False)
            ax_b.spines["right"].set_visible(False)
            for b in barras:
                y = b.get_height()
                ax_b.annotate(
                    f"{y:,}".replace(",", "."),
                    (b.get_x() + b.get_width() / 2, y + 100),
                    ha="center",
                    fontsize=8.5,
                )
            st.pyplot(fig_b)
            plt.close(fig_b)
        st.markdown("</div>", unsafe_allow_html=True)


def tela_engenharia_dados(df_raw: pd.DataFrame, df_proc: pd.DataFrame) -> None:
    """Tela 02: Etapa 1 — Saneamento Físico, Cálculo do IEM e EDA."""
    st.markdown(
        """
        <div class="theory-box">
            <strong><i class="bi bi-journal-bookmark-fill"></i>
            FUNDAMENTAÇÃO DE ENGENHARIA DE DADOS (<code>src/data_pipeline/</code>)</strong><br>
            Leituras brutas apresentam ruídos de Interferência Eletromagnética (EMI),
            como temperaturas espúrias de <span class="math-pill">999.9 °C</span>.
            O módulo <code>preprocessor.py</code> valida limites físicos e aplica
            interpolação temporal por célula. Em seguida, <code>feature_engineering.py</code>
            deriva o <strong>Índice de Esforço Mecânico (IEM)</strong> e calcula médias e
            desvios móveis em janelas de 30 min (6 períodos) e 2 horas (24 períodos).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"\text{IEM} = \frac{\text{Torque Fuso (Nm)} \times "
        r"\text{Vibração RMS (mm/s)}}{\text{Vazão Refrigerante (L/min)} + 1.0}"
    )

    c1, c2, c3 = st.columns(3)
    celulas = (
        sorted(df_proc["id_celula"].unique())
        if not df_proc.empty
        else ["CEL-201"]
    )
    cel_sel = c1.selectbox("Selecionar Célula Industrial:", celulas)
    var_opcoes = [
        "temperatura_fuso_c",
        "vibracao_triaxial_rms_mms",
        "torque_fuso_nm",
        "vazao_refrigerante_lmin",
        "indice_esforco_mecanico",
    ]
    var_sel = c2.selectbox("Sensor / Feature Analisada:", var_opcoes)
    janela = c3.slider("Janela Cronológica (passos de 10 min):", 60, 600, 200, 20)

    df_cel = (
        df_proc[df_proc["id_celula"] == cel_sel].head(janela)
        if not df_proc.empty
        else pd.DataFrame()
    )

    g_esq, g_dir = st.columns(2)
    with g_esq:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-graph-up"></i>
                    SÉRIE TEMPORAL SANEADA & MÉDIAS MÓVEIS (30 MIN / 2 H)
                </div>
            """,
            unsafe_allow_html=True,
        )
        if not df_cel.empty and var_sel in df_cel.columns:
            fig_t, ax_t = plt.subplots(figsize=(8, 4.2))
            ax_t.plot(
                df_cel.index,
                df_cel[var_sel],
                label="Leitura Saneada",
                color="#94A3B8",
                linewidth=1.2,
            )
            col_30 = f"{var_sel}_rolling_mean_30min"
            col_2h = f"{var_sel}_rolling_mean_2h"
            if col_30 in df_cel.columns:
                ax_t.plot(
                    df_cel.index,
                    df_cel[col_30],
                    label="Média Móvel 30 min (6p)",
                    color="#0090D4",
                    linewidth=2.0,
                )
            if col_2h in df_cel.columns:
                ax_t.plot(
                    df_cel.index,
                    df_cel[col_2h],
                    label="Média Móvel 2 h (24p)",
                    color="#0A192F",
                    linewidth=2.2,
                )
            ax_t.set_title(
                f"Série Temporal — {cel_sel} ({var_sel})",
                fontsize=10,
                fontweight="bold",
                color="#0A192F",
            )
            ax_t.spines["top"].set_visible(False)
            ax_t.spines["right"].set_visible(False)
            ax_t.legend(frameon=False, fontsize=8.5)
            ax_t.grid(True, linestyle="--", alpha=0.35)
            st.pyplot(fig_t)
            plt.close(fig_t)
        st.markdown("</div>", unsafe_allow_html=True)

    with g_dir:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-grid-3x3-gap-fill"></i>
                    MATRIZ DE CORRELAÇÃO LINEAR DE PEARSON
                </div>
            """,
            unsafe_allow_html=True,
        )
        if not df_proc.empty:
            cols_corr = [
                c
                for c in [
                    "temperatura_fuso_c",
                    "vibracao_triaxial_rms_mms",
                    "torque_fuso_nm",
                    "vazao_refrigerante_lmin",
                    "corrente_estator_amp",
                    "pressao_hidraulica_bar",
                    "nivel_ruido_db",
                    "indice_esforco_mecanico",
                    "tipo_falha",
                ]
                if c in df_proc.columns
            ]
            fig_c, ax_c = plt.subplots(figsize=(8, 4.2))
            sns.heatmap(
                df_proc[cols_corr].corr(),
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                ax=ax_c,
                cbar=False,
                annot_kws={"size": 7.5},
            )
            ax_c.set_title(
                "Correlação Multivariada dos Sensores",
                fontsize=10,
                fontweight="bold",
                color="#0A192F",
            )
            st.pyplot(fig_c)
            plt.close(fig_c)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="corp-card">
            <div class="corp-card-title">
                <i class="bi bi-table"></i>
                AMOSTRA DO DATASET PROCESSADO (data/processed/features_eng.parquet)
            </div>
        """,
        unsafe_allow_html=True,
    )
    if not df_cel.empty:
        st.dataframe(df_cel.head(10), width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)


def tela_modelagem_ml(df_proc: pd.DataFrame, artefatos: Dict[str, Any]) -> None:
    """Tela 03: Etapa 2 — Avaliação de Machine Learning, Matriz de Confusão e SHAP."""
    st.markdown(
        """
        <div class="theory-box">
            <strong><i class="bi bi-journal-bookmark-fill"></i>
            FUNDAMENTAÇÃO DE MLOPS (<code>src/ml/train.py &amp; evaluation.py</code>)</strong><br>
            A modelagem multiclasse utiliza divisão cronológica 70/30 para mitigar
            vazamento temporal. A sobreamostragem <strong>SMOTE</strong> é restrita ao
            treinamento. O classificador <code>RandomForestClassifier</code> atinge
            <strong>Acurácia de 99.73%</strong> e <strong>Recall &gt; 97%</strong> em todas
            as falhas críticas, operando conjuntamente com o detector <code>Isolation Forest</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    renderizar_kpi(
        k1,
        "Acurácia Global Teste",
        "99.73%",
        "Classificador Random Forest",
        "bi-check2-circle",
    )
    renderizar_kpi(
        k2,
        "Recall Classe 1",
        "97.87%",
        "Desgaste Severo de Ferramenta",
        "bi-tools",
    )
    renderizar_kpi(
        k3,
        "Recall Classe 2",
        "100.0%",
        "Obstrução de Arrefecimento",
        "bi-snow2",
    )
    renderizar_kpi(
        k4,
        "Recall Classe 3",
        "97.12%",
        "Desalinhamento da Prensa",
        "bi-exclamation-triangle-fill",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c_conf, c_imp = st.columns([1.1, 1.2])
    with c_conf:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-grid-3x3"></i>
                    MATRIZ DE CONFUSÃO MULTICLASSE (70/30 TESTE)
                </div>
            """,
            unsafe_allow_html=True,
        )
        if not df_proc.empty and "classifier" in artefatos:
            features = artefatos["features"]
            X = df_proc[features]
            y = df_proc["tipo_falha"]
            split_idx = int(len(df_proc) * 0.7)
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]
            X_test_scaled = artefatos["scaler"].transform(X_test)
            X_test_scaled_df = pd.DataFrame(
                X_test_scaled, columns=artefatos["features"]
            )
            y_pred = artefatos["classifier"].predict(X_test_scaled_df)
            cm = confusion_matrix(y_test, y_pred)

            fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax_cm,
                cbar=False,
                xticklabels=["Classe 0", "Classe 1", "Classe 2", "Classe 3"],
                yticklabels=["Classe 0", "Classe 1", "Classe 2", "Classe 3"],
                annot_kws={"size": 10, "weight": "bold"},
            )
            ax_cm.set_xlabel("Classe Predita", fontweight="bold")
            ax_cm.set_ylabel("Classe Real", fontweight="bold")
            st.pyplot(fig_cm)
            plt.close(fig_cm)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_imp:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-bar-chart-line-fill"></i>
                    IMPORTÂNCIA DOS ATRIBUTOS (RANDOM FOREST)
                </div>
            """,
            unsafe_allow_html=True,
        )
        if (
            "classifier" in artefatos
            and hasattr(artefatos["classifier"], "feature_importances_")
        ):
            importancias = artefatos["classifier"].feature_importances_
            nomes = artefatos["features"]
            df_imp = (
                pd.DataFrame({"Feature": nomes, "Importância": importancias})
                .sort_values("Importância", ascending=True)
                .tail(12)
            )

            fig_imp, ax_imp = plt.subplots(figsize=(7, 4.5))
            ax_imp.barh(
                df_imp["Feature"],
                df_imp["Importância"],
                color="#0090D4",
                edgecolor="#0A192F",
                height=0.65,
            )
            ax_imp.set_xlabel("Gini Importance")
            ax_imp.spines["top"].set_visible(False)
            ax_imp.spines["right"].set_visible(False)
            ax_imp.grid(axis="x", linestyle="--", alpha=0.35)
            st.pyplot(fig_imp)
            plt.close(fig_imp)
        st.markdown("</div>", unsafe_allow_html=True)


def tela_simulador_edge(artefatos: Dict[str, Any]) -> None:
    """Tela 04: Etapa 4 — Simulador de Borda, Cálculo de OEE e Malha Fechada."""
    st.markdown(
        """
        <div class="theory-box">
            <strong><i class="bi bi-journal-bookmark-fill"></i>
            FUNDAMENTAÇÃO DE MALHA FECHADA & OEE
            (<code>src/api/app.py &amp; oee_calculator.py</code>)</strong><br>
            O simulador opera como um nó de borda industrial (Edge Computing). As leituras
            dos 7 sensores são validadas contra limites físicos, normalizadas pela liga e
            inferidas em tempo real. O modelo estima as penalidades de OEE e emite comandos
            imediatos ao CLP da célula: <code>MANTER_OPERACAO</code>,
            <code>REDUZIR_VELOCIDADE_50%</code> ou <code>INTERROMPER_EMERGENCIA</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "preset_vulcantech" not in st.session_state:
        st.session_state.preset_vulcantech = "Normal"

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Simular: Operação Nominal", width='stretch'):
        st.session_state.preset_vulcantech = "Normal"
    if b2.button("Simular: Desgaste de Ferramenta", width='stretch'):
        st.session_state.preset_vulcantech = "Ferramenta"
    if b3.button("Simular: Falha Arrefecimento", width='stretch'):
        st.session_state.preset_vulcantech = "Arrefecimento"
    if b4.button("Simular: Desalinhamento Prensa", width='stretch'):
        st.session_state.preset_vulcantech = "Prensa"

    presets = {
        "Normal": (55.0, 2.0, 115.0, 10.0, 30.0, 4.0, 65.0),
        "Ferramenta": (68.0, 4.8, 165.0, 9.5, 42.0, 3.8, 78.0),
        "Arrefecimento": (88.0, 2.5, 120.0, 2.2, 32.0, 4.1, 72.0),
        "Prensa": (60.0, 5.5, 185.0, 8.0, 45.0, 2.0, 85.0),
    }
    p = presets[st.session_state.preset_vulcantech]

    c_s1, c_s2, c_diag = st.columns([1, 1, 1.35])
    with c_s1:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-sliders"></i> SENSORES TERMOMECÂNICOS
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.selectbox(
            "Célula Industrial:",
            ["CEL-201", "CEL-202", "CEL-203", "CEL-204"],
        )
        liga_sel = st.selectbox(
            "Liga Metálica:",
            ["Aco_Carbono_316L", "Aluminio_7075", "Liga_Titanio_Ti6Al4V"],
        )
        temp_val = st.slider("Temperatura do Fuso (°C):", 20.0, 120.0, p[0], 0.5)
        vib_val = st.slider("Vibração Triaxial (mm/s):", 0.1, 12.0, p[1], 0.1)
        torque_val = st.slider("Torque do Fuso (Nm):", 20.0, 250.0, p[2], 1.0)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_s2:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-moisture"></i> SENSORES DE FLUXO E ELÉTRICOS
                </div>
            """,
            unsafe_allow_html=True,
        )
        vazao_val = st.slider("Vazão Refrigerante (L/min):", 0.5, 25.0, p[3], 0.5)
        corr_val = st.slider("Corrente do Estator (A):", 5.0, 60.0, p[4], 0.5)
        press_val = st.slider("Pressão Hidráulica (bar):", 0.5, 6.0, p[5], 0.1)
        ruido_val = st.slider("Nível de Ruído (dB):", 40.0, 100.0, p[6], 0.5)
        st.markdown("</div>", unsafe_allow_html=True)

    # Cálculo do Índice de Esforço Mecânico (IEM)
    iem_val = (torque_val * vib_val) / (vazao_val + 1.0)
    medias_liga = MEDIAS_HISTORICAS_LIGA.get(
        liga_sel, MEDIAS_HISTORICAS_LIGA["Aco_Carbono_316L"]
    )

    registro_in = {
        "temperatura_fuso_c": temp_val,
        "vibracao_triaxial_rms_mms": vib_val,
        "torque_fuso_nm": torque_val,
        "vazao_refrigerante_lmin": vazao_val,
        "corrente_estator_amp": corr_val,
        "pressao_hidraulica_bar": press_val,
        "nivel_ruido_db": ruido_val,
        "indice_esforco_mecanico": iem_val,
        "torque_fuso_nm_norm_liga": torque_val / medias_liga["torque_fuso_nm"],
        "corrente_estator_amp_norm_liga": (
            corr_val / medias_liga["corrente_estator_amp"]
        ),
        "vibracao_triaxial_rms_mms_norm_liga": (
            vib_val / medias_liga["vibracao_triaxial_rms_mms"]
        ),
        "temperatura_fuso_c_norm_liga": temp_val / medias_liga["temperatura_fuso_c"],
        "indice_esforco_mecanico_norm_liga": (
            iem_val / medias_liga["indice_esforco_mecanico"]
        ),
        "vibracao_triaxial_rms_mms_rolling_mean_30min": vib_val,
        "vibracao_triaxial_rms_mms_rolling_std_30min": 0.0,
        "temperatura_fuso_c_rolling_mean_30min": temp_val,
        "temperatura_fuso_c_rolling_std_30min": 0.0,
        "torque_fuso_nm_rolling_mean_30min": torque_val,
        "torque_fuso_nm_rolling_std_30min": 0.0,
        "corrente_estator_amp_rolling_mean_30min": corr_val,
        "corrente_estator_amp_rolling_std_30min": 0.0,
        "indice_esforco_mecanico_rolling_mean_30min": iem_val,
        "indice_esforco_mecanico_rolling_std_30min": 0.0,
        "vibracao_triaxial_rms_mms_rolling_mean_2h": vib_val,
        "vibracao_triaxial_rms_mms_rolling_std_2h": 0.0,
        "temperatura_fuso_c_rolling_mean_2h": temp_val,
        "temperatura_fuso_c_rolling_std_2h": 0.0,
        "torque_fuso_nm_rolling_mean_2h": torque_val,
        "torque_fuso_nm_rolling_std_2h": 0.0,
        "corrente_estator_amp_rolling_mean_2h": corr_val,
        "corrente_estator_amp_rolling_std_2h": 0.0,
        "indice_esforco_mecanico_rolling_mean_2h": iem_val,
        "indice_esforco_mecanico_rolling_std_2h": 0.0,
    }

    df_in = pd.DataFrame([registro_in])[artefatos["features"]]
    X_scaled = artefatos["scaler"].transform(df_in)
    X_scaled_df = pd.DataFrame(X_scaled, columns=artefatos["features"])
    classe_pred = int(artefatos["classifier"].predict(X_scaled_df)[0])
    probabilidades = artefatos["classifier"].predict_proba(X_scaled_df)[0]
    eh_anomalia = bool(artefatos["iso_forest"].predict(X_scaled_df)[0] == -1)

    # Cálculo físico de OEE
    oee_res = artefatos["oee_calc"].calcular_oee_instancia(classe_pred)

    with c_diag:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-cpu"></i> LAUDO DE INFERÊNCIA & MALHA FECHADA
                </div>
            """,
            unsafe_allow_html=True,
        )

        diag_desc = MAPA_DIAGNOSTICO[classe_pred]
        if classe_pred == 0 and not eh_anomalia:
            st.markdown(
                f'<div class="status-badge-ok"><i class="bi bi-check-circle-fill"></i> '
                f'{diag_desc}</div>',
                unsafe_allow_html=True,
            )
        elif classe_pred in [1, 2]:
            st.markdown(
                f'<div class="status-badge-warn"><i class="bi bi-exclamation-circle-fill"></i> '
                f'{diag_desc}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="status-badge-crit"><i class="bi bi-x-octagon-fill"></i> '
                f'{diag_desc}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Índice Esforço Mecânico (IEM):** `{iem_val:.2f}` (Base: 1.0)")
        iso_str = "ANOMALIA DETECTADA" if eh_anomalia else "PADRÃO CONHECIDO"
        st.markdown(f"**Detector Isolation Forest:** `{iso_str}`")

        # Probabilidades por classe
        st.markdown("**Distribuição de Probabilidades:**")
        for idx, prob in enumerate(probabilidades):
            st.progress(float(prob), text=f"Classe {idx}: {prob * 100:.1f}%")

        # Indicadores OEE
        st.markdown("<hr style='margin: 12px 0;'>", unsafe_allow_html=True)
        st.markdown("**Estimativa Dinâmica de OEE:**")
        col_o1, col_o2, col_o3 = st.columns(3)
        col_o1.metric("Disp. (A)", f"{oee_res['disponibilidade'] * 100:.1f}%")
        col_o2.metric("Perf. (P)", f"{oee_res['performance'] * 100:.1f}%")
        col_o3.metric("Qual. (Q)", f"{oee_res['qualidade'] * 100:.1f}%")

        # Comando de CLP em Malha Fechada
        acao_clp = oee_res["acao_clp"]
        if acao_clp == "MANTER_OPERACAO":
            cor_acao = "#0E9F6E"
        elif "REDUZIR" in acao_clp:
            cor_acao = "#EAB308"
        else:
            cor_acao = "#DC2626"

        st.markdown(
            f"""
            <div style="background-color: {cor_acao}; color: #FFFFFF; padding: 10px;
                        border-radius: 6px; text-align: center; font-weight: 700; margin-top: 10px;">
                <i class="bi bi-broadcast"></i> COMANDO CLP: {acao_clp}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def tela_visao_computacional() -> None:
    """Tela 05: Etapa 3 — CVOps: Inspeção Geométrica de Solda e Termografia HAZ."""
    st.markdown(
        """
        <div class="theory-box">
            <strong><i class="bi bi-journal-bookmark-fill"></i>
            FUNDAMENTAÇÃO DE VISÃO COMPUTACIONAL (<code>src/vision/</code>)</strong><br>
            1. <strong>Módulo Geométrico (<code>weld_inspector.py</code>):</strong>
            Aplica convolução Laplaciana 3x3 de nitidez, binarização Canny e algoritmo
            morfológico para quantificar microporos (<span class="math-pill">6 &le; área
            &le; 120 px</span>) e continuidade longitudinal (<span class="math-pill">&ge;
            240 px</span>).<br>
            2. <strong>Módulo Termográfico (<code>thermal_haz_inspector.py</code>):</strong>
            Converte a matriz para o espaço HSV e segmenta zonas quentes para aferir a
            Zona Afetada pelo Calor (HAZ), prevenindo fragilização e trincas térmicas.
        </div>
        """,
        unsafe_allow_html=True,
    )

    sub_aba = st.radio(
        "Selecione o Módulo CVOps:",
        [
            "Inspeção Geométrica do Cordão de Solda (Canny + Detecção de Microporos)",
            "Inspeção Termográfica Infravermelha (Segmentação HSV da Zona Afetada - HAZ)",
        ],
        horizontal=True,
    )

    if "Geométrica" in sub_aba:
        c_in, c_res = st.columns([1, 2.2])
        with c_in:
            st.markdown(
                """
                <div class="corp-card">
                    <div class="corp-card-title">
                        <i class="bi bi-gear-fill"></i> PARÂMETROS DE CONTROLE
                    </div>
                """,
                unsafe_allow_html=True,
            )
            fonte_img = st.radio(
                "Origem da Imagem:",
                ["Amostras de Bancada", "Upload de Imagem Externa"],
            )
            canny_l = st.slider("Canny Low Threshold:", 20, 100, 50, 5)
            canny_h = st.slider("Canny High Threshold:", 100, 250, 150, 10)
            max_poros = st.slider("Tolerância Máxima de Poros:", 0, 5, 2, 1)

            img_bgr = None
            nome_amostra = "Amostra"
            if "Upload" in fonte_img:
                arquivo = st.file_uploader(
                    "Carregar Imagem (.png, .jpg, .jpeg)",
                    type=["png", "jpg", "jpeg"],
                )
                if arquivo is not None:
                    img_pil = Image.open(arquivo).convert("RGB")
                    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    nome_amostra = arquivo.name
            else:
                opcoes_amostra = {
                    "Cordão Conforme (Aprovado)": "amostra_cordao_conforme.png",
                    "Cordão com Porosidade Excessiva": "amostra_cordao_poroso.png",
                    "Cordão Descontínuo (Quebra de Arco)": (
                        "amostra_cordao_descontinuo.png"
                    ),
                }
                escolha = st.selectbox(
                    "Amostra Sintética:", list(opcoes_amostra.keys())
                )
                caminho_amostra = (
                    ROOT_DIR
                    / "data"
                    / "raw"
                    / "vision"
                    / "geometric"
                    / opcoes_amostra[escolha]
                )
                if caminho_amostra.exists():
                    img_bgr = cv2.imread(str(caminho_amostra))
                    nome_amostra = escolha
            st.markdown("</div>", unsafe_allow_html=True)

        with c_res:
            st.markdown(
                """
                <div class="corp-card">
                    <div class="corp-card-title">
                        <i class="bi bi-eye-fill"></i> RESULTADO DA INSPEÇÃO GEOMÉTRICA
                    </div>
                """,
                unsafe_allow_html=True,
            )
            if img_bgr is not None:
                inspector_geom = WeldGeometricInspector(
                    canny_low=canny_l,
                    canny_high=canny_h,
                    max_poros_permitidos=max_poros,
                )
                res = inspector_geom.inspecionar_cordao(img_bgr)
                aprovado = res["aprovado"]

                if aprovado:
                    st.markdown(
                        f'<div class="status-badge-ok"><i class="bi bi-check-circle-fill"></i> '
                        f'CORDÃO APROVADO — {nome_amostra}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="status-badge-crit"><i class="bi bi-x-octagon-fill"></i> '
                        f'CORDÃO REPROVADO ({res["motivo"]}) — {nome_amostra}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Microporos Detectados", f"{res['total_poros']} un")
                m2.metric(
                    "Comprimento do Cordão", f"{res['comprimento_cordao_px']:.1f} px"
                )
                m3.metric("Largura Média", f"{res['largura_media_px']:.1f} px")

                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    st.caption("1. Imagem Óptica Original")
                    st.image(
                        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                        width='stretch',
                    )
                with col_v2:
                    st.caption("2. Máscara de Bordas (Canny)")
                    st.image(res["mascara_bordas"], width='stretch')
                with col_v3:
                    st.caption("3. Delimitação & Poros")
                    st.image(
                        cv2.cvtColor(res["imagem_anotada"], cv2.COLOR_BGR2RGB),
                        width='stretch',
                    )
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Inspeção Termográfica da HAZ
        c_in, c_res = st.columns([1, 2.2])
        with c_in:
            st.markdown(
                """
                <div class="corp-card">
                    <div class="corp-card-title">
                        <i class="bi bi-thermometer-half"></i> PARÂMETROS TÉRMICOS
                    </div>
                """,
                unsafe_allow_html=True,
            )
            fonte_t = st.radio(
                "Origem da Imagem:",
                ["Amostras de Bancada", "Upload de Imagem"],
                key="termo_src",
            )
            limiar_area = st.slider(
                "Limiar de Área Crítica HAZ (px):", 2000, 15000, 8000, 500
            )

            img_t_bgr = None
            nome_t = "Termografia"
            if "Upload" in fonte_t:
                arq_t = st.file_uploader(
                    "Carregar Termografia (.png, .jpg)",
                    type=["png", "jpg", "jpeg"],
                    key="up_t",
                )
                if arq_t is not None:
                    img_t_pil = Image.open(arq_t).convert("RGB")
                    img_t_bgr = cv2.cvtColor(np.array(img_t_pil), cv2.COLOR_RGB2BGR)
                    nome_t = arq_t.name
            else:
                opcoes_termo = {
                    "Termografia Estável (Nominal)": "termografia_haz_estavel.png",
                    "Termografia Superaquecimento Crítico": "termografia_haz_critica.png",
                }
                escolha_t = st.selectbox(
                    "Amostra Termográfica:", list(opcoes_termo.keys())
                )
                caminho_t = (
                    ROOT_DIR
                    / "data"
                    / "raw"
                    / "vision"
                    / "thermal"
                    / opcoes_termo[escolha_t]
                )
                if caminho_t.exists():
                    img_t_bgr = cv2.imread(str(caminho_t))
                    nome_t = escolha_t
            st.markdown("</div>", unsafe_allow_html=True)

        with c_res:
            st.markdown(
                """
                <div class="corp-card">
                    <div class="corp-card-title">
                        <i class="bi bi-fire"></i> DIAGNÓSTICO TERMOGRÁFICO DA HAZ
                    </div>
                """,
                unsafe_allow_html=True,
            )
            if img_t_bgr is not None:
                inspector_haz = ThermalHAZInspector(
                    limiar_area_critica_px=float(limiar_area)
                )
                res_haz = inspector_haz.inspecionar_termografia_haz(img_t_bgr)
                aprov_haz = res_haz["aprovado"]

                if aprov_haz:
                    st.markdown(
                        f'<div class="status-badge-ok"><i class="bi bi-check-circle-fill"></i> '
                        f'DISPERSÃO TÉRMICA ESTÁVEL — {nome_t}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="status-badge-crit"><i class="bi bi-fire"></i> '
                        f'ALERTA DE SUPERAQUECIMENTO CRÍTICO — {nome_t}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                t1, t2, t3 = st.columns(3)
                t1.metric(
                    "Área da Zona Afetada (HAZ)",
                    f"{res_haz['area_critica_px']:.1f} px",
                )
                t2.metric("Focos Térmicos", f"{res_haz['focos_detectados']} un")
                t3.metric("Status Térmico", "Nominal" if aprov_haz else "Crítico")

                col_th1, col_th2, col_th3 = st.columns(3)
                with col_th1:
                    st.caption("1. Termografia BGR Original")
                    st.image(
                        cv2.cvtColor(img_t_bgr, cv2.COLOR_BGR2RGB),
                        width='stretch',
                    )
                with col_th2:
                    st.caption("2. Máscara HSV Morfológica")
                    st.image(res_haz["mascara_termica"], width='stretch')
                with col_th3:
                    st.caption("3. Delimitação da HAZ Anotada")
                    st.image(
                        cv2.cvtColor(res_haz["imagem_anotada"], cv2.COLOR_BGR2RGB),
                        width='stretch',
                    )
            st.markdown("</div>", unsafe_allow_html=True)


def tela_documentacao_teorica() -> None:
    """Tela 06: Etapa 5 — Governança Regulatória, ESG e Documentos Técnicos."""
    st.markdown(
        """
        <div class="theory-box">
            <strong><i class="bi bi-shield-check"></i>
            GOVERNANÇA REGULATÓRIA, ESG & CONFORMIDADE (<code>docs/</code>)</strong><br>
            A operação na VulcanTech Smart Metals S.A. segue normas rigorosas da
            Indústria 4.0: enquadramento no <strong>EU AI Act (Sistemas de Alto Risco)</strong>,
            restrição pela <strong>LGPD</strong> (câmeras com campo de visão restrito à poça de
            fusão) e segregação de redes pelo <strong>Modelo Purdue (Nível 2/3 com DMZ)</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_gov, c_pdf = st.columns([1.3, 1])
    with c_gov:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-file-earmark-lock2-fill"></i> PILARES TÉCNICOS & ÉTICOS
                </div>
                <ul style="font-size: 0.88rem; line-height: 1.7; padding-left: 18px; margin-bottom: 0;">
                    <li><strong>EU AI Act (Alto Risco):</strong> A atuação em malha fechada sobre os CLPs impõe requisitos de explicabilidade (SHAP), validação humana e robustez de dados.</li>
                    <li><strong>LGPD & Privacidade:</strong> Câmeras operam em plano fechado (FOV restrito de 15x15 cm) sem enquadramento biométrico de operadores.</li>
                    <li><strong>Cibersegurança OT/IT:</strong> Nó Edge opera no Nível 2 (Supervisão), comunicando-se com Nível 3 via mTLS e sem acesso direto à Internet.</li>
                    <li><strong>Sustentabilidade & ESG:</strong> A detecção precoce previne o refugo de ligas nobres como Titânio Ti-6Al-4V, reduzindo pegada de carbono.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_pdf:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title">
                    <i class="bi bi-file-earmark-pdf-fill"></i> RELATÓRIOS TÉCNICOS OFICIAIS
                </div>
                <p style="font-size: 0.85rem; color: #475569;">
                    Acesse e baixe os laudos de engenharia gerados para auditoria:
                </p>
            """,
            unsafe_allow_html=True,
        )
        arquivos_docs = [
            ("docs/parecer_tecnico_etico_regulatorio.pdf", "Parecer Regulatório & ESG"),
            ("docs/plano_mlops_drift.pdf", "Plano de Monitoramento de Data Drift"),
            ("docs/relatorio_saneamento.pdf", "Relatório de Saneamento & Features"),
        ]
        for rel_path, rotulo in arquivos_docs:
            p_pdf = ROOT_DIR / rel_path
            if p_pdf.exists():
                with open(p_pdf, "rb") as f_pdf:
                    st.download_button(
                        label=f"Baixar {rotulo}",
                        data=f_pdf.read(),
                        file_name=p_pdf.name,
                        mime="application/pdf",
                        width='stretch',
                    )
            else:
                st.caption(f"{rotulo} (não encontrado em {rel_path})")
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# 4. LOOP PRINCIPAL DA APLICAÇÃO STREAMLIT
# =============================================================================
def main() -> None:
    """Função de entrada que orquestra estados e renderiza o dashboard."""
    st.set_page_config(
        page_title="VulcanTech — Edge AI & CVOps Cockpit",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    aplicar_estilo_industrial()

    df_raw, df_proc = carregar_dados()
    artefatos = carregar_artefatos_ml()

    # Barra lateral de navegação executiva
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 12px 0 16px 0;">
            <div style="font-size: 1.25rem; font-weight: 800; color: #0A192F; letter-spacing: 0.5px;">
                VULCANTECH
            </div>
            <div style="font-size: 0.76rem; font-weight: 600; color: #0090D4; text-transform: uppercase;">
                Smart Metals S.A. — Edge AI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lista_modulos = list(MODULOS_PROJETO.keys())
    modulo_selecionado = st.sidebar.radio(
        "Módulos Operacionais:",
        lista_modulos,
        index=0,
    )

    st.sidebar.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="font-size: 0.78rem; color: #475569; line-height: 1.5;">
            <strong>Status de Borda:</strong>
            <span style="color: #10B981; font-weight: 700;">ONLINE</span><br>
            <strong>Latência Inferência:</strong> &lt; 2.2 ms<br>
            <strong>Células Conectadas:</strong> CEL-201..204<br>
            <strong>Cibersistema:</strong> CentroWEG / SENAI
        </div>
        """,
        unsafe_allow_html=True,
    )

    meta = MODULOS_PROJETO[modulo_selecionado]
    renderizar_cabecalho_sistema(meta)

    # Roteamento de telas
    if modulo_selecionado == "Visão Geral & Planta":
        tela_visao_geral(df_raw, df_proc, artefatos)
    elif modulo_selecionado == "Saneamento & Engenharia (Etapa 1)":
        tela_engenharia_dados(df_raw, df_proc)
    elif modulo_selecionado == "Modelagem Preditiva & MLOps (Etapa 2)":
        tela_modelagem_ml(df_proc, artefatos)
    elif modulo_selecionado == "Simulador Edge em Malha Fechada (Etapa 4)":
        tela_simulador_edge(artefatos)
    elif modulo_selecionado == "Visão Computacional CVOps (Etapa 3)":
        tela_visao_computacional()
    elif modulo_selecionado == "Governança, ESG & Documentação (Etapa 5)":
        tela_documentacao_teorica()

    renderizar_rodape_sistema()


if __name__ == "__main__":
    main()
