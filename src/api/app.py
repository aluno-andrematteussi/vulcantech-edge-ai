"""Microsserviço RESTful assíncrono em FastAPI para predição e malha fechada.

Executa inferência de alta frequência na borda (Edge Computing) e orienta
ações automáticas aos Controladores Lógicos Programáveis (CLP) da fábrica.
"""

import os
from typing import Dict, List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.api.schemas import (
    BatchSensorReadingInput,
    HealthResponse,
    PredictionOutput,
    SensorReadingInput,
)
from src.ml.oee_calculator import IndustrialOEECalculator

app = FastAPI(
    title="VulcanTech Smart Metals — Predictive & Edge AI Service",
    description=(
        "Microsserviço de inferência de telemetria em tempo real, cálculo de"
        " OEE e atuação em malha fechada para usinagem, estamparia e solda."
    ),
    version="1.0.0",
)

# Mapeamento oficial dos diagnósticos das células
MAPA_DIAGNOSTICO = {
    0: "Operação Normal",
    1: "Desgaste Severo de Ferramenta / Fuso",
    2: "Obstrução no Sistema de Refrigeração / Superaquecimento",
    3: "Desalinhamento da Matriz de Prensa / Trinca Mecânica",
}

# Médias históricas de processo por liga para normalização de lote
MEDIAS_HISTORICAS_LIGA = {
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

# Inicialização e carregamento dos artefatos treinados na memória
DIRETORIO_MODELOS = os.getenv("MODELS_DIR", "models")
try:
    classifier = joblib.load(
        os.path.join(DIRETORIO_MODELOS, "classifier_model.joblib")
    )
    isolation_forest = joblib.load(
        os.path.join(DIRETORIO_MODELOS, "isolation_forest.joblib")
    )
    scaler = joblib.load(os.path.join(DIRETORIO_MODELOS, "scaler.joblib"))
    feature_names = joblib.load(
        os.path.join(DIRETORIO_MODELOS, "feature_names.joblib")
    )
    oee_calculator = IndustrialOEECalculator()
except Exception as e:
    raise RuntimeError(
        f"Falha ao carregar artefatos MLOps em '{DIRETORIO_MODELOS}': {str(e)}"
    )


def _processar_features_entrada(
    dados: SensorReadingInput,
) -> pd.DataFrame:
    """Deriva o Índice de Esforço Mecânico (IEM), normalização e estatísticas.

    Args:
        dados: Leitura de telemetria validada pelo Pydantic.

    Returns:
        pd.DataFrame: DataFrame alinhado com as 33 features esperadas.
    """
    registro = dados.model_dump()

    # 1. Cálculo do Índice de Esforço Mecânico (IEM)
    vazao = max(0.0, registro["vazao_refrigerante_lmin"])
    iem = (registro["torque_fuso_nm"] * registro["vibracao_triaxial_rms_mms"]) / (
        vazao + 1.0
    )
    registro["indice_esforco_mecanico"] = iem

    # 2. Normalização condicionada pela liga metálica
    medias_liga = MEDIAS_HISTORICAS_LIGA.get(
        registro["liga_metalica"],
        MEDIAS_HISTORICAS_LIGA["Aco_Carbono_316L"],
    )
    for var, media_val in medias_liga.items():
        registro[f"{var}_norm_liga"] = registro[var] / media_val

    # 3. Mapeamento de estatísticas móveis pontuais para inferência em borda
    for var in [
        "vibracao_triaxial_rms_mms",
        "temperatura_fuso_c",
        "torque_fuso_nm",
        "corrente_estator_amp",
        "indice_esforco_mecanico",
    ]:
        registro[f"{var}_rolling_mean_30min"] = registro[var]
        registro[f"{var}_rolling_std_30min"] = 0.0
        registro[f"{var}_rolling_mean_2h"] = registro[var]
        registro[f"{var}_rolling_std_2h"] = 0.0

    df_entrada = pd.DataFrame([registro])
    df_selecionado = df_entrada[feature_names]
    return df_selecionado


@app.get(
    "/health", response_model=HealthResponse, tags=["Infraestrutura & Edge"]
)
def check_health():
    """Valida o estado de prontidão da API e dos modelos serializados."""
    return {
        "status": "healthy",
        "versao_servico": "1.0.0",
        "versao_modelo": type(classifier).__name__,
        "features_esperadas": len(feature_names),
        "dispositivo_edge": "Industrial IPC / Jetson Edge Node",
    }


@app.post(
    "/predict", response_model=PredictionOutput, tags=["Inferência & Malha Fechada"]
)
def predict_single(reading: SensorReadingInput):
    """Executa a inferência em tempo real e prescreve ação direta para o CLP."""
    try:
        X_bruto = _processar_features_entrada(reading)
        X_escalado = pd.DataFrame(
            scaler.transform(X_bruto), columns=feature_names
        )

        classe_idx = int(classifier.predict(X_escalado)[0])

        probabilidades: Dict[str, float] = {}
        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(X_escalado)[0]
            probabilidades = {
                MAPA_DIAGNOSTICO[i]: round(float(p), 4)
                for i, p in enumerate(probs)
            }

        # Detecção não supervisionada (-1 = anomalia operacional, 1 = normal)
        anomalia = bool(isolation_forest.predict(X_escalado)[0] == -1)

        # Cálculo do impacto em OEE e prescrição da malha fechada
        dados_oee = oee_calculator.calcular_oee_instancia(classe_idx)
        acao_clp = dados_oee["acao_clp"]

        status_operacional = (
            "INTERVENCAO_IMEDIATA"
            if (classe_idx != 0 or anomalia)
            else "NOMINAL"
        )

        return {
            "id_celula": reading.id_celula,
            "liga_metalica": reading.liga_metalica,
            "classe_predita": classe_idx,
            "diagnostico": MAPA_DIAGNOSTICO.get(
                classe_idx, "Indeterminado"
            ),
            "probabilidades": probabilidades,
            "acao_industrial_recomendada": acao_clp,
            "anomalia_detectada": anomalia,
            "oee_estimado": dados_oee,
            "status_operacional": status_operacional,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no pipeline de inferência: {str(e)}",
        )


@app.post(
    "/predict-batch",
    response_model=List[PredictionOutput],
    tags=["Inferência & Malha Fechada"],
)
def predict_batch(batch: BatchSensorReadingInput):
    """Processa lote simultâneo de telemetria das 4 células de manufatura."""
    respostas = []
    for leitura in batch.leituras:
        respostas.append(predict_single(leitura))
    return respostas

