"""Testes automatizados para a API FastAPI e simulação de malha fechada."""

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_check_status_e_prontidao():
    """Valida se o microsserviço está ativo e reporta 33 features na borda."""
    resposta = client.get("/health")
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["status"] == "healthy"
    assert dados["features_esperadas"] == 33
    assert "Edge" in dados["dispositivo_edge"]


def test_predict_regime_nominal_manter_operacao():
    """Valida inferência nominal com parâmetros calibrados no regime normal."""
    payload = {
        "id_celula": "CEL-201",
        "liga_metalica": "Aco_Carbono_316L",
        "temperatura_fuso_c": 54.0,
        "vibracao_triaxial_rms_mms": 1.4,
        "torque_fuso_nm": 108.0,
        "vazao_refrigerante_lmin": 28.0,
        "corrente_estator_amp": 27.5,
        "pressao_hidraulica_bar": 160.0,
        "nivel_ruido_db": 53.0,
    }
    resposta = client.post("/predict", json=payload)
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["classe_predita"] == 0
    assert dados["diagnostico"] == "Operação Normal"
    assert dados["acao_industrial_recomendada"] == "MANTER_OPERACAO"
    assert dados["status_operacional"] == "NOMINAL"
    assert dados["oee_estimado"]["oee"] > 0.85


def test_predict_desgaste_ferramenta_reduzir_velocidade():
    """Valida diagnóstico de desgaste severo e ação de desaceleração do fuso."""
    payload = {
        "id_celula": "CEL-202",
        "liga_metalica": "Aco_Carbono_316L",
        "temperatura_fuso_c": 72.0,
        "vibracao_triaxial_rms_mms": 5.2,
        "torque_fuso_nm": 195.0,
        "vazao_refrigerante_lmin": 25.0,
        "corrente_estator_amp": 45.0,
        "pressao_hidraulica_bar": 155.0,
        "nivel_ruido_db": 82.0,
    }
    resposta = client.post("/predict", json=payload)
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["classe_predita"] == 1
    assert "Desgaste" in dados["diagnostico"]
    assert dados["acao_industrial_recomendada"] == "REDUZIR_VELOCIDADE_50%"
    assert dados["status_operacional"] == "INTERVENCAO_IMEDIATA"


def test_predict_obstrucao_refrigeracao_superaquecimento():
    """Valida falha de refrigeração com queda drástica de vazão e aquecimento."""
    payload = {
        "id_celula": "CEL-203",
        "liga_metalica": "Aluminio_7075",
        "temperatura_fuso_c": 98.0,
        "vibracao_triaxial_rms_mms": 1.8,
        "torque_fuso_nm": 105.0,
        "vazao_refrigerante_lmin": 2.5,
        "corrente_estator_amp": 30.0,
        "pressao_hidraulica_bar": 150.0,
        "nivel_ruido_db": 65.0,
    }
    resposta = client.post("/predict", json=payload)
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["classe_predita"] == 2
    assert "Refrigeração" in dados["diagnostico"]
    assert dados["acao_industrial_recomendada"] == "REDUZIR_VELOCIDADE_50%"
    assert dados["oee_estimado"]["qualidade"] < 0.80


def test_predict_trinca_desalinhamento_interromper_emergencia():
    """Valida trinca mecânica com perda de pressão e ordem de parada ao CLP."""
    payload = {
        "id_celula": "CEL-204",
        "liga_metalica": "Liga_Titanio_Ti6Al4V",
        "temperatura_fuso_c": 65.0,
        "vibracao_triaxial_rms_mms": 9.5,
        "torque_fuso_nm": 270.0,
        "vazao_refrigerante_lmin": 27.0,
        "corrente_estator_amp": 68.0,
        "pressao_hidraulica_bar": 95.0,
        "nivel_ruido_db": 98.0,
    }
    resposta = client.post("/predict", json=payload)
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["classe_predita"] == 3
    assert "Trinca" in dados["diagnostico"]
    assert dados["acao_industrial_recomendada"] == "INTERROMPER_EMERGENCIA"
    assert dados["oee_estimado"]["disponibilidade"] <= 0.50


def test_predict_bloqueio_violacao_limite_fisico_pydantic():
    """Garante bloqueio (HTTP 422) de leituras que violam limites físicos absolutos."""
    payload_invalido = {
        "id_celula": "CEL-201",
        "liga_metalica": "Aco_Carbono_316L",
        "temperatura_fuso_c": 999.9,  # Violação de limite le=200.0
        "vibracao_triaxial_rms_mms": 1.5,
        "torque_fuso_nm": 110.0,
        "vazao_refrigerante_lmin": 28.0,
        "corrente_estator_amp": 28.0,
        "pressao_hidraulica_bar": -1.0,  # Violação de limite ge=0.0
        "nivel_ruido_db": 55.0,
    }
    resposta = client.post("/predict", json=payload_invalido)
    assert resposta.status_code == 422


def test_predict_batch():
    """Valida a inferência simultânea em lote para múltiplas células."""
    payload_lote = {
        "leituras": [
            {
                "id_celula": "CEL-201",
                "liga_metalica": "Aco_Carbono_316L",
                "temperatura_fuso_c": 55.0,
                "vibracao_triaxial_rms_mms": 1.5,
                "torque_fuso_nm": 110.0,
                "vazao_refrigerante_lmin": 28.0,
                "corrente_estator_amp": 28.0,
                "pressao_hidraulica_bar": 160.0,
                "nivel_ruido_db": 55.0,
            },
            {
                "id_celula": "CEL-202",
                "liga_metalica": "Liga_Titanio_Ti6Al4V",
                "temperatura_fuso_c": 75.0,
                "vibracao_triaxial_rms_mms": 7.0,
                "torque_fuso_nm": 240.0,
                "vazao_refrigerante_lmin": 25.0,
                "corrente_estator_amp": 55.0,
                "pressao_hidraulica_bar": 140.0,
                "nivel_ruido_db": 88.0,
            },
        ]
    }
    resposta = client.post("/predict-batch", json=payload_lote)
    assert resposta.status_code == 200
    dados = resposta.json()
    assert len(dados) == 2
    assert dados[0]["id_celula"] == "CEL-201"
    assert dados[1]["id_celula"] == "CEL-202"

