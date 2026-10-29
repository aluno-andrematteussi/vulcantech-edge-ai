"""Testes automatizados para os módulos de modelagem preditiva, OEE e MLOps."""

import os
import joblib
import pandas as pd
import pytest

from src.ml.evaluation import IndustrialModelEvaluator
from src.ml.oee_calculator import IndustrialOEECalculator
from src.ml.train import IndustrialModelTrainer


@pytest.fixture
def parquet_data_path() -> str:
    """Retorna o caminho do dataset de features enriquecidas."""
    return "data/processed/features_eng.parquet"


def test_oee_calculator_instancias():
    """Valida o cálculo do OEE e penalizações físicas por classe de falha."""
    calc = IndustrialOEECalculator()

    # Regime Normal (Classe 0)
    res_normal = calc.calcular_oee_instancia(0)
    assert res_normal["oee"] == pytest.approx(0.8844, rel=1e-2)
    assert res_normal["acao_clp"] == "MANTER_OPERACAO"
    assert res_normal["nivel_risco"] == "Baixo"

    # Desgaste de Ferramenta (Classe 1)
    res_desgaste = calc.calcular_oee_instancia(1)
    assert res_desgaste["disponibilidade"] < res_normal["disponibilidade"]
    assert res_desgaste["acao_clp"] == "REDUZIR_VELOCIDADE_50%"

    # Falha de Refrigeração (Classe 2)
    res_refrig = calc.calcular_oee_instancia(2)
    assert res_refrig["qualidade"] < res_normal["qualidade"]
    assert res_refrig["nivel_risco"] == "Alto"

    # Desalinhamento / Trinca Mecânica (Classe 3)
    res_trinca = calc.calcular_oee_instancia(3)
    assert res_trinca["disponibilidade"] <= 0.50
    assert res_trinca["acao_clp"] == "INTERROMPER_EMERGENCIA"
    assert res_trinca["nivel_risco"] == "Critico"


def test_oee_calculator_simulacao_lote():
    """Valida a simulação de impacto no OEE sobre um vetor de predições."""
    calc = IndustrialOEECalculator()
    predicoes = pd.Series([0, 0, 1, 2, 3, 0])
    resumo = calc.simular_impacto_lote(predicoes)

    assert resumo["total_amostras"] == 6
    assert resumo["oee_medio_operacional"] < resumo["oee_nominal_referencia"]
    assert "MANTER_OPERACAO" in resumo["distribuicao_acoes_clp"]
    assert "INTERROMPER_EMERGENCIA" in resumo["distribuicao_acoes_clp"]


def test_artefatos_mlops_carregamento():
    """Valida a integridade dos modelos e metadados persistidos em models/."""
    diretorio_modelos = "models"
    caminho_classificador = os.path.join(
        diretorio_modelos, "classifier_model.joblib"
    )
    caminho_iso = os.path.join(
        diretorio_modelos, "isolation_forest.joblib"
    )
    caminho_scaler = os.path.join(diretorio_modelos, "scaler.joblib")
    caminho_features = os.path.join(
        diretorio_modelos, "feature_names.joblib"
    )

    assert os.path.exists(caminho_classificador)
    assert os.path.exists(caminho_iso)
    assert os.path.exists(caminho_scaler)
    assert os.path.exists(caminho_features)

    classificador = joblib.load(caminho_classificador)
    iso = joblib.load(caminho_iso)
    feature_names = joblib.load(caminho_features)

    assert hasattr(classificador, "predict")
    assert hasattr(iso, "predict")
    assert len(feature_names) == 33


def test_classificador_recall_minimo_90_pct(parquet_data_path: str):
    """Garante que o classificador salvo atinja Recall >= 90% em todas as falhas."""
    trainer = IndustrialModelTrainer(parquet_data_path)
    _, X_test, _, y_test = trainer.preparar_dados()

    classificador = joblib.load("models/classifier_model.joblib")
    relatorio = IndustrialModelEvaluator.avaliar_classificador(
        modelo=classificador,
        X_test=X_test,
        y_test=y_test,
        nome_modelo="Classifier_Producao",
        exibir_grafico=False,
    )

    # Critério da rubrica de avaliação: Recall >= 90% nas falhas
    recall_desgaste = relatorio["Desgaste_Ferramenta"]["recall"]
    recall_refrig = relatorio["Falha_Refrigeracao"]["recall"]
    recall_trinca = relatorio["Desalinhamento_Trinca"]["recall"]

    assert recall_desgaste >= 0.90, f"Recall desgaste ({recall_desgaste}) < 0.90"
    assert recall_refrig >= 0.90, f"Recall refrigeracao ({recall_refrig}) < 0.90"
    assert recall_trinca >= 0.90, f"Recall trinca ({recall_trinca}) < 0.90"

