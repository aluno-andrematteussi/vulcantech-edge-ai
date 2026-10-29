"""Testes unitários para o módulo de saneamento e engenharia de dados da VulcanTech."""

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.feature_engineering import IndustrialFeatureEngineer
from src.data_pipeline.preprocessor import IndustrialDataSanitizer


@pytest.fixture
def raw_data_path() -> str:
    """Retorna o caminho do arquivo de telemetria bruta de teste."""
    return "data/raw/dados_sensores_vulcantech.csv"


def test_sanitizer_carregamento_e_estrutura(raw_data_path: str):
    """Valida a carga estruturada e o relatório inicial de integridade."""
    sanitizer = IndustrialDataSanitizer(raw_data_path)
    df = sanitizer.carregar_dados()

    assert not df.empty
    assert len(df) == 10000
    assert "timestamp" in df.columns
    assert "id_celula" in df.columns

    relatorio = sanitizer.inspecionar_estrutura()
    assert relatorio["total_registos"] == 10000
    assert len(relatorio["celulas"]) == 4
    assert len(relatorio["ligas_metalicas"]) == 3


def test_sanitizer_filtragem_ruidos_fisicos(raw_data_path: str):
    """Garante que leituras de sensores defeituosos (-99 mm/s, 999.9 °C, etc.)

    sejam convertidas para valores nulos (NaN).
    """
    sanitizer = IndustrialDataSanitizer(raw_data_path)
    sanitizer.carregar_dados()
    df_filtrado = sanitizer.filtrar_ruidos_sensores()

    # Confirma que os valores impossíveis foram removidos
    assert (df_filtrado["temperatura_fuso_c"] >= 900.0).sum() == 0
    assert (df_filtrado["vibracao_triaxial_rms_mms"] < 0.0).sum() == 0
    assert (df_filtrado["pressao_hidraulica_bar"] < 0.0).sum() == 0
    assert (df_filtrado["vazao_refrigerante_lmin"] < 0.0).sum() == 0


def test_sanitizer_pipeline_completo(raw_data_path: str):
    """Valida o preenchimento contínuo sem nenhum valor nulo restante."""
    sanitizer = IndustrialDataSanitizer(raw_data_path)
    df_limpo = sanitizer.executar_pipeline_saneamento()

    assert df_limpo.isnull().sum().sum() == 0
    assert len(df_limpo) == 10000


def test_feature_engineering_iem_e_normalizacao(raw_data_path: str):
    """Valida o cálculo do IEM e a equalização de baseline por liga metálica."""
    sanitizer = IndustrialDataSanitizer(raw_data_path)
    df_limpo = sanitizer.executar_pipeline_saneamento()

    engineer = IndustrialFeatureEngineer(df_limpo)
    df_final = engineer.executar_pipeline()

    # 1. Verifica presença do IEM
    assert "indice_esforco_mecanico" in df_final.columns
    assert (df_final["indice_esforco_mecanico"] >= 0).all()

    # 2. Verifica variáveis normalizadas por liga metálica
    assert "torque_fuso_nm_norm_liga" in df_final.columns
    assert "corrente_estator_amp_norm_liga" in df_final.columns

    # A média da variável normalizada por liga deve convergir para ~1.0
    for liga in df_final["liga_metalica"].unique():
        sub = df_final[df_final["liga_metalica"] == liga]
        assert pytest.approx(sub["torque_fuso_nm_norm_liga"].mean(), rel=1e-2) == 1.0

    # 3. Verifica estatísticas móveis
    assert "vibracao_triaxial_rms_mms_rolling_mean_30min" in df_final.columns
    assert "vibracao_triaxial_rms_mms_rolling_std_2h" in df_final.columns
    assert df_final["vibracao_triaxial_rms_mms_rolling_std_30min"].isnull().sum() == 0
