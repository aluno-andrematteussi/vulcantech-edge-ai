"""Testes unitários automatizados para o dashboard executivo Streamlit."""

from pathlib import Path
import pandas as pd
from src.dashboard.app import (
    alinhar_features_para_modelo,
    carregar_artefatos_ml,
    carregar_dados,
)


def test_carregar_dados_dashboard() -> None:
    """Verifica se os datasets brutos e processados são carregados corretamente."""
    df_raw, df_proc = carregar_dados()
    assert not df_raw.empty, "Dataset bruto não deveria estar vazio."
    assert not df_proc.empty, "Dataset processado não deveria estar vazio."
    assert len(df_raw) == 10000, "Dataset bruto deve conter 10.000 amostras."
    assert len(df_proc) == 10000, "Dataset processado deve conter 10.000 amostras."
    assert "tipo_falha" in df_proc.columns, "Coluna tipo_falha deve existir no parquet."


def test_carregar_artefatos_ml_dashboard() -> None:
    """Garante que todos os modelos serializados sejam recuperados pelo cache."""
    artefatos = carregar_artefatos_ml()
    chaves_esperadas = ["classifier", "iso_forest", "scaler", "features", "oee_calc"]
    for chave in chaves_esperadas:
        assert chave in artefatos, f"Artefato '{chave}' ausente no dicionário."
    assert len(artefatos["features"]) == 33, "Devem haver 33 features no modelo."


def test_alinhar_features_para_modelo() -> None:
    """Verifica se o alinhamento de colunas preenche colunas ausentes com 0.0."""
    features_esperadas = ["feature_a", "feature_b", "feature_c"]
    df_incompleto = pd.DataFrame([{"feature_a": 10.5}])
    df_alinhado = alinhar_features_para_modelo(df_incompleto, features_esperadas)

    assert list(df_alinhado.columns) == features_esperadas
    assert df_alinhado.iloc[0]["feature_a"] == 10.5
    assert df_alinhado.iloc[0]["feature_b"] == 0.0
    assert df_alinhado.iloc[0]["feature_c"] == 0.0

