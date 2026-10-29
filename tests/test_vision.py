"""Testes automatizados para os módulos de visão computacional (CVOps) de soldagem."""

import os
import cv2
import numpy as np
import pytest

from src.vision.generate_samples import gerar_amostras_soldagem
from src.vision.thermal_haz_inspector import ThermalHAZInspector
from src.vision.weld_inspector import WeldGeometricInspector


@pytest.fixture(scope="session", autouse=True)
def setup_amostras():
    """Gera as amostras de imagem sintéticas no início da sessão de testes."""
    gerar_amostras_soldagem()


def test_geracao_amostras_arquivos_existentes():
    """Garante que todas as 5 amostras de calibração foram gravadas em disco."""
    amostras = [
        "data/raw/vision/geometric/amostra_cordao_conforme.png",
        "data/raw/vision/geometric/amostra_cordao_poroso.png",
        "data/raw/vision/geometric/amostra_cordao_descontinuo.png",
        "data/raw/vision/thermal/termografia_haz_estavel.png",
        "data/raw/vision/thermal/termografia_haz_critica.png",
    ]
    for p in amostras:
        assert os.path.exists(p), f"Arquivo não encontrado: {p}"


def test_inspecao_cordao_conforme():
    """Valida a aprovação de cordão de solda contínuo e sem microporos."""
    inspector = WeldGeometricInspector()
    img = cv2.imread("data/raw/vision/geometric/amostra_cordao_conforme.png")
    resultado = inspector.inspecionar_cordao(img)

    assert resultado["aprovado"] is True
    assert resultado["total_poros"] <= 2
    assert resultado["descontinuidade"] is False
    assert resultado["comprimento_cordao_px"] >= 240.0
    assert "conforme" in resultado["motivo"].lower()


def test_inspecao_cordao_poroso():
    """Valida a rejeição de junta soldada com excesso de porosidade."""
    inspector = WeldGeometricInspector()
    img = cv2.imread("data/raw/vision/geometric/amostra_cordao_poroso.png")
    resultado = inspector.inspecionar_cordao(img)

    assert resultado["aprovado"] is False
    assert resultado["total_poros"] > 2
    assert "microporos" in resultado["motivo"].lower()


def test_inspecao_cordao_descontinuo():
    """Valida a rejeição por descontinuidade/fragmentação do cordão."""
    inspector = WeldGeometricInspector()
    img = cv2.imread("data/raw/vision/geometric/amostra_cordao_descontinuo.png")
    resultado = inspector.inspecionar_cordao(img)

    assert resultado["aprovado"] is False
    assert resultado["descontinuidade"] is True


def test_inspecao_termografia_haz_estavel():
    """Valida o perfil térmico nominal da Zona Afetada pelo Calor."""
    inspector = ThermalHAZInspector()
    img = cv2.imread("data/raw/vision/thermal/termografia_haz_estavel.png")
    resultado = inspector.inspecionar_termografia_haz(img)

    assert resultado["aprovado"] is True
    assert resultado["alerta_sobreaquecimento"] is False
    assert resultado["area_critica_px"] < inspector.limiar_area_critica_px


def test_inspecao_termografia_haz_critica():
    """Valida o alerta de sobreaquecimento excessivo na HAZ."""
    inspector = ThermalHAZInspector()
    img = cv2.imread("data/raw/vision/thermal/termografia_haz_critica.png")
    resultado = inspector.inspecionar_termografia_haz(img)

    assert resultado["aprovado"] is False
    assert resultado["alerta_sobreaquecimento"] is True
    assert resultado["area_critica_px"] >= inspector.limiar_area_critica_px
    assert "sobreaquecimento" in resultado["diagnostico_termico"].lower()

