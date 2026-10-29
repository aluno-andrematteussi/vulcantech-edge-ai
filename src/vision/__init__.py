"""Módulo de visão computacional (CVOps) para inspeção óptica e térmica de solda."""

from src.vision.generate_samples import gerar_amostras_soldagem
from src.vision.thermal_haz_inspector import ThermalHAZInspector
from src.vision.weld_inspector import WeldGeometricInspector

__all__ = [
    "WeldGeometricInspector",
    "ThermalHAZInspector",
    "gerar_amostras_soldagem",
]

