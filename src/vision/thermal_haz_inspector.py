"""Módulo de análise termográfica infravermelha da Zona Afetada pelo Calor (HAZ).

Segmenta no espaço de cores HSV as zonas críticas de sobreaquecimento
durante os ciclos de solda robotizada MIG/MAG nas ligas metálicas.
"""

from typing import Any, Dict, Tuple
import cv2
import numpy as np


class ThermalHAZInspector:
    """Inspeciona termografias industriais e delimita a Zona Afetada pelo Calor

    (HAZ - Heat Affected Zone) para prevenir fragilização e trincas térmicas.
    """

    def __init__(
        self,
        lower_hsv1: Tuple[int, int, int] = (0, 90, 90),
        upper_hsv1: Tuple[int, int, int] = (25, 255, 255),
        lower_hsv2: Tuple[int, int, int] = (160, 90, 90),
        upper_hsv2: Tuple[int, int, int] = (180, 255, 255),
        limiar_area_critica_px: float = 8000.0,
    ) -> None:
        """Inicializa os limiares de matiz térmico no espaço HSV.

        Args:
            lower_hsv1: Limiar inferior da primeira faixa de tons quentes
                (vermelho/laranja).
            upper_hsv1: Limiar superior da primeira faixa de tons quentes.
            lower_hsv2: Limiar inferior da segunda faixa de tons quentes
                (vermelho puro).
            upper_hsv2: Limiar superior da segunda faixa de tons quentes.
            limiar_area_critica_px: Área máxima acumulada aceitável de sobreaquecimento.
        """
        self.lower_hsv1 = np.array(lower_hsv1, dtype=np.uint8)
        self.upper_hsv1 = np.array(upper_hsv1, dtype=np.uint8)
        self.lower_hsv2 = np.array(lower_hsv2, dtype=np.uint8)
        self.upper_hsv2 = np.array(upper_hsv2, dtype=np.uint8)
        self.limiar_area_critica_px = limiar_area_critica_px

    def inspecionar_termografia_haz(
        self, imagem_bgr: np.ndarray
    ) -> Dict[str, Any]:
        """Processa a termografia em HSV e segmenta zonas térmicas anômalas.

        Args:
            imagem_bgr: Imagem termográfica no formato BGR.

        Returns:
            Dict[str, Any]: Métricas de área da HAZ, contornos e diagnóstico.
        """
        # Conversão para o espaço cromático HSV
        hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

        # Limiarização dupla para isolar os picos de calor da poça de fusão
        mascara1 = cv2.inRange(hsv, self.lower_hsv1, self.upper_hsv1)
        mascara2 = cv2.inRange(hsv, self.lower_hsv2, self.upper_hsv2)
        mascara_termica = cv2.bitwise_or(mascara1, mascara2)

        # Filtragem morfológica para consolidação da região HAZ
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mascara_limpa = cv2.morphologyEx(
            mascara_termica, cv2.MORPH_OPEN, kernel
        )
        mascara_limpa = cv2.morphologyEx(
            mascara_limpa, cv2.MORPH_CLOSE, kernel
        )

        contornos, _ = cv2.findContours(
            mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        anotada = imagem_bgr.copy()
        focos_validos = []
        area_total_haz = 0.0

        for c in contornos:
            area = float(cv2.contourArea(c))
            if area >= 20.0:
                focos_validos.append(c)
                area_total_haz += area
                cv2.drawContours(anotada, [c], -1, (0, 0, 255), 2)

        alerta_sobreaquecimento = bool(
            area_total_haz >= self.limiar_area_critica_px
        )

        diagnostico = (
            "Perfil termografico conforme (HAZ controlada)"
            if not alerta_sobreaquecimento
            else "Sobreaquecimento critico na HAZ (Risco de trinca termica)"
        )

        return {
            "aprovado": not alerta_sobreaquecimento,
            "alerta_sobreaquecimento": alerta_sobreaquecimento,
            "area_critica_px": round(area_total_haz, 2),
            "focos_detectados": len(focos_validos),
            "diagnostico_termico": diagnostico,
            "mascara_termica": mascara_limpa,
            "imagem_anotada": anotada,
        }
