"""Utilitário gerador de amostras sintéticas de solda para validação de CVOps.

Popula os diretórios data/raw/vision/geometric e data/raw/vision/thermal
com casos de teste representativos de juntas soldadas e termografia HAZ.
"""

from pathlib import Path
import cv2
import numpy as np


def gerar_amostras_soldagem() -> None:
    """Gera e grava em disco imagens de teste geométricas e termográficas."""
    base_dir = Path(__file__).resolve().parents[2]
    dir_geom = base_dir / "data" / "raw" / "vision" / "geometric"
    dir_therm = base_dir / "data" / "raw" / "vision" / "thermal"

    dir_geom.mkdir(parents=True, exist_ok=True)
    dir_therm.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Amostra Geométrica Conforme (Cordão contínuo e uniforme)
    # ---------------------------------------------------------
    geom_ok = np.full((200, 400, 3), (110, 110, 110), dtype=np.uint8)
    # Placa base com chanfro
    cv2.line(geom_ok, (20, 100), (380, 100), (80, 80, 80), 30)
    # Cordão de solda metálico
    cv2.ellipse(geom_ok, (200, 100), (160, 20), 0, 0, 360, (200, 200, 200), -1)
    cv2.imwrite(str(dir_geom / "amostra_cordao_conforme.png"), geom_ok)

    # ---------------------------------------------------------
    # 2. Amostra Geométrica com Porosidade (Presença de microporos)
    # ---------------------------------------------------------
    geom_poroso = geom_ok.copy()
    # Pores (microporos circulares escuros aprisionados no cordão)
    poros = [(120, 95, 4), (160, 102, 5), (210, 98, 4), (270, 105, 5), (310, 96, 4)]
    for x, y, r in poros:
        cv2.circle(geom_poroso, (x, y), r, (30, 30, 30), -1)
    cv2.imwrite(str(dir_geom / "amostra_cordao_poroso.png"), geom_poroso)

    # ---------------------------------------------------------
    # 3. Amostra Geométrica com Descontinuidade (Cordão interrompido)
    # ---------------------------------------------------------
    geom_descontinuo = np.full((200, 400, 3), (110, 110, 110), dtype=np.uint8)
    cv2.line(geom_descontinuo, (20, 100), (380, 100), (80, 80, 80), 30)
    # Segmento esquerdo do cordão
    cv2.ellipse(
        geom_descontinuo, (100, 100), (60, 18), 0, 0, 360, (200, 200, 200), -1
    )
    # Falha de arco / interrupção no meio (x=160 a 240)
    # Segmento direito do cordão
    cv2.ellipse(
        geom_descontinuo, (300, 100), (60, 18), 0, 0, 360, (200, 200, 200), -1
    )
    cv2.imwrite(str(dir_geom / "amostra_cordao_descontinuo.png"), geom_descontinuo)

    # ---------------------------------------------------------
    # 4. Amostra Termográfica Estável (HAZ controlada)
    # ---------------------------------------------------------
    # Gradiente de fundo frio (azul escuro/violeta em BGR)
    therm_ok = np.full((200, 400, 3), (120, 30, 20), dtype=np.uint8)
    # Zona quente moderada da poça (vermelho contido)
    cv2.ellipse(therm_ok, (200, 100), (70, 14), 0, 0, 360, (0, 0, 240), -1)
    cv2.ellipse(therm_ok, (200, 100), (40, 8), 0, 0, 360, (0, 200, 255), -1)
    cv2.imwrite(str(dir_therm / "termografia_haz_estavel.png"), therm_ok)

    # ---------------------------------------------------------
    # 5. Amostra Termográfica Crítica (Sobreaquecimento excessivo)
    # ---------------------------------------------------------
    therm_critica = np.full((200, 400, 3), (120, 30, 20), dtype=np.uint8)
    # Zona HAZ excessivamente dilatada em alta temperatura (vermelho/laranja)
    cv2.ellipse(therm_critica, (200, 100), (150, 45), 0, 0, 360, (0, 0, 255), -1)
    cv2.ellipse(
        therm_critica, (200, 100), (100, 28), 0, 0, 360, (0, 180, 255), -1
    )
    cv2.imwrite(str(dir_therm / "termografia_haz_critica.png"), therm_critica)

    print(
        f"Amostras de visao geradas com sucesso em:\n- {dir_geom}\n- {dir_therm}"
    )


if __name__ == "__main__":
    gerar_amostras_soldagem()
