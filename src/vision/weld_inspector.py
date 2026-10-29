"""Módulo de inspeção geométrica de cordões de solda robotizados MIG/MAG.

Aplica filtros de nitidez, conversão para escala de cinza e o algoritmo
Canny Edge Detection para quantificar porosidade e descontinuidade geométrica.
"""

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


class WeldGeometricInspector:
    """Inspeciona visualmente cordões de solda utilizando visão computacional

    clássica para certificar a conformidade dimensional e ausência de poros.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        blur_ksize: Tuple[int, int] = (5, 5),
        max_poros_permitidos: int = 2,
        min_area_poro_px: float = 6.0,
        max_area_poro_px: float = 120.0,
        min_largura_cordao_px: float = 15.0,
        min_comprimento_cordao_px: float = 240.0,
    ) -> None:
        """Inicializa os parâmetros do algoritmo de inspeção óptica de solda.

        Args:
            canny_low: Limiar inferior para o detector de bordas Canny.
            canny_high: Limiar superior para o detector de bordas Canny.
            blur_ksize: Dimensão do kernel para suavização Gaussiana.
            max_poros_permitidos: Quantidade máxima tolerada de microporos.
            min_area_poro_px: Área mínima em pixels para caracterizar poro.
            max_area_poro_px: Área máxima em pixels para caracterizar poro.
            min_largura_cordao_px: Largura mínima aceitável do cordão.
            min_comprimento_cordao_px: Comprimento contínuo mínimo esperado.
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_ksize = blur_ksize
        self.max_poros_permitidos = max_poros_permitidos
        self.min_area_poro_px = min_area_poro_px
        self.max_area_poro_px = max_area_poro_px
        self.min_largura_cordao_px = min_largura_cordao_px
        self.min_comprimento_cordao_px = min_comprimento_cordao_px

        # Kernel 3x3 de convolução para realce de nitidez de bordas
        self.kernel_nitidez = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
        )

    def aplicar_filtro_nitidez(self, imagem_bgr: np.ndarray) -> np.ndarray:
        """Aplica filtro de realce de nitidez na imagem de entrada.

        Args:
            imagem_bgr: Imagem original no espaço de cores BGR.

        Returns:
            np.ndarray: Imagem filtrada com contraste de borda elevado.
        """
        return cv2.filter2D(imagem_bgr, -1, self.kernel_nitidez)

    def _deduplicar_segmentos_cordao(
        self, contornos: List[np.ndarray], min_area: float = 2000.0
    ) -> List[np.ndarray]:
        """Isola contornos primários do cordão eliminando anéis concêntricos Canny.

        Args:
            contornos: Lista de contornos extraídos da imagem.
            min_area: Área mínima para caracterizar corpo do cordão de solda.

        Returns:
            List[np.ndarray]: Contornos únicos desduplicados do cordão.
        """
        grandes = [c for c in contornos if cv2.contourArea(c) >= min_area]
        caixas_unicas: List[Tuple[int, int, int, int]] = []
        segmentos_unicos: List[np.ndarray] = []

        for c in sorted(grandes, key=cv2.contourArea, reverse=True):
            x, y, w, h = cv2.boundingRect(c)
            duplicado = False
            for ux, uy, uw, uh in caixas_unicas:
                if (
                    abs(x - ux) < 15
                    and abs(y - uy) < 15
                    and abs(w - uw) < 20
                    and abs(h - uh) < 20
                ):
                    duplicado = True
                    break
            if not duplicado:
                caixas_unicas.append((x, y, w, h))
                segmentos_unicos.append(c)

        return segmentos_unicos

    def inspecionar_cordao(
        self, imagem_bgr: np.ndarray
    ) -> Dict[str, Any]:
        """Processa o cordão de solda e diagnostica porosidade e descontinuidades.

        Args:
            imagem_bgr: Imagem óptica da junta soldada.

        Returns:
            Dict[str, Any]: Diagnóstico dimensional, métricas e máscaras.
        """
        # 1. Filtro de nitidez
        nitida = self.aplicar_filtro_nitidez(imagem_bgr)

        # 2. Conversão para tons de cinza e suavização
        cinza = cv2.cvtColor(nitida, cv2.COLOR_BGR2GRAY)
        suavizada = cv2.GaussianBlur(cinza, self.blur_ksize, 0)

        # 3. Detecção de bordas via algoritmo Canny
        bordas = cv2.Canny(suavizada, self.canny_low, self.canny_high)

        # 4. Análise morfológica de contornos
        contornos, _ = cv2.findContours(
            bordas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        anotada = imagem_bgr.copy()

        if not contornos:
            return {
                "aprovado": False,
                "motivo": "Nenhum contorno identificado na junta soldada",
                "total_poros": 0,
                "descontinuidade": True,
                "largura_media_px": 0.0,
                "comprimento_cordao_px": 0.0,
                "mascara_bordas": bordas,
                "imagem_anotada": anotada,
            }

        # Identificação de microporosidades internas
        poros_identificados: List[Tuple[float, Tuple[int, int]]] = []
        caixas_poros: List[Tuple[int, int]] = []

        for c in contornos:
            area = float(cv2.contourArea(c))
            if self.min_area_poro_px <= area <= self.max_area_poro_px:
                (x, y), raio = cv2.minEnclosingCircle(c)
                pt = (int(x), int(y))
                # Evita duplicar anéis internos do Canny
                if not any(
                    abs(pt[0] - px) < 5 and abs(pt[1] - py) < 5
                    for px, py in caixas_poros
                ):
                    caixas_poros.append(pt)
                    poros_identificados.append((area, pt))
                    cv2.circle(anotada, pt, int(raio) + 3, (0, 0, 255), 2)

        # Identificação de descontinuidades na geometria contínua do cordão
        segmentos_cordao = self._deduplicar_segmentos_cordao(contornos)

        descontinuidade = False
        largura_media = 0.0
        comprimento_total = 0.0

        if len(segmentos_cordao) > 1:
            # Fragmentação do cordão de solda (interrupção de arco/descontinuidade)
            descontinuidade = True
            maior = max(segmentos_cordao, key=cv2.contourArea)
            _, _, w, h = cv2.boundingRect(maior)
            comprimento_total = float(max(w, h))
            largura_media = float(min(w, h))
            for seg in segmentos_cordao:
                cv2.drawContours(anotada, [seg], -1, (0, 0, 255), 2)
        elif len(segmentos_cordao) == 1:
            maior = segmentos_cordao[0]
            _, _, w, h = cv2.boundingRect(maior)
            comprimento_total = float(max(w, h))
            largura_media = float(min(w, h))

            if (
                comprimento_total < self.min_comprimento_cordao_px
                or largura_media < self.min_largura_cordao_px
            ):
                descontinuidade = True
                cv2.drawContours(anotada, [maior], -1, (0, 0, 255), 2)
            else:
                cv2.drawContours(anotada, [maior], -1, (0, 255, 0), 2)
        else:
            descontinuidade = True

        total_poros = len(poros_identificados)
        tolerancia_poros = total_poros <= self.max_poros_permitidos
        tolerancia_geometrica = not descontinuidade

        aprovado = bool(tolerancia_poros and tolerancia_geometrica)

        motivos: List[str] = []
        if not tolerancia_poros:
            motivos.append(f"Excesso de microporos ({total_poros} detectados)")
        if descontinuidade:
            motivos.append("Descontinuidade ou largura inadequada no cordao")

        motivo_final = (
            "Cordao conforme com especificacao"
            if aprovado
            else " | ".join(motivos)
        )

        return {
            "aprovado": aprovado,
            "motivo": motivo_final,
            "total_poros": total_poros,
            "descontinuidade": descontinuidade,
            "largura_media_px": round(largura_media, 2),
            "comprimento_cordao_px": round(comprimento_total, 2),
            "mascara_bordas": bordas,
            "imagem_anotada": anotada,
        }
