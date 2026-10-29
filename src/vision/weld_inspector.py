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
        altura_orig, largura_orig = imagem_bgr.shape[:2]
        fator_escala = (
            400.0 / float(largura_orig) if largura_orig != 400 else 1.0
        )
        if fator_escala != 1.0:
            h_proc = int(round(altura_orig * fator_escala))
            imagem_trabalho = cv2.resize(
                imagem_bgr, (400, h_proc), interpolation=cv2.INTER_AREA
            )
        else:
            imagem_trabalho = imagem_bgr

        # 1. Filtro de nitidez
        nitida = self.aplicar_filtro_nitidez(imagem_trabalho)

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
            mascara_nula = cv2.resize(
                bordas, (largura_orig, altura_orig), interpolation=cv2.INTER_NEAREST
            )
            return {
                "aprovado": False,
                "motivo": "Nenhum contorno identificado na junta soldada",
                "total_poros": 0,
                "descontinuidade": True,
                "largura_media_px": 0.0,
                "comprimento_cordao_px": 0.0,
                "mascara_bordas": mascara_nula,
                "imagem_anotada": anotada,
            }

        # Identificação de microporosidades internas (cavidades escuras e circulares)
        poros_identificados: List[Tuple[float, Tuple[int, int]]] = []
        caixas_poros: List[Tuple[int, int]] = []

        for c in contornos:
            area = float(cv2.contourArea(c))
            if self.min_area_poro_px <= area <= self.max_area_poro_px:
                peri = cv2.arcLength(c, True)
                circ = (
                    (4.0 * np.pi * area) / (peri * peri)
                    if peri > 0.0
                    else 0.0
                )
                if circ >= 0.38:
                    mask_poro = np.zeros(cinza.shape, dtype=np.uint8)
                    cv2.drawContours(mask_poro, [c], -1, 255, -1)
                    lum_media = cv2.mean(cinza, mask=mask_poro)[0]
                    if lum_media < 90.0:
                        (x, y), raio = cv2.minEnclosingCircle(c)
                        pt = (int(x), int(y))
                        if not any(
                            abs(pt[0] - px) < 6 and abs(pt[1] - py) < 6
                            for px, py in caixas_poros
                        ):
                            caixas_poros.append(pt)
                            poros_identificados.append((area, pt))
                            pt_orig = (
                                int(round(x / fator_escala)),
                                int(round(y / fator_escala)),
                            )
                            r_orig = int(round(raio / fator_escala))
                            cv2.circle(
                                anotada, pt_orig, r_orig + 4, (0, 0, 255), 2
                            )

        # Identificação de descontinuidades na geometria contínua do cordão
        segmentos_cordao = self._deduplicar_segmentos_cordao(
            contornos, min_area=1500.0
        )
        if not segmentos_cordao:
            kernel_fusao = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
            bordas_fundidas = cv2.morphologyEx(
                bordas, cv2.MORPH_CLOSE, kernel_fusao
            )
            contornos_fundidos, _ = cv2.findContours(
                bordas_fundidas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            segmentos_cordao = self._deduplicar_segmentos_cordao(
                contornos_fundidos, min_area=1000.0
            )

        descontinuidade = False
        largura_media = 0.0
        comprimento_total = 0.0

        if len(segmentos_cordao) > 1:
            descontinuidade = True
            maior = segmentos_cordao[0]
            _, _, w, h = cv2.boundingRect(maior)
            comprimento_total = float(max(w, h)) / fator_escala
            largura_media = float(min(w, h)) / fator_escala
            for seg in segmentos_cordao:
                seg_rescaled = (seg / fator_escala).astype(np.int32)
                cv2.drawContours(anotada, [seg_rescaled], -1, (0, 0, 255), 2)
        elif len(segmentos_cordao) == 1:
            maior = segmentos_cordao[0]
            _, _, w, h = cv2.boundingRect(maior)
            comprimento_total = float(max(w, h)) / fator_escala
            largura_media = float(min(w, h)) / fator_escala

            if (
                comprimento_total * fator_escala < self.min_comprimento_cordao_px
                or largura_media * fator_escala < self.min_largura_cordao_px
            ):
                descontinuidade = True
                seg_rescaled = (maior / fator_escala).astype(np.int32)
                cv2.drawContours(anotada, [seg_rescaled], -1, (0, 0, 255), 2)
            else:
                seg_rescaled = (maior / fator_escala).astype(np.int32)
                cv2.drawContours(anotada, [seg_rescaled], -1, (0, 255, 0), 2)
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
        if not motivos:
            motivos.append("Cordao de solda conforme parametros nominais")

        mascara_completa = cv2.resize(
            bordas, (largura_orig, altura_orig), interpolation=cv2.INTER_NEAREST
        )

        return {
            "aprovado": aprovado,
            "motivo": " | ".join(motivos),
            "total_poros": total_poros,
            "descontinuidade": descontinuidade,
            "largura_media_px": round(largura_media, 1),
            "comprimento_cordao_px": round(comprimento_total, 1),
            "mascara_bordas": mascara_completa,
            "imagem_anotada": anotada,
        }
