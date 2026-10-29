"""Módulo de Engenharia de Features Industriais da VulcanTech.

Implementa a criação do Índice de Esforço Mecânico (IEM), normalização
condicionada por liga metálica e extração de estatísticas móveis dinâmicas.
"""

import os
from typing import Dict, List
import pandas as pd


class IndustrialFeatureEngineer:
    """Implementa a extração de variáveis dinâmicas e indicadores

    físicos multivariados a partir dos sinais estabilizados das células.
    """

    VARIAVEIS_ESTATISTICAS_MOVEIS = [
        "vibracao_triaxial_rms_mms",
        "temperatura_fuso_c",
        "torque_fuso_nm",
        "corrente_estator_amp",
        "indice_esforco_mecanico",
    ]

    VARIAVEIS_NORMALIZACAO_LIGA = [
        "torque_fuso_nm",
        "corrente_estator_amp",
        "vibracao_triaxial_rms_mms",
        "temperatura_fuso_c",
        "indice_esforco_mecanico",
    ]

    def __init__(self, df: pd.DataFrame) -> None:
        """Inicializa o engenheiro de features com o dataframe saneado.

        Args:
            df: DataFrame saneado e ordenado temporalmente por célula.
        """
        self.df = df.copy()

    def calcular_indice_esforco_mecanico(self) -> pd.DataFrame:
        """Calcula o Índice de Esforço Mecânico (IEM):

        IEM = (Torque * Vibracao_RMS) / (Vazao_Refrigerante + 1.0)

        Returns:
            pd.DataFrame: DataFrame acrescido da coluna 'indice_esforco_mecanico'.
        """
        vazao_segura = self.df["vazao_refrigerante_lmin"].clip(lower=0.0)
        self.df["indice_esforco_mecanico"] = (
            self.df["torque_fuso_nm"] * self.df["vibracao_triaxial_rms_mms"]
        ) / (vazao_segura + 1.0)
        return self.df

    def normalizar_por_liga_metalica(self) -> pd.DataFrame:
        """Normaliza as variáveis dividindo-as pela média histórica

        da respectiva liga_metalica, mitigando alarmes falsos em ligas
        de alta resistência e dureza (ex.: Titânio Ti-6Al-4V).

        Returns:
            pd.DataFrame: DataFrame com variáveis normalizadas por lote/material.
        """
        for var in self.VARIAVEIS_NORMALIZACAO_LIGA:
            media_por_liga = self.df.groupby("liga_metalica")[var].transform(
                "mean"
            )
            # Evita divisão por zero
            media_por_liga = media_por_liga.replace(0, 1.0)
            self.df[f"{var}_norm_liga"] = self.df[var] / media_por_liga

        return self.df

    def adicionar_estatisticas_moveis(self) -> pd.DataFrame:
        """Calcula médias móveis (Rolling Mean) e desvios padrão (Rolling Std)

        em janelas temporais de 30 minutos (6 passos de 5 min) e
        2 horas (24 passos de 5 min) agrupadas por célula de manufatura.

        Returns:
            pd.DataFrame: DataFrame enriquecido com estatísticas temporais.
        """
        # Intervalo amostral nominal de 5 minutos:
        # 30 minutos = 6 observações
        # 2 horas (120 minutos) = 24 observações
        janelas: Dict[str, int] = {"30min": 6, "2h": 24}

        self.df.sort_values(
            by=["id_celula", "timestamp"], inplace=True
        )

        for nome_janela, passos in janelas.items():
            for var in self.VARIAVEIS_ESTATISTICAS_MOVEIS:
                # Rolling Mean
                col_media = f"{var}_rolling_mean_{nome_janela}"
                self.df[col_media] = (
                    self.df.groupby("id_celula")[var]
                    .transform(
                        lambda s: s.rolling(
                            window=passos, min_periods=1
                        ).mean()
                    )
                )

                # Rolling Std
                col_desvio = f"{var}_rolling_std_{nome_janela}"
                self.df[col_desvio] = (
                    self.df.groupby("id_celula")[var]
                    .transform(
                        lambda s: s.rolling(
                            window=passos, min_periods=1
                        ).std()
                    )
                    .fillna(0.0)
                )

        return self.df

    def executar_pipeline(self) -> pd.DataFrame:
        """Executa a sequência completa de engenharia de atributos industriais.

        Returns:
            pd.DataFrame: Conjunto de dados final processado.
        """
        self.calcular_indice_esforco_mecanico()
        self.normalizar_por_liga_metalica()
        self.adicionar_estatisticas_moveis()
        return self.df

    def salvar_dados_processados(
        self,
        caminho_saida: str = "data/processed/features_eng.parquet",
    ) -> str:
        """Salva os dados enriquecidos em formato Parquet para alta performance.

        Args:
            caminho_saida: Caminho do arquivo de destino.

        Returns:
            str: Caminho do arquivo gravado.
        """
        pasta_destino = os.path.dirname(caminho_saida)
        if pasta_destino:
            os.makedirs(pasta_destino, exist_ok=True)

        self.df.to_parquet(caminho_saida, index=False)
        return caminho_saida

