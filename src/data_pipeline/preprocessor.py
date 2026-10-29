"""Módulo de saneamento e tratamento de dados industriais da VulcanTech.

Responsável pela ingestão estruturada, detecção de falhas de sensores,
limpeza de ruídos eletromagnéticos e interpolação temporal condicionada.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class IndustrialDataSanitizer:
    """Implementa o saneamento robusto e a interpolação temporal dos sensores

    instalados nas células de manufatura da VulcanTech Smart Metals S.A.
    """

    # Limites físicos absolutos de engenharia para filtragem de anomalias
    LIMITES_FISICOS = {
        "temperatura_fuso_c": {"min": 0.0, "max": 200.0},
        "vibracao_triaxial_rms_mms": {"min": 0.0, "max": 25.0},
        "torque_fuso_nm": {"min": 0.0, "max": 400.0},
        "vazao_refrigerante_lmin": {"min": 0.0, "max": 60.0},
        "corrente_estator_amp": {"min": 0.0, "max": 100.0},
        "pressao_hidraulica_bar": {"min": 0.0, "max": 300.0},
        "nivel_ruido_db": {"min": 20.0, "max": 150.0},
    }

    COLUNAS_NUMERICAS = [
        "temperatura_fuso_c",
        "vibracao_triaxial_rms_mms",
        "torque_fuso_nm",
        "vazao_refrigerante_lmin",
        "corrente_estator_amp",
        "pressao_hidraulica_bar",
        "nivel_ruido_db",
    ]

    def __init__(self, caminho_ficheiro: str) -> None:
        """Inicializa o sanitizador com o caminho do arquivo de entrada.

        Args:
            caminho_ficheiro: Caminho do arquivo CSV de telemetria industrial.
        """
        self.caminho_ficheiro = caminho_ficheiro
        self.df: pd.DataFrame = pd.DataFrame()

    def carregar_dados(self) -> pd.DataFrame:
        """Carrega os dados brutos e normaliza carimbos temporais.

        Returns:
            pd.DataFrame: DataFrame ordenado cronologicamente por célula.
        """
        self.df = pd.read_csv(self.caminho_ficheiro)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df.sort_values(
            by=["id_celula", "timestamp"], inplace=True
        )
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def inspecionar_estrutura(self) -> Dict[str, Any]:
        """Avalia taxas de ausência de dados (NaNs) agrupadas por célula

        e liga metálica, além da consistência amostral de 5 minutos.

        Returns:
            Dict[str, Any]: Relatório estruturado de integridade de dados.
        """
        if self.df.empty:
            self.carregar_dados()

        # Taxas de NaNs gerais
        nulos_globais = self.df.isnull().sum().to_dict()

        # Taxas de NaNs agrupadas por célula de manufatura
        nulos_por_celula = (
            self.df.groupby("id_celula")[self.COLUNAS_NUMERICAS]
            .apply(lambda g: g.isnull().sum())
            .to_dict()
        )

        # Taxas de NaNs agrupadas por liga metálica processada
        nulos_por_liga = (
            self.df.groupby("liga_metalica")[self.COLUNAS_NUMERICAS]
            .apply(lambda g: g.isnull().sum())
            .to_dict()
        )

        # Verificação do intervalo temporal nominal de 5 minutos
        desvios_amostragem: Dict[str, int] = {}
        for celula, grupo in self.df.groupby("id_celula"):
            deltas = grupo["timestamp"].diff().dropna()
            inconsistencias = int((deltas != pd.Timedelta(minutes=5)).sum())
            desvios_amostragem[celula] = inconsistencias

        return {
            "total_registos": len(self.df),
            "celulas": self.df["id_celula"].unique().tolist(),
            "ligas_metalicas": self.df["liga_metalica"].unique().tolist(),
            "nulos_globais": nulos_globais,
            "nulos_por_celula": nulos_por_celula,
            "nulos_por_liga": nulos_por_liga,
            "desvios_amostragem_5min": desvios_amostragem,
        }

    def filtrar_ruidos_sensores(self) -> pd.DataFrame:
        """Converte códigos de falha de hardware e leituras fora

        dos limites operacionais físicos em valores nulos (NaN).

        Returns:
            pd.DataFrame: DataFrame com ruídos de sensor convertidos em NaN.
        """
        # Conversão explícita de códigos de erro documentados no sistema
        # 1. Temperatura aberta/descalibrada: 999.9 °C
        self.df.loc[
            self.df["temperatura_fuso_c"] >= 900.0,
            "temperatura_fuso_c",
        ] = np.nan

        # 2. Vibração triaxial em saturação negativa: -99.0 mm/s
        self.df.loc[
            self.df["vibracao_triaxial_rms_mms"] < 0.0,
            "vibracao_triaxial_rms_mms",
        ] = np.nan

        # 3. Pressão hidráulica em vácuo/erro: -1.0 bar
        self.df.loc[
            self.df["pressao_hidraulica_bar"] < 0.0,
            "pressao_hidraulica_bar",
        ] = np.nan

        # 4. Vazão reversa não física de fluido de corte
        self.df.loc[
            self.df["vazao_refrigerante_lmin"] < 0.0,
            "vazao_refrigerante_lmin",
        ] = np.nan

        # Aplicação dos limites físicos operacionais absolutos
        for coluna, limites in self.LIMITES_FISICOS.items():
            if coluna in self.df.columns:
                invalido = (self.df[coluna] < limites["min"]) | (
                    self.df[coluna] > limites["max"]
                )
                self.df.loc[invalido, coluna] = np.nan

        return self.df

    def aplicar_zscore_condicional_liga(
        self, limiar_z: float = 4.5
    ) -> pd.DataFrame:
        """Aplica Z-Score condicional por liga metálica para expurgar ruídos

        estocásticos severos mantendo preservados os padrões reais de falha.

        Args:
            limiar_z: Quantidade de desvios-padrão para classificação de ruído.

        Returns:
            pd.DataFrame: DataFrame com ruídos estocásticos isolados.
        """
        for coluna in self.COLUNAS_NUMERICAS:
            # Calcula média e desvio padrão condicionados à liga
            medias = self.df.groupby("liga_metalica")[coluna].transform("mean")
            desvios = self.df.groupby("liga_metalica")[coluna].transform("std")

            # Evita divisão por desvio nulo
            desvios = desvios.replace(0, np.nan)
            z_score = ((self.df[coluna] - medias) / desvios).abs()

            # Apenas pontos operacionais estritamente fora do regime de processo
            # e com dispersão estatística extrema são saneados
            ruido_estatistico = (z_score > limiar_z) & (
                self.df["tipo_falha"] == 0
            )
            self.df.loc[ruido_estatistico, coluna] = np.nan

        return self.df

    def interpolar_series_temporais(
        self, metodo: str = "time"
    ) -> pd.DataFrame:
        """Preenche lacunas operacionais utilizando interpolação temporal

        condicionada a cada célula flexível de manufatura (id_celula).

        Args:
            metodo: Método de interpolação do pandas ('time' recomendado).

        Returns:
            pd.DataFrame: DataFrame com continuidade temporal restaurada.
        """
        # Garante ordenação cronológica estrita antes da reconstrução contínua
        self.df.sort_values(
            by=["id_celula", "timestamp"], inplace=True
        )

        dfs_processados: List[pd.DataFrame] = []
        for _, grupo in self.df.groupby("id_celula"):
            grupo_temp = grupo.copy().set_index("timestamp")
            grupo_temp[self.COLUNAS_NUMERICAS] = grupo_temp[
                self.COLUNAS_NUMERICAS
            ].interpolate(method=metodo, limit_direction="both")
            dfs_processados.append(grupo_temp.reset_index())

        self.df = pd.concat(dfs_processados, ignore_index=True)
        return self.df

    def executar_pipeline_saneamento(self) -> pd.DataFrame:
        """Executa a sequência completa de saneamento de dados industriais.

        Returns:
            pd.DataFrame: DataFrame limpo e estável para engenharia de features.
        """
        self.carregar_dados()
        self.filtrar_ruidos_sensores()
        self.aplicar_zscore_condicional_liga()
        self.interpolar_series_temporais()
        return self.df
