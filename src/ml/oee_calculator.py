"""Módulo de cálculo preditivo do impacto operacional em OEE.

Modela o impacto das falhas preditas nos componentes de Disponibilidade,
Performance e Qualidade em conformidade com as normas da Indústria 4.0.
"""

from typing import Any, Dict, List
import pandas as pd


class IndustrialOEECalculator:
    """Calcula e simula o impacto preditivo das condições de máquinas

    no indicador global de eficiência OEE (Overall Equipment Effectiveness).
    """

    # Baselines nominais da planta da VulcanTech (condição sem falhas)
    BASELINE_DISPONIBILIDADE = 0.95
    BASELINE_PERFORMANCE = 0.95
    BASELINE_QUALIDADE = 0.98

    # Penalidades físicas associadas a cada diagnóstico predito
    PENALIDADES_FALHA = {
        0: {
            "diagnostico": "Operacao_Normal",
            "penalidade_disp": 0.00,
            "penalidade_qual": 0.00,
            "penalidade_perf": 0.00,
            "acao_clp": "MANTER_OPERACAO",
            "risco": "Baixo",
        },
        1: {
            "diagnostico": "Desgaste_Severo_Ferramenta",
            "penalidade_disp": 0.15,  # Tempo de parada para substituição
            "penalidade_qual": 0.05,  # Rugosidade superficial em usinagem
            "penalidade_perf": 0.05,  # Desaceleração para poupar a pastilha
            "acao_clp": "REDUZIR_VELOCIDADE_50%",
            "risco": "Medio",
        },
        2: {
            "diagnostico": "Falha_Refrigeracao_Superaquecimento",
            "penalidade_disp": 0.10,  # Parada técnica para desobstrução
            "penalidade_qual": 0.38,  # Distorção térmica, trincas e refugo
            "penalidade_perf": 0.05,  # Perda de avanço por superaquecimento
            "acao_clp": "REDUZIR_VELOCIDADE_50%",
            "risco": "Alto",
        },
        3: {
            "diagnostico": "Desalinhamento_Matriz_Trinca",
            "penalidade_disp": 0.50,  # Parada de emergência e manutenção pesada
            "penalidade_qual": 0.20,  # Peças fora de tolerância dimensional
            "penalidade_perf": 0.10,  # Vibração severa que impede operação plena
            "acao_clp": "INTERROMPER_EMERGENCIA",
            "risco": "Critico",
        },
    }

    def calcular_oee_instancia(self, classe_falha: int) -> Dict[str, Any]:
        """Calcula o OEE predito decorrente de um estado operacional pontual.

        Args:
            classe_falha: Código inteiro do modo de falha (0, 1, 2 ou 3).

        Returns:
            Dict[str, Any]: Indicadores de Disponibilidade, Performance,
                            Qualidade, OEE resultante e recomendação de CLP.
        """
        dados_falha = self.PENALIDADES_FALHA.get(
            classe_falha, self.PENALIDADES_FALHA[0]
        )

        disp = max(
            0.0,
            self.BASELINE_DISPONIBILIDADE - dados_falha["penalidade_disp"],
        )
        perf = max(
            0.0,
            self.BASELINE_PERFORMANCE - dados_falha["penalidade_perf"],
        )
        qual = max(
            0.0,
            self.BASELINE_QUALIDADE - dados_falha["penalidade_qual"],
        )

        oee = disp * perf * qual

        oee_nominal = (
            self.BASELINE_DISPONIBILIDADE
            * self.BASELINE_PERFORMANCE
            * self.BASELINE_QUALIDADE
        )
        perda_oee_pct = (oee_nominal - oee) * 100.0

        return {
            "classe_falha": classe_falha,
            "diagnostico": dados_falha["diagnostico"],
            "disponibilidade": round(disp, 4),
            "performance": round(perf, 4),
            "qualidade": round(qual, 4),
            "oee": round(oee, 4),
            "oee_nominal": round(oee_nominal, 4),
            "perda_oee_pontos_percentuais": round(perda_oee_pct, 2),
            "acao_clp": dados_falha["acao_clp"],
            "nivel_risco": dados_falha["risco"],
        }

    def simular_impacto_lote(
        self, predicoes: pd.Series
    ) -> Dict[str, Any]:
        """Simula o impacto em OEE acumulado ao longo de um histórico de predições.

        Args:
            predicoes: Série temporal contendo as classes preditas pelo modelo.

        Returns:
            Dict[str, Any]: Resumo das perdas de OEE e distribuição de ações.
        """
        resultados = [
            self.calcular_oee_instancia(int(p)) for p in predicoes
        ]
        df_resultados = pd.DataFrame(resultados)

        oee_medio = df_resultados["oee"].mean()
        oee_nominal = (
            self.BASELINE_DISPONIBILIDADE
            * self.BASELINE_PERFORMANCE
            * self.BASELINE_QUALIDADE
        )

        acoes_distribuicao = (
            df_resultados["acao_clp"].value_counts().to_dict()
        )

        return {
            "total_amostras": len(predicoes),
            "oee_medio_operacional": round(oee_medio, 4),
            "oee_nominal_referencia": round(oee_nominal, 4),
            "gap_oee_medio": round((oee_nominal - oee_medio) * 100, 2),
            "distribuicao_acoes_clp": acoes_distribuicao,
            "distribuicao_risco": df_resultados["nivel_risco"].value_counts().to_dict(),
        }

