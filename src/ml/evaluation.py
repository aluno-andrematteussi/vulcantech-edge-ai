"""Módulo de avaliação de desempenho, diagnóstico de falhas e explicabilidade SHAP.

Gera matrizes de confusão 4x4, métricas de Recall por modo de falha,
tabelas comparativas e análise de relevância física de atributos.
"""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import classification_report, confusion_matrix


class IndustrialModelEvaluator:
    """Implementa a validação diagnóstica dos modelos preditivos

    instalados nas células de usinagem e estampagem da planta.
    """

    CLASSES = [
        "Normal",
        "Desgaste_Ferramenta",
        "Falha_Refrigeracao",
        "Desalinhamento_Trinca",
    ]

    @staticmethod
    def avaliar_classificador(
        modelo: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        nome_modelo: str,
        exibir_grafico: bool = True,
    ) -> Dict[str, Any]:
        """Calcula métricas detalhadas e exibe a matriz de confusão 4x4.

        Args:
            modelo: Estimador treinado (RandomForest, LightGBM, etc.).
            X_test: Matriz de features do conjunto de teste temporal.
            y_test: Classes reais observadas no teste.
            nome_modelo: Nome descritivo para identificação do algoritmo.
            exibir_grafico: Se True, renderiza a figura com plt.show().

        Returns:
            Dict[str, Any]: Dicionário com métricas completas de classificação.
        """
        y_pred = modelo.predict(X_test)
        report = classification_report(
            y_test,
            y_pred,
            target_names=IndustrialModelEvaluator.CLASSES,
            output_dict=True,
        )

        print(
            f"\n================ Desempenho do Modelo: {nome_modelo} ================"
        )
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=IndustrialModelEvaluator.CLASSES,
                digits=4,
            )
        )

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=IndustrialModelEvaluator.CLASSES,
            yticklabels=IndustrialModelEvaluator.CLASSES,
            ax=ax,
        )
        ax.set_title(f"Matriz de Confusão 4x4 - {nome_modelo}")
        ax.set_xlabel("Diagnóstico Predito")
        ax.set_ylabel("Estado Real da Máquina")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()

        if exibir_grafico:
            plt.show()
        else:
            plt.close(fig)

        return report

    @staticmethod
    def comparar_desempenho(relatorios: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Estrutura uma tabela comparativa com Acurácia, Recall Macro e por falha.

        Args:
            relatorios: Dicionário associando nomes de modelos aos seus relatórios.

        Returns:
            pd.DataFrame: Tabela sintética de desempenho.
        """
        linhas = []
        for nome, rep in relatorios.items():
            linhas.append(
                {
                    "Modelo": nome,
                    "Acurácia": round(rep["accuracy"], 4),
                    "Recall Macro": round(
                        rep["macro avg"]["recall"], 4
                    ),
                    "F1-Score Macro": round(
                        rep["macro avg"]["f1-score"], 4
                    ),
                    "Recall Desgaste": round(
                        rep["Desgaste_Ferramenta"]["recall"], 4
                    ),
                    "Recall Refrigeração": round(
                        rep["Falha_Refrigeracao"]["recall"], 4
                    ),
                    "Recall Trinca": round(
                        rep["Desalinhamento_Trinca"]["recall"], 4
                    ),
                }
            )
        return pd.DataFrame(linhas)

    @staticmethod
    def plotar_feature_importance(
        modelo: Any,
        feature_names: List[str],
        top_n: int = 10,
        exibir_grafico: bool = True,
    ) -> None:
        """Gera gráfico de barras das variáveis físicas mais determinantes.

        Args:
            modelo: Estimador ajustado contendo feature_importances_.
            feature_names: Lista ordenada dos nomes dos atributos de entrada.
            top_n: Quantidade máxima de atributos exibidos.
            exibir_grafico: Se True, renderiza a figura com plt.show().
        """
        if hasattr(modelo, "feature_importances_"):
            importancias = modelo.feature_importances_
            indices = np.argsort(importancias)[-top_n:]

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.set_title("Top Variáveis Mais Relevantes (Assinatura Física)")
            ax.barh(
                range(len(indices)),
                importancias[indices],
                align="center",
                color="#008080",
            )
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel("Importância Relativa")
            fig.tight_layout()

            if exibir_grafico:
                plt.show()
            else:
                plt.close(fig)

    @staticmethod
    def gerar_explicabilidade_shap(
        modelo: Any,
        X_sample: pd.DataFrame,
        feature_names: List[str],
        max_display: int = 10,
        exibir_grafico: bool = True,
    ) -> None:
        """Gera o gráfico de interpretabilidade SHAP para validação MLOps.

        Args:
            modelo: Modelo de árvores ajustado.
            X_sample: Amostra de features para extração de valores de Shapley.
            feature_names: Lista dos nomes dos atributos.
            max_display: Quantidade de features exibidas no gráfico de resumo.
            exibir_grafico: Se True, renderiza a figura com plt.show().
        """
        try:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(X_sample)

            plt.figure(figsize=(10, 6))
            # No caso multiclasse, shap_values pode ser uma lista de arrays por classe
            if isinstance(shap_values, list):
                # Classe 1: Desgaste de Ferramenta
                shap.summary_plot(
                    shap_values[1],
                    X_sample,
                    feature_names=feature_names,
                    max_display=max_display,
                    show=False,
                )
                plt.title("Impacto SHAP - Desgaste de Ferramenta / Fuso")
            elif hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
                # API moderna do SHAP para arrays 3D (amostras, features, classes)
                shap.summary_plot(
                    shap_values.values[:, :, 1],
                    X_sample,
                    feature_names=feature_names,
                    max_display=max_display,
                    show=False,
                )
                plt.title("Impacto SHAP - Desgaste de Ferramenta / Fuso")
            else:
                shap.summary_plot(
                    shap_values,
                    X_sample,
                    feature_names=feature_names,
                    max_display=max_display,
                    show=False,
                )
                plt.title("Impacto SHAP Global das Variáveis")
            plt.tight_layout()

            if exibir_grafico:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            print(f"Nota de interpretabilidade SHAP: {str(e)}")
