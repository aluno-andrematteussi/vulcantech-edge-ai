"""Módulo de treinamento de modelos preditivos e MLOps da VulcanTech.

Orquestra a divisão temporal estrita, o balanceamento de classes com SMOTE,
a busca hiperparamétrica dos ensembles e a serialização dos artefatos.
"""

import os
from typing import Any, Dict, List, Tuple
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler


class IndustrialModelTrainer:
    """Responsável pelo particionamento temporal, balanceamento e

    treinamento dos modelos de machine learning multiclasse da planta.
    """

    def __init__(
        self,
        caminho_dados: str = "data/processed/features_eng.parquet",
    ) -> None:
        """Inicializa o orquestrador de treinamento.

        Args:
            caminho_dados: Caminho para o conjunto de dados pré-processado.
        """
        self.caminho_dados = caminho_dados
        self.df: pd.DataFrame = pd.DataFrame()
        self.X_train: pd.DataFrame = pd.DataFrame()
        self.X_test: pd.DataFrame = pd.DataFrame()
        self.y_train: pd.Series = pd.Series(dtype=int)
        self.y_test: pd.Series = pd.Series(dtype=int)
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []

    def preparar_dados(
        self,
        target_col: str = "tipo_falha",
        proporcao_treino: float = 0.7,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Carrega e aplica a divisão temporal sem embaralhamento (shuffle=False).

        Ajusta o RobustScaler exclusivamente sobre a fatia de treinamento
        para prevenir qualquer vazamento de dados (*data leakage*).

        Args:
            target_col: Coluna alvo com os códigos multiclasse de falha.
            proporcao_treino: Proporção cronológica reservada para treino.

        Returns:
            Tuple com X_train, X_test, y_train e y_test.
        """
        self.df = pd.read_parquet(self.caminho_dados)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df.sort_values(by="timestamp", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        colunas_ignorar = [
            "timestamp",
            "id_celula",
            "liga_metalica",
            target_col,
            "diagnostico_rotulo",
        ]
        self.feature_names = [
            c for c in self.df.columns if c not in colunas_ignorar
        ]

        X = self.df[self.feature_names]
        y = self.df[target_col].astype(int)

        # Divisão cronológica estrita
        ponto_corte = int(len(self.df) * proporcao_treino)
        X_train_raw, X_test_raw = (
            X.iloc[:ponto_corte],
            X.iloc[ponto_corte:],
        )
        self.y_train, self.y_test = (
            y.iloc[:ponto_corte],
            y.iloc[ponto_corte:],
        )

        # Ajuste de escala estritamente no treino
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train_raw),
            columns=self.feature_names,
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(X_test_raw),
            columns=self.feature_names,
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def balancear_treino_smote(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Aplica a técnica SMOTE exclusivamente aos dados de treinamento.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Dados de treino balanceados.
        """
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(
            self.X_train, self.y_train
        )
        return pd.DataFrame(X_resampled, columns=self.feature_names), y_resampled

    def treinar_supervisionados(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, Any]:
        """Treina e otimiza classificadores ensemble com RandomizedSearchCV.

        Args:
            X_train: Matriz de features de treino rebalanceada.
            y_train: Vetor alvo de treino.

        Returns:
            Dict[str, Any]: Melhores estimadores ajustados por algoritmo.
        """
        configuracoes = {
            "RandomForest": {
                "estimator": RandomForestClassifier(
                    random_state=42, class_weight="balanced"
                ),
                "params": {
                    "n_estimators": [100, 150, 200],
                    "max_depth": [10, 15, 20],
                    "min_samples_split": [2, 5],
                },
            },
            "LightGBM": {
                "estimator": LGBMClassifier(
                    random_state=42, class_weight="balanced", verbose=-1
                ),
                "params": {
                    "n_estimators": [100, 150, 200],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "num_leaves": [31, 63],
                },
            },
        }

        melhores_estimadores = {}
        for nome, config in configuracoes.items():
            search = RandomizedSearchCV(
                estimator=config["estimator"],
                param_distributions=config["params"],
                n_iter=4,
                scoring="recall_macro",
                cv=3,
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            melhores_estimadores[nome] = search.best_estimator_

        return melhores_estimadores

    def treinar_isolation_forest(self) -> IsolationForest:
        """Treina o Isolation Forest para detecção não supervisionada

        de anomalias em regime estritamente nominal (tipo_falha == 0).

        Returns:
            IsolationForest: Modelo calibrado para anomalias operacionais.
        """
        indices_normais = self.y_train[self.y_train == 0].index
        X_normais = self.X_train.loc[indices_normais]

        iso = IsolationForest(
            n_estimators=150,
            contamination=0.03,
            random_state=42,
            n_jobs=-1,
        )
        iso.fit(X_normais)
        return iso

    def salvar_artefatos(
        self,
        melhor_classificador: Any,
        modelo_iso: IsolationForest,
        pasta_destino: str = "models",
    ) -> None:
        """Serializa os modelos e transformadores treinados em formato joblib.

        Args:
            melhor_classificador: Modelo supervisionado selecionado.
            modelo_iso: Modelo Isolation Forest treinado.
            pasta_destino: Diretório de destino para exportação.
        """
        os.makedirs(pasta_destino, exist_ok=True)
        joblib.dump(
            melhor_classificador,
            os.path.join(pasta_destino, "classifier_model.joblib"),
        )
        joblib.dump(
            modelo_iso,
            os.path.join(pasta_destino, "isolation_forest.joblib"),
        )
        joblib.dump(
            self.scaler,
            os.path.join(pasta_destino, "scaler.joblib"),
        )
        joblib.dump(
            self.feature_names,
            os.path.join(pasta_destino, "feature_names.joblib"),
        )
        print(f"Artefatos salvos com sucesso no diretório '{pasta_destino}/'.")

