"""Schemas Pydantic para validação estrutural e tipagem estrita da API VulcanTech."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class SensorReadingInput(BaseModel):
    """Schema de entrada para telemetria instantânea de uma célula flexível."""

    id_celula: str = Field(
        ...,
        examples=["CEL-201"],
        description="Identificador da Célula de Manufatura (CEL-201 a CEL-204)",
    )
    liga_metalica: str = Field(
        ...,
        examples=["Aco_Carbono_316L"],
        description=(
            "Liga processada: Aco_Carbono_316L, Aluminio_7075 ou"
            " Liga_Titanio_Ti6Al4V"
        ),
    )
    temperatura_fuso_c: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Temperatura do fuso CNC ou matriz de estampagem em °C",
    )
    vibracao_triaxial_rms_mms: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        description="Vibração triaxial global do cabeçote em mm/s RMS",
    )
    torque_fuso_nm: float = Field(
        ...,
        ge=0.0,
        le=400.0,
        description="Torque instantâneo aplicado ao fuso em Nm",
    )
    vazao_refrigerante_lmin: float = Field(
        ...,
        ge=0.0,
        le=60.0,
        description="Vazão do fluido de refrigeração e corte em L/min",
    )
    corrente_estator_amp: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Corrente consumida pelo motor de acionamento em A",
    )
    pressao_hidraulica_bar: float = Field(
        ...,
        ge=0.0,
        le=300.0,
        description="Pressão do fixador hidráulico da prensa em Bar",
    )
    nivel_ruido_db: float = Field(
        ...,
        ge=20.0,
        le=150.0,
        description="Emissão acústica e ultrassônica de usinagem em dB",
    )


class BatchSensorReadingInput(BaseModel):
    """Schema de entrada para inferência de telemetria em lote."""

    leituras: List[SensorReadingInput]


class PredictionOutput(BaseModel):
    """Schema de resposta preditiva contendo diagnóstico e malha fechada."""

    id_celula: str
    liga_metalica: str
    classe_predita: int
    diagnostico: str
    probabilidades: Dict[str, float]
    acao_industrial_recomendada: str
    anomalia_detectada: bool
    oee_estimado: Dict[str, Any]
    status_operacional: str


class HealthResponse(BaseModel):
    """Schema para verificação de integridade e prontidão do serviço."""

    status: str
    versao_servico: str
    versao_modelo: str
    features_esperadas: int
    dispositivo_edge: str

