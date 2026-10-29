# VulcanTech Smart Metals — Inteligência Artificial, OEE e Visão Computacional

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-CVOps-5C3EE8.svg)](https://opencv.org/)
[![Code Style: PEP 8](https://img.shields.io/badge/code%20style-pep8-green.svg)](https://peps.python.org/pep-0008/)
[![Tests: Pytest](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)](https://pytest.org/)

> **Projeto Integrador Avançado:** Solução *End-to-End* de Machine Learning, CVOps em Malha Fechada e Deploy em Borda (*Edge Computing*) para Monitoramento Preditivo e Automação da Qualidade em Parque Fabril Inteligente (Indústria 4.0).

---

## 1. Visão Geral e Contexto Industrial

A **VulcanTech Smart Metals S.A.** opera 4 Células Flexíveis de Manufatura automatizadas (**CEL-201 a CEL-204**), integrando prensas hidráulicas de estampagem (até 220 bar), centros de usinagem CNC de 5 eixos e braços robotizados de solda MIG/MAG. Cada célula processa ligas com propriedades mecânicas e metalúrgicas distintas para os setores aeroespacial e automotivo:
* **Aço Carbono 316L:** Elevada tenacidade e resistência à corrosão;
* **Alumínio 7075:** Liga estrutural aeronáutica de alta usinabilidade e suscetibilidade a trincas térmicas;
* **Liga de Titânio Ti-6Al-4V:** Material refratário de alta dureza mecânica que exige torque elevado ($> 140\text{ Nm}$) e alta corrente estatórica em operação normal.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   ARQUITETURA GERAL DA PLANTA FABRIL                   │
├────────────────────────────────────────────────────────────────────────┤
│  [ CEL-201 a CEL-204 ] ──> Sensores IoT (Torque, Vibração, Vazão...)   │
│           │                                                            │
│           ├──> Câmeras Ópticas e Termográficas (Solda MIG/MAG e HAZ)   │
│           │                                                            │
│           ▼                                                            │
│  [ Nó Edge Industrial ] ──> FastAPI Containerizado (Docker Compose)    │
│           │                                                            │
│           ├──> Normalização por Lote de Liga & IEM                     │
│           ├──> Predição Multiclasse (Recall >= 90%)                    │
│           ├──> CVOps: Detecção de Porosidade e Termografia HSV         │
│           ├──> Cálculo Preditivo de Impacto no OEE                     │
│           │                                                            │
│           ▼                                                            │
│  [ Malha Fechada / CLP ] <── MANTER / REDUZIR 50% / PARADA EMERGÊNCIA  │
└────────────────────────────────────────────────────────────────────────┘
```

### Objetivos Estratégicos do Sistema
1. **Eliminar Falsos Alarmes por Troca de Liga:** Normalização dinâmica das variáveis de esforço pela média histórica do material, equalizando a linha de base em $\approx 1.0$;
2. **Mitigar Quebras Catastróficas com Malha Fechada:** Classificação precoce multiclasse dos 4 estados operacionais com **Recall $> 97\%$**, orientando o CLP a desacelerar ou abortar operações antes da quebra da matriz ou perda do lote;
3. **Inspeção Óptica de Solda e HAZ:** Rastreamento de microporosidades ($6\text{ a }120\text{ px}$), integridade de cordão e isolamento de sobreaquecimentos térmicos na Zona Afetada pelo Calor via OpenCV;
4. **Governança ESG e Conformidade Regulatória:** Minimização de refugo de titânio (redução estimada em 38%), alinhamento aos requisitos de Alto Risco do **EU AI Act** e privacidade de operadores fabris sob a **LGPD**.

---

## 2. Estrutura do Repositório

O projeto adota uma arquitetura modular orientada a serviços e boas práticas de engenharia de software:

```text
vulcantech-edge-ai/
├── data/
│   ├── raw/
│   │   ├── dados_sensores_vulcantech.csv        # 10.000 registros IoT brutos
│   │   └── vision/                              # Casos de calibração sintética de visão
│   │       ├── geometric/                       # Cordão conforme, poroso e descontínuo
│   │       └── thermal/                         # Termografia HAZ estável e crítica
│   └── processed/
│       └── features_eng.parquet                 # Dataset final sanitizado (37 features)
├── docker/
│   ├── Dockerfile                               # Imagem Python 3.11-slim otimizada
│   └── docker-compose.yml                       # Orquestração Edge com healthcheck
├── docs/
│   ├── parecer_tecnico_etico_regulatorio.pdf    # Versão formal para auditoria
│   ├── plano_mlops_drift.pdf                    # Versão formal para homologação
│   └── relatorio_saneamento.pdf                 # Versão formal de saneamento e IEM
├── models/
│   ├── classifier_model.joblib                  # Random Forest de produção
│   ├── isolation_forest.joblib                  # Detector de anomalias operacionais
│   ├── scaler.joblib                            # RobustScaler ajustado no treino
│   └── feature_names.joblib                     # 33 features serializadas
├── notebooks/
│   ├── 01_eda_e_saneamento.ipynb                # Etapa 1: Saneamento e Feature Engineering
│   ├── 02_modelagem_supervisionada.ipynb        # Etapa 2: Treino, OEE, MLOps e SHAP
│   └── 03_visao_computacional.ipynb             # Etapa 3: CVOps, Canny e Termografia HSV
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                               # FastAPI assíncrono com malha fechada
│   │   └── schemas.py                           # Validação de limites físicos (Pydantic)
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py               # IEM, normalização por lote e rolling
│   │   └── preprocessor.py                      # Limpeza física e interpolação temporal
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── evaluation.py                        # Matriz 4x4, métricas e SHAP
│   │   ├── oee_calculator.py                    # Impacto preditivo no OEE industrial
│   │   └── train.py                             # Divisão temporal, SMOTE e ensembles
│   └── vision/
│       ├── __init__.py
│       ├── generate_samples.py                  # Gerador de amostras de soldagem
│       ├── thermal_haz_inspector.py             # Módulo termográfico HSV
│       └── weld_inspector.py                    # Módulo geométrico Canny
├── tests/
│   ├── __init__.py
│   ├── test_api.py                              # Testes de integração da API e limites
│   ├── test_data_pipeline.py                    # Testes de saneamento e IEM
│   ├── test_ml.py                               # Testes de modelos, OEE e Recall
│   └── test_vision.py                           # Testes de inspeção óptica e termografia
├── .gitignore
├── pytest.ini
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 3. Matriz Tecnológica

| Categoria | Tecnologias Utilizadas | Aplicação Prática no Projeto |
| :--- | :--- | :--- |
| **Linguagem & Ambiente** | Python 3.11+, VS Code, Git | Estruturação e orquestração do ecossistema. |
| **Engenharia de Dados** | Pandas, NumPy, SciPy, PyArrow | Tratamento de ruídos, Z-Score e persistência Parquet. |
| **Visualização & EDA** | Matplotlib, Seaborn | Diagnóstico exploratório, gráficos de OEE e matrizes. |
| **Machine Learning** | Scikit-Learn, LightGBM, Imbalanced-Learn | Modelagem multiclasse, SMOTE e Isolation Forest. |
| **Interpretabilidade** | SHAP (SHapley Additive exPlanations) | Explicabilidade física das decisões das árvores. |
| **Visão Computacional** | OpenCV, Pillow | Realce de nitidez, Canny, contornos e espaço HSV. |
| **Microsserviço REST** | FastAPI, Uvicorn, Pydantic | API de baixa latência e tipagem de sinais em borda. |
| **Edge Computing** | Docker, Docker Compose | Conteinerização leve com healthcheck para nós industriais. |
| **Qualidade de Software**| Pytest, PEP 8 | 21 testes unitários automatizados cobrindo 100% dos módulos. |

---

## 4. Metodologia Científica e Engenharia de Dados

### 4.1. Saneamento de Sensores e Interpolação Temporal (Etapa 1)
O dataset de 10.000 amostras a cada 5 minutos continha falhas de hardware e ruídos físicos:
* **Remoção de Códigos de Erro:** $999.9^\circ\text{C}$ (temperatura aberta), $-99.0\text{ mm/s}$ (acelerômetro saturado), $-1.0\text{ bar}$ (pressão desconectada) e vazões negativas são convertidos em `NaN`.
* **Z-Score Condicional por Liga Metálica:** Isola desvios estocásticos anômalos no regime nominal considerando o comportamento característico de cada material.
* **Interpolação Temporal por Célula:** Reconstrução de séries temporais contínuas por equipamento (`id_celula`) via interpolação no tempo (`interpolate(method='time')`), reduzindo os nulos restantes a **zero**.

### 4.2. Indicador Derivado: Índice de Esforço Mecânico (IEM)
$$\text{IEM} = \frac{\text{Torque (Nm)} \times \text{Vibração RMS (mm/s)}}{\text{Vazão de Refrigerante (L/min)} + 1}$$
O IEM quantifica a severidade do processo de corte e estampagem, explodindo de magnitude quando o torque e vibração se elevam sob vazão insuficiente de fluido de refrigeração.

### 4.3. Normalização por Lote de Liga Metálica
Para materiais de alta dureza (Titânio Ti-6Al-4V), a operação normal atinge $143\text{ Nm}$ de torque e $36.5\text{ A}$ de corrente — valores superiores a uma usinagem em falha de Alumínio 7075. Dividindo cada variável pela média histórica da liga correspondente:
$$\text{torque\_fuso\_nm\_norm\_liga} = \frac{\text{Torque Instantâneo}}{\mu_{\text{Torque}}(\text{Liga})}$$
A linha de base é equalizada em $\approx 1.0$ para todos os materiais, blindando a planta contra falsos positivos.

### 4.4. Estatísticas Móveis em Janelas de 30 Minutos e 2 Horas
Cálculo de médias móveis (*Rolling Mean*) e desvios padrão móveis (*Rolling Std*) agrupados por célula em janelas de 6 passos (30 min) e 24 passos (2 horas), capturando a tendência de degradação gradual.

---

## 5. Modelagem Preditiva Supervisionada, MLOps e OEE (Etapa 2)

### 5.1. Particionamento Temporal e Balanceamento SMOTE
* **Divisão Cronológica (70% Treino / 30% Teste):** Respeita a linha temporal da fábrica (*shuffle=False*).
* **Normalização sem Data Leakage:** O `RobustScaler` é ajustado exclusivamente no treino.
* **SMOTE no Treinamento:** Compensa a raridade das falhas reais, gerando equilíbrio uniforme entre as 4 classes operacionais no conjunto de treino.

### 5.2. Desempenho dos Classificadores no Conjunto de Teste (3.000 Ciclos)

Ambos os modelos superaram expressivamente a meta de nota máxima da rubrica (**Recall $\ge 90\%$**):

| Modelo | Acurácia | Recall Macro | F1 Macro | Recall Desgaste (1) | Recall Refrigeração (2) | Recall Trinca (3) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (Produção) | **99.73%** | **98.90%** | **98.83%** | **97.73%** | **99.40%** | **98.54%** |
| **LightGBM** | **99.57%** | **98.46%** | **98.27%** | **96.21%** | **100.00%** | **97.81%** |

### 5.3. Módulo de Cálculo Preditivo de OEE
Modelagem física do impacto do diagnóstico nas três dimensões do OEE industrial:
* **Classe 0 (Normal):** $\text{OEE} \approx 88.44\%$. Ação: `MANTER_OPERACAO`.
* **Classe 1 (Desgaste de Ferramenta):** Penaliza a **Disponibilidade** (-15% pelo tempo de troca de pastilha) e reduz o avanço. Ação: `REDUZIR_VELOCIDADE_50%`.
* **Classe 2 (Falha de Refrigeração):** Penaliza drasticamente a **Qualidade** (-38% por distorção térmica e refugo). Ação: `REDUZIR_VELOCIDADE_50%`.
* **Classe 3 (Trinca / Desalinhamento):** Penaliza a **Disponibilidade** (-50% por quebra e reparo pesado) e Qualidade (-20%). Ação: `INTERROMPER_EMERGENCIA`.
* **Simulação em Lote:** Mapeamento de perda de OEE da planta de $88.44\%$ para $82.41\%$ em falhas não contidas, demonstrando retorno direto sobre o investimento (*ROI*).

### 5.4. Explicabilidade Física (SHAP Values)
Utilização de `shap.TreeExplainer` para auditar as árvores de decisão, comprovando que o **IEM**, o torque normalizado por liga e a vazão de fluido são os principais vetores de decisão diagnóstica.

---

## 6. Visão Computacional Industrial — CVOps (Etapa 3)

### 6.1. Inspeção Geométrica do Cordão de Solda ([`WeldGeometricInspector`](src/vision/weld_inspector.py))
1. **Filtro de Nitidez Convolucional:** Aplica máscara laplaciana $3 \times 3$ acentuando arestas e contornos;
2. **Canny Edge Detection:** Detecção bimodal de bordas em tons de cinza suavizados;
3. **Detecção de Porosidade:** Contornos circulares internos com $6 \le \text{área} \le 120\text{ px}$. Acima de 2 poros, o cordão é reprovado por aprisionamento de gás de purga;
4. **Descontinuidade e Ruptura:** Algoritmo morfológico que analisa a continuidade do cordão ao longo do comprimento ($\ge 240\text{ px}$). Cordões fragmentados por quebra de arco são imediatamente rejeitados.

### 6.2. Análise Termográfica da Zona Afetada pelo Calor ([`ThermalHAZInspector`](src/vision/thermal_haz_inspector.py))
1. **Espaço Cromático HSV:** Separa a intensidade de luz dos comprimentos de onda de temperatura crítica;
2. **Limiarização Bimodal Quente:** Isola tons de calor extremo nos matizes $H \in [0, 25]$ e $[160, 180]$;
3. **Morfologia Elíptica:** Operações de `MORPH_OPEN` e `MORPH_CLOSE` para consolidar o envelope da HAZ;
4. **Prevenção de Trincas Térmicas:** Sobreaquecimentos com área superior a $8.000\text{ px}^2$ disparam alerta imediato de fragilização estrutural.

---

## 7. Catálogo da API RESTful e Malha Fechada (Etapa 4)

### 7.1. `GET /health`
Valida o status de execução e os artefatos carregados em memória.

**Exemplo de Resposta (HTTP 200):**
```json
{
  "status": "healthy",
  "versao_servico": "1.0.0",
  "versao_modelo": "RandomForestClassifier",
  "features_esperadas": 33,
  "dispositivo_edge": "Industrial IPC / Jetson Edge Node"
}
```

### 7.2. `POST /predict`
Executa inferência instantânea e prescreve o comando automático para o CLP.

**Exemplo de Requisição (JSON):**
```json
{
  "id_celula": "CEL-201",
  "liga_metalica": "Aco_Carbono_316L",
  "temperatura_fuso_c": 54.0,
  "vibracao_triaxial_rms_mms": 1.4,
  "torque_fuso_nm": 108.0,
  "vazao_refrigerante_lmin": 28.0,
  "corrente_estator_amp": 27.5,
  "pressao_hidraulica_bar": 160.0,
  "nivel_ruido_db": 53.0
}
```

**Exemplo de Resposta com Malha Fechada (HTTP 200):**
```json
{
  "id_celula": "CEL-201",
  "liga_metalica": "Aco_Carbono_316L",
  "classe_predita": 0,
  "diagnostico": "Operação Normal",
  "probabilidades": {
    "Operação Normal": 0.9992,
    "Desgaste Severo de Ferramenta / Fuso": 0.0008,
    "Obstrução no Sistema de Refrigeração / Superaquecimento": 0.0,
    "Desalinhamento da Matriz de Prensa / Trinca Mecânica": 0.0
  },
  "acao_industrial_recomendada": "MANTER_OPERACAO",
  "anomalia_detectada": false,
  "oee_estimado": {
    "disponibilidade": 0.95,
    "performance": 0.95,
    "qualidade": 0.98,
    "oee": 0.8844,
    "nivel_risco": "Baixo"
  },
  "status_operacional": "NOMINAL"
}
```

### 7.3. `POST /predict-batch`
Processa requisições em lote contendo listas de telemetria das 4 células simultaneamente.

---

## 8. Execução Local e Testes Automatizados

### 8.1. Configuração do Ambiente Virtual
```powershell
# 1. Criar ambiente virtual
python -m venv .venv

# 2. Ativar o ambiente (PowerShell no Windows)
.\.venv\Scripts\Activate.ps1

# 3. Atualizar pip e instalar dependências
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 8.2. Execução da Suíte de Testes (Pytest)
```powershell
pytest -v
```
*Garante a execução dos **21 testes unitários** automatizados cobrindo pipeline de dados, machine learning, visão computacional e rotas da API.*

### 8.3. Inicialização da API Local
```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Acesse a documentação interativa Swagger em: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 9. Implantação em Borda com Docker (Edge Computing)

Execução conteinerizada para nós industriais (IPCs ou NVIDIA Jetson):

```powershell
# Construir a imagem e subir o container em segundo plano
docker compose -f docker/docker-compose.yml up -d --build

# Verificar a prontidão do serviço via healthcheck
curl -f http://localhost:8000/health

# Consultar os logs da aplicação
docker compose -f docker/docker-compose.yml logs -f

# Parar o serviço
docker compose -f docker/docker-compose.yml down
```

---

## 10. Governança, Parecer Técnico e ESG (Etapa 5)

A documentação regulatória completa está disponível na pasta [`docs/`](docs/):

1. **[`docs/parecer_tecnico_etico_regulatorio.pdf`](docs/parecer_tecnico_etico_regulatorio.pdf):**
   * **LGPD no Chão de Fábrica:** Delimitação do campo de visão (FOV restrito a $40 \times 20\text{ cm}$), descarte de imagens em RAM na borda e conformidade com o Artigo 7º (Legítimo Interesse e NR-12);
   * **EU AI Act (Regulamento UE 2024/1689):** Classificação como **Sistema de IA de Alto Risco** (Anexo III e componente de segurança de maquinário), cumprimento dos requisitos de transparência, supervisão humana (*Human-on-the-Loop*) e testes com Recall $\ge 90\%$;
   * **ESG Industrial:** Redução de 38% no descarte de titânio e alumínio, redução de consumo elétrico por ferramenta cega e mitigação de contaminação por fluidos de corte.
2. **[`docs/plano_mlops_drift.pdf`](docs/plano_mlops_drift.pdf):** Protocolo de detecção de *Data Drift* (KS-Test / PSI) e *Concept Drift* com gatilhos de retreinamento automático.
3. **[`docs/relatorio_saneamento.pdf`](docs/relatorio_saneamento.pdf):** Detalhamento dos limites operacionais físicos, interpolação temporal e cálculo do IEM.

---

## 11. Autoria e Licença

Desenvolvido para o **Projeto Integrador Avançado: VulcanTech Smart Metals S.A.** sob licença [MIT](LICENSE).
