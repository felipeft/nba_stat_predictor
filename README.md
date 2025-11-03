
# NBA Stat Predictor

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Um pipeline de MLOps de ponta a ponta para prever estatísticas de jogadores da NBA (PTS, AST, REB, FG3M) e a probabilidade de um Double-Double. O projeto segue a metodologia CRISP-DM, é orquestrado via `Makefile` e culmina em um dashboard interativo em Streamlit.

## Pipeline de Execução (CRISP-DM)

Este projeto é orquestrado usando `make`. Os comandos abaixo executam o pipeline completo, desde a coleta de dados até a implantação do app.

### 1. Instalação
Primeiro, crie o ambiente conda e instale as dependências:
```sh
make create_environment
conda activate nba_stat_predictor
make requirements
````

### 2\. Pipeline Completo de Ponta a Ponta

Para executar todo o processo (coletar dados, processar, treinar e iniciar o app):

```sh
make all
```

### 3\. Execução por Etapa

Você também pode executar cada fase do CRISP-DM separadamente:

**Fase 1 & 2: Coleta e Preparação de Dados (Raw)**
Coleta os dados brutos da `nba_api` e os salva em `data/raw/`.

```sh
make fetch_data
```

**Fase 2: Preparação de Dados (Processamento & Features)**
Executa o script `src/features/build_features.py`. Ele limpa os dados brutos, faz o merge com estatísticas de defesa, calcula médias móveis e salva o dataset final em `data/processed/`.

```sh
make process_data
```

**Fase 3-5: Modelagem e Treinamento**
Executa `src/models/train_model.py`. Este script carrega os dados processados, treina os modelos campeões (`Ridge` e `RandomForestClassifier`) e salva os artefatos (`.joblib`) na pasta `models/`.

```sh
make train
```

**Fase 6: Implantação (Dashboard)**
Inicia o dashboard interativo do Streamlit. Este comando depende do `make train` ter sido executado pelo menos uma vez.

```sh
make app
```

-----

## Modelos Selecionados

Após a fase de avaliação (ver `notebooks/03-modelagem-e-avaliacao.ipynb`), os seguintes modelos foram escolhidos para produção:

  * **Regressão (PTS, AST, REB, FG3M):**

      * **Modelo:** `Ridge` (Regressão Linear com Regularização)
      * **Artefato:** `models/reg_model_ridge.joblib`

  * **Classificação (Previsão de Double-Double):**

      * **Modelo:** `RandomForestClassifier` (com `class_weight='balanced'`)
      * **Artefato:** `models/clf_model_rf.joblib`

-----

## Project Organization

```
.
├── app.py                <- O dashboard interativo Streamlit
├── data
│   ├── processed         <- Dados limpos e com features, prontos para modelagem
│   │   └── nba_player_gamelogs_processed.csv
│   └── raw               <- Dados brutos originais (coletados da API)
│       ├── nba_player_gamelogs_raw.csv
│       └── nba_team_defense_stats_raw.csv
├── LICENSE
├── Makefile              <- Orquestrador do pipeline (make fetch_data, make process_data, etc.)
├── models                <- Modelos treinados e serializados (.joblib)
│   ├── clf_model_rf.joblib
│   ├── preprocessor.joblib
│   └── reg_model_ridge.joblib
├── notebooks             <- Notebooks de exploração e prototipagem (CRISP-DM)
│   ├── 01-exploracao-e-coleta.ipynb
│   ├── 02-limpeza-e-features.ipynb
│   └── 03-modelagem-e-avaliacao.ipynb
├── pyproject.toml
├── README.md
├── requirements.txt      <- Lista de dependências Python (pandas, sklearn, streamlit, etc.)
├── setup.cfg
└── src                   <- Código-fonte do projeto
    ├── __init__.py
    ├── data              <- Scripts para coleta de dados (make_dataset.py)
    │   ├── __init__.py
    │   ├── fetch_defense_stats.py
    │   └── make_dataset.py
    ├── features          <- Scripts para processamento e engenharia de features (build_features.py)
    │   ├── __init__.py
    │   └── build_features.py
    └── models            <- Scripts para treinamento do modelo (train_model.py)
        ├── __init__.py
        └── train_model.py
```

```
```