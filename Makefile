#################################################################################
# GLOBALS
#################################################################################
PROJECT_NAME = nba_stat_predictor
PYTHON_INTERPRETER = python
PYTHON_VERSION = 3.10


#################################################################################
# COMANDOS BÁSICOS (Instalação, Limpeza, Formato)
#################################################################################

## Cria o ambiente conda
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> Ambiente conda criado. Ative com:\nconda activate $(PROJECT_NAME)"

## Instala dependências
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Limpa arquivos .pyc e cache
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Formata o código com ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Roda testes
.PHONY: test
test:
	python -m pytest tests


#################################################################################
# PIPELINE DE DADOS E MODELO
#################################################################################

## ETAPA 1: Coleta os dados brutos (Raw Data)
.PHONY: fetch_data
fetch_data: requirements
	@echo ">>> ETAPA 1: Coletando dados brutos dos jogadores (make_dataset.py)..."
	$(PYTHON_INTERPRETER) src/data/make_dataset.py
	@echo ">>> ETAPA 1: Coletando dados brutos de defesa (fetch_defense_stats.py)..."
	$(PYTHON_INTERPRETER) src/data/fetch_defense_stats.py
	@echo ">>> Coleta de dados brutos finalizada. (Salvo em /data/raw/)"

## ETAPA 2: Processa dados e cria features (Processed Data)
# Esta regra DEPENDE que 'fetch_data' tenha sido executada.
.PHONY: process_data
process_data: fetch_data
	@echo ">>> ETAPA 2: Processando dados e criando features (build_features.py)..."
	$(PYTHON_INTERPRETER) src/features/build_features.py
	@echo ">>> Processamento finalizado. (Salvo em /data/processed/)"

## ETAPA 3: Treina o modelo (Models)
# Esta regra DEPENDE que 'process_data' tenha sido executada.
.PHONY: train
train: process_data
	@echo ">>> ETAPA 3: Treinando e serializando modelos (train_model.py)..."
	$(PYTHON_INTERPRETER) src/models/train_model.py
	@echo ">>> Modelos salvos. (Salvo em /models/)"

## ETAPA 4: Executa o App (App)
# Esta regra DEPENDE que 'train' tenha sido executada.
.PHONY: app
app: train
	@echo ">>> ETAPA 4: Iniciando o Dashboard Streamlit..."
	streamlit run app.py

## Comando Mestre: Roda o pipeline completo de ponta a ponta
.PHONY: all
all: fetch_data process_data train app


#################################################################################
# AJUDA (Self-Documenting)
#################################################################################
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Disponível no Makefile:\n'); \
print('\n'.join(['  \033[36m%-25s\033[0m %s'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)