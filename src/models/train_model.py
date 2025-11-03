# Esse coódigo tem por objetivo treinar os 2 modelos rankeados e escolhidos no 
# notebook "02-limpeza-e-features.ipynb"
# A ideia aqui é carregar os dados de data/processed, treinar esses modelos e salvar os
# artefatos finais (arquivo .joblib na pasta models)
# alem disso, aqui da pra treinar com 100% dos dados


import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier

# Configuração do Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# dados
PROCESSED_DATA_PATH = 'data/processed/nba_player_gamelogs_processed.csv'
MODEL_OUTPUT_DIR = 'models'

def main():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # Carregamento e preparação dos dados
    try:
        logging.info(f"Carregando dados processados de {PROCESSED_DATA_PATH}")
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['GAME_DATE'])
        
        if 'STL' not in df.columns or 'BLK' not in df.columns:
            logging.error("Erro: 'STL' ou 'BLK' não encontradas no arquivo processado.")
            logging.error("Certifique-se que 'build_features.py' foi executado corretamente.")
            exit()
            
        logging.info("Dados processados carregados com sucesso.")

    except FileNotFoundError:
        logging.error(f"ERRO: Arquivo de dados processados não encontrado em {PROCESSED_DATA_PATH}.")
        logging.error("Por favor, execute 'make process_data' primeiro.")
        exit()
    except Exception as e:
        logging.error(f"Erro ao carregar dados: {e}")
        exit()

    # Definição das features e targets
    logging.info("Definindo features e alvos...")

    target_cols_regressao = ['PTS', 'AST', 'REB', 'FG3M']
    y_reg = df[target_cols_regressao]

    df['PTS_10'] = (df['PTS'] >= 10).astype(int)
    df['REB_10'] = (df['REB'] >= 10).astype(int)
    df['AST_10'] = (df['AST'] >= 10).astype(int)
    df['STL_10'] = (df['STL'] >= 10).astype(int)
    df['BLK_10'] = (df['BLK'] >= 10).astype(int)
    df['DD_Categories'] = df[['PTS_10', 'REB_10', 'AST_10', 'STL_10', 'BLK_10']].sum(axis=1)
    y_class = (df['DD_Categories'] >= 2).astype(int)

    # FEATURES (X)
    feature_cols = [
        'MIN', 'HOME', 'DAYS_REST', 'IS_B2B', 'WIN_LAST_GAME',
        'OPPONENT',
    ] + [col for col in df.columns if '_MA_' in col]

    opp_cols = [col for col in df.columns if col.startswith('OPP_')]
    feature_cols.extend(opp_cols)
    feature_cols = list(dict.fromkeys(feature_cols)) 

    X = df[feature_cols]

    logging.info(f"Features selecionadas: {len(feature_cols)} colunas")

    # Definição e Fit do Pré-processador
    logging.info("Definindo e treinando o pré-processador (OneHotEncoder)...")
    categorical_features = ['OPPONENT']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X)
    logging.info(f"Dados processados. Novo formato de X: {X_processed.shape}")

    # Treinamento dos modelos escolhidos
    logging.info("Treinando modelo de Regressão (Ridge)...")
    reg_model = Ridge(alpha=1.0)
    reg_model.fit(X_processed, y_reg)
    logging.info("Modelo Ridge treinado.")

    logging.info("Treinando modelo de Classificação (RandomForestClassifier)...")
    clf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        max_depth=15,
        min_samples_leaf=5
    )
    clf_model.fit(X_processed, y_class)
    logging.info("Modelo RandomForestClassifier treinado.")

    # Serialização dos Artefatos
    logging.info(f"Salvando artefatos em {MODEL_OUTPUT_DIR}...")

    joblib.dump(preprocessor, os.path.join(MODEL_OUTPUT_DIR, 'preprocessor.joblib'))
    joblib.dump(reg_model, os.path.join(MODEL_OUTPUT_DIR, 'reg_model_ridge.joblib'))
    joblib.dump(clf_model, os.path.join(MODEL_OUTPUT_DIR, 'clf_model_rf.joblib'))

    logging.info("--- Script de treinamento concluído com sucesso! ---")


if __name__ == '__main__':
    main()