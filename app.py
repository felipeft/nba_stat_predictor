# Aplicativo streamlit que carrega os 3 artefatos gerados em models/ 
# e cria uma interface simples que só pede ao usuário as features e mostra
# as previsões

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import logging
from nba_api.stats.static import players

# Configuração inicial e loading dos artefatos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Diretórios
MODEL_DIR = 'models'
DATA_PATH = 'data/processed/nba_player_gamelogs_processed.csv'
RAW_DATA_PATH = 'data/raw/nba_player_gamelogs_raw.csv' # Para buscar nomes de jogadores

# Configuração da página do stre2amlit
st.set_page_config(page_title="NBA Player Stat Predictor", page_icon="🏀", layout="wide")

@st.cache_data
def load_artifacts():
    """Carrega o pré-processador e os modelos treinados."""
    logging.info("Carregando artefatos do modelo...")
    try:
        preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
        reg_model = joblib.load(os.path.join(MODEL_DIR, 'reg_model_ridge.joblib'))
        clf_model = joblib.load(os.path.join(MODEL_DIR, 'clf_model_rf.joblib'))
        logging.info("Artefatos carregados com sucesso.")
        return preprocessor, reg_model, clf_model
    except FileNotFoundError:
        st.error(f"ERRO: Artefatos do modelo não encontrados na pasta '{MODEL_DIR}'.")
        st.error("Por favor, execute 'make train' (ou 'python src/models/train_model.py') primeiro.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar artefatos: {e}")
        st.stop()

@st.cache_data
def load_data():
    """Carrega os dados processados e a lista de jogadores/times."""
    logging.info("Carregando dados processados...")
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['GAME_DATE'])
        # Ordena por data para pegarmos facilmente o "último jogo"
        df = df.sort_values(by=['Player_ID', 'GAME_DATE'])
        
        # Pega a lista de oponentes
        opponent_teams = sorted(df['OPPONENT'].unique())
        
        # Pega a lista de IDs de jogadores QUE ESTÃO NO NOSSO DATASET
        player_ids_in_data = sorted(df['Player_ID'].unique())

        # Busca os nomes usando nba_api.stats.static.players
        player_map = {}
        try:
            logging.info("Buscando mapa de nomes de jogadores da NBA API (static)...")
            all_players_list = players.get_players() # Retorna uma lista de dicionários
            
            player_map = {player['id']: player['full_name'] for player in all_players_list}
            
            # Cria a lista de IDs que deve mostrar, 
            player_ids_to_show = [pid for pid in player_ids_in_data if pid in player_map]
            player_list_sorted_by_name = sorted(player_ids_to_show, key=lambda x: player_map[x])

            if not player_list_sorted_by_name:
                 raise ValueError("Nenhum jogador do dataset foi encontrado no mapa estático da API.")
            
            logging.info("Mapa de nomes de jogadores criado com sucesso.")

        except Exception as e:
            # FALLBACK FINAL: Se a chamada da API falhar
            logging.warning(f"Erro ao buscar nomes estáticos da API ({e}). Usando Player_ID como nome.")
            player_map = {pid: str(pid) for pid in player_ids_in_data}
            player_list_sorted_by_name = player_ids_in_data
            
        return df, player_list_sorted_by_name, player_map, opponent_teams
    
    except FileNotFoundError:
        st.error(f"ERRO: Arquivo de dados processados não encontrado em '{DATA_PATH}'.")
        st.error("Por favor, execute o notebook 02 (ou 'make data') primeiro.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()

# Carregamento principal
preprocessor, reg_model, clf_model = load_artifacts()
df_processed, player_list, player_map, opponent_list = load_data()


# UI na Sidebar
st.sidebar.title("Previsão de Jogo da NBA!!")
st.sidebar.markdown("Selecione os parâmetros para o próximo jogo:")

# Formata a lista de jogadores para "Nome (ID)"
player_display_list = [f"{player_map[pid]} ({pid})" for pid in player_list]

selected_player_display = st.sidebar.selectbox(
    "Selecione o Jogador:",
    options=player_display_list,
    help="O app buscará as estatísticas (médias, etc.) do último jogo deste jogador."
)
# Extrai o ID do nome selecionado
selected_player_id = int(selected_player_display.split('(')[-1].replace(')', ''))


selected_opponent = st.sidebar.selectbox(
    "Selecione o Time Oponente:",
    options=opponent_list
)

selected_location = st.sidebar.radio(
    "Onde será o jogo?",
    ('Em Casa', 'Fora de Casa'),
    horizontal=True
)
home_feature = 1 if selected_location == 'Em Casa' else 0


# --- 5. Lógica de Predição (quando o botão é clicado) ---
if st.sidebar.button("Prever Estatísticas"):
    
    logging.info(f"Iniciando predição para PlayerID: {selected_player_id} vs {selected_opponent}")
    
    try:
        # 5.1: Encontra o último jogo do jogador
        last_game = df_processed[df_processed['Player_ID'] == selected_player_id].iloc[-1]
        
        # 5.2: Pega as colunas de features (mesma ordem do train_model.py)
        feature_cols = [
            'MIN', 'HOME', 'DAYS_REST', 'IS_B2B', 'WIN_LAST_GAME', 'OPPONENT'
        ] + [col for col in df_processed.columns if '_MA_' in col]
        
        opp_cols = [col for col in df_processed.columns if col.startswith('OPP_')]
        feature_cols.extend(opp_cols)
        feature_cols = list(dict.fromkeys(feature_cols))

        # 5.3: Cria o DataFrame de 1 linha para a predição
        input_data = {}
        for col in feature_cols:
            if col == 'OPPONENT':
                input_data[col] = selected_opponent
            elif col == 'HOME':
                input_data[col] = home_feature
            else:
                # Pega a estatística do último jogo do jogador
                input_data[col] = last_game[col]
        
        input_df = pd.DataFrame([input_data])
        
        # 5.4: Pré-processa os dados (OneHotEncode do 'OPPONENT')
        input_processed = preprocessor.transform(input_df[feature_cols])
        
        # 5.5: Faz as predições
        reg_preds = reg_model.predict(input_processed)
        clf_probs = clf_model.predict_proba(input_processed)
        
        # --- 6. Exibição dos Resultados ---
        st.title(f"Previsões para {player_map[selected_player_id]} vs. {selected_opponent}")

        st.subheader("Estatísticas Previstas")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pontos (PTS)", f"{reg_preds[0][0]:.1f}")
        col2.metric("Assistências (AST)", f"{reg_preds[0][1]:.1f}")
        col3.metric("Rebotes (REB)", f"{reg_preds[0][2]:.1f}")
        col4.metric("Bolas de 3 (FG3M)", f"{reg_preds[0][3]:.1f}")
        
        st.subheader("Probabilidade de Double-Double")
        prob_dd = clf_probs[0][1] # Probabilidade da classe '1' (DD)
        st.progress(prob_dd, text=f"{prob_dd*100:.1f}% de Chance")

        # Expansor para transparência (mostra as features usadas)
        with st.expander("Ver features usadas para a predição (baseadas no último jogo)"):
            lookup_features = input_df.drop(columns=['OPPONENT', 'HOME']).T
            lookup_features.columns = ['Valor da Feature']
            st.dataframe(lookup_features)

    except IndexError:
        st.error(f"Erro: Player ID {selected_player_id} ({player_map[selected_player_id]}) não foi encontrado nos dados processados.")
    except Exception as e:
        st.error(f"Ocorreu um erro durante a predição: {e}")
        logging.error(f"Erro na predição: {e}")
        st.exception(e)

else:
    st.info("Preencha os dados na barra lateral e clique em 'Prever Estatísticas' para começar.")