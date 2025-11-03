import pandas as pd
import time
import os
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import logging

# Configuração básica do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configurações ---
# Lista de temporadas que queremos buscar
SEASONS_TO_FETCH = ['2022-23', '2023-24', '2024-25'] 

# Caminho para salvar o arquivo final (dentro da estrutura Cookiecutter)
OUTPUT_RAW_PATH = os.path.join('data', 'raw', 'nba_player_gamelogs_raw.csv')

# Tempo de pausa entre as requisições à API (em segundos)
SLEEP_TIME = 0.7 
# ---------------------

def get_active_player_ids():
    """Busca todos os jogadores ativos na NBA e retorna seus IDs."""
    logging.info("Buscando lista de jogadores ativos...")
    try:
        all_players = players.get_players()
        active_players = [player['id'] for player in all_players if player['is_active']]
        logging.info(f"Encontrados {len(active_players)} jogadores ativos.")
        return active_players
    except Exception as e:
        logging.error(f"Erro ao buscar lista de jogadores: {e}")
        return []

def fetch_player_gamelogs(player_id, season):
    """Busca os logs de jogos para um jogador e temporada específicos."""
    logging.info(f"Buscando dados para jogador ID {player_id} na temporada {season}...")
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=30) # Aumenta timeout
        df_gamelog = gamelog.get_data_frames()[0]
        
        # Pequena verificação se retornou dados
        if df_gamelog.empty:
            logging.warning(f" -> Nenhum jogo encontrado para jogador {player_id} na temporada {season}.")
            return None
            
        logging.info(f" -> Dados encontrados: {len(df_gamelog)} jogos.")
        return df_gamelog
        
    except Exception as e:
        logging.error(f" -> Erro ao buscar dados para jogador {player_id} na temporada {season}: {e}")
        return None

def main():
    """Função principal para orquestrar a coleta de dados."""
    player_ids = get_active_player_ids()
    
    if not player_ids:
        logging.error("Nenhum ID de jogador encontrado. Abortando.")
        return

    all_gamelogs_list = [] # Lista para guardar os DataFrames

    # Loop principal de coleta
    for season in SEASONS_TO_FETCH:
        for player_id in player_ids:
            df_player_season = fetch_player_gamelogs(player_id, season)
            
            if df_player_season is not None:
                all_gamelogs_list.append(df_player_season)
                
            # PAUSA ESTRATÉGICA
            time.sleep(SLEEP_TIME)

    if not all_gamelogs_list:
        logging.warning("Nenhum dado de jogo foi coletado. Verifique a API ou os parâmetros.")
        return

    # Junta tudo em um DataFrame
    logging.info("Combinando todos os dados coletados...")
    df_complete_raw = pd.concat(all_gamelogs_list, ignore_index=True)
    logging.info(f"DataFrame final criado com {len(df_complete_raw)} registros.")

    # Garante que a pasta de destino exista
    output_dir = os.path.dirname(OUTPUT_RAW_PATH)
    os.makedirs(output_dir, exist_ok=True)

    # Salva o arquivo CSV
    logging.info(f"Salvando dados brutos em: {OUTPUT_RAW_PATH}")
    df_complete_raw.to_csv(OUTPUT_RAW_PATH, index=False)
    logging.info("Dados brutos salvos com sucesso!")

if __name__ == '__main__':
    main()