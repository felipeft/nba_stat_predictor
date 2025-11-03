import pandas as pd
import time
import os
import logging
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.library.parameters import MeasureTypeDetailedDefense, PerModeDetailed, SeasonTypeAllStar

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configurações ---
# Temporadas para buscar (formato string 'AAAA-AA')
SEASONS_TO_FETCH = ['2022-23', '2023-24', '2024-25'] 

# Caminho para salvar o arquivo final
OUTPUT_RAW_PATH = os.path.join('data', 'raw', 'nba_team_defense_stats_raw.csv')

# Tempo de pausa entre as requisições à API (em segundos)
SLEEP_TIME = 0.7 
# ---------------------

def fetch_season_defense_stats(season):
    """Busca as estatísticas defensivas (opponent stats) para uma temporada."""
    logging.info(f"Buscando dados defensivos para a temporada {season}...")
    try:
        # Usa LeagueDashTeamStats com os parâmetros corretos
        # MeasureTypeDetailedDefense='Opponent' para pegar stats dos adversários
        # PerModeDetailed='PerGame' para pegar médias por jogo
        # SeasonTypeAllStar='Regular Season' para focar na temporada regular
        defense_stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense=MeasureTypeDetailedDefense.opponent, 
            per_mode_detailed=PerModeDetailed.per_game,
            season=season,
            season_type_all_star='Regular Season',
            timeout=30
        )
        
        df_defense = defense_stats.get_data_frames()[0]
        
        if df_defense.empty:
            logging.warning(f" -> Nenhum dado defensivo encontrado para a temporada {season}.")
            return None
        
        # Adiciona a coluna da temporada para referência
        df_defense['Season'] = season
        logging.info(f" -> Dados defensivos da temporada {season} encontrados.")
        return df_defense
        
    except Exception as e:
        logging.error(f" -> Erro ao buscar dados defensivos para a temporada {season}: {e}")
        return None

def main():
    """Função principal para orquestrar a coleta de dados defensivos."""
    all_defense_stats_list = []

    for season in SEASONS_TO_FETCH:
        df_season_defense = fetch_season_defense_stats(season)
        
        if df_season_defense is not None:
            all_defense_stats_list.append(df_season_defense)
            
        # PAUSA ESTRATÉGICA
        logging.info(f"Aguardando {SLEEP_TIME} segundos...")
        time.sleep(SLEEP_TIME)

    if not all_defense_stats_list:
        logging.warning("Nenhum dado defensivo foi coletado via API.")
        return

    # Junta tudo em um DataFrame
    logging.info("Combinando dados defensivos de todas as temporadas...")
    df_complete_defense = pd.concat(all_defense_stats_list, ignore_index=True)
    logging.info(f"DataFrame defensivo final criado com {len(df_complete_defense)} registros (Times x Temporadas).")
    logging.info("Colunas disponíveis:")
    logging.info(df_complete_defense.columns.tolist())

    # Garante que a pasta de destino exista
    output_dir = os.path.dirname(OUTPUT_RAW_PATH)
    os.makedirs(output_dir, exist_ok=True)

    # Salva o arquivo CSV
    logging.info(f"Salvando dados defensivos brutos em: {OUTPUT_RAW_PATH}")
    df_complete_defense.to_csv(OUTPUT_RAW_PATH, index=False)
    logging.info("Dados defensivos salvos com sucesso!")

if __name__ == '__main__':
    main()