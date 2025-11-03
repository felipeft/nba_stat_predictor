import pandas as pd
import numpy as np
import os
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definição de caminhos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_GAMELOG_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'nba_player_gamelogs_raw.csv')
RAW_DEFENSE_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'nba_team_defense_stats_raw.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PROCESSED_FILE_PATH = os.path.join(PROCESSED_DIR, 'nba_player_gamelogs_processed.csv')

# Funções auxiliares (do Notebook 02)

def get_season_from_date(date):
    """Extrai a temporada (ex: 2022-23) de um datetime."""
    year = date.year
    month = date.month
    if month >= 10:
        return f"{year}-{str(year+1)[-2:]}"
    else:
        return f"{year-1}-{str(year)[-2:]}"

def extrair_adversario(matchup):
    """Extrai a abreviação do oponente da string MATCHUP."""
    if '@' in matchup:
        return matchup.split('@')[1].strip()
    elif 'vs.' in matchup:
        return matchup.split('vs.')[1].strip()
    return None


def main():
    logging.info("Iniciando o script de engenharia de features (build_features.py)...")

    # Carregar dados brutos
    try:
        logging.info(f"Carregando dados brutos de {RAW_GAMELOG_PATH}")
        df_raw = pd.read_csv(RAW_GAMELOG_PATH)
        logging.info(f"Carregando dados de defesa de {RAW_DEFENSE_PATH}")
        df_defense = pd.read_csv(RAW_DEFENSE_PATH)
    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo de dados brutos não encontrado. {e}")
        logging.error("Execute 'make fetch_data' primeiro.")
        return

    # Limpeza e seleção Inicial (NB02, Células 66d7ac15, 954eb890) ---
    logging.info("Iniciando limpeza inicial e primeira engenharia de features...")
    df_raw['GAME_DATE'] = pd.to_datetime(df_raw['GAME_DATE'])
    
    colunas_relevantes = [
        'Player_ID', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 
        'AST', 'REB', 'FG3M', 'FGM', 'FGA', 'FTM', 'FTA', 'OREB', 'DREB', 
        'TOV', 'PF', 'PLUS_MINUS', 'STL', 'BLK' 
    ]

    colunas_relevantes_existentes = [col for col in colunas_relevantes if col in df_raw.columns]
    df_limpo = df_raw[colunas_relevantes_existentes].copy()

    df_limpo['OPPONENT'] = df_limpo['MATCHUP'].apply(extrair_adversario)
    df_limpo['HOME'] = df_limpo['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    
    df_limpo = df_limpo.sort_values(by=['Player_ID', 'GAME_DATE'], ascending=[True, True])

    # Processamento e merge dos dados de defesa (NB02, Célula 8eb271da)
    logging.info("Processando e fazendo merge dos dados de defesa...")
    
    # Lista mestra de times da NBA
    nba_teams_abbr = [team for team in df_limpo['OPPONENT'].unique() if team is not None]
    
    # Mapa de nomes
    team_name_map = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }
    
    # Mapeia e Filtra o df_defense
    team_col_defense = 'TEAM_NAME' if 'TEAM_NAME' in df_defense.columns else 'Team'
    df_defense['TEAM_ABBR_Mapped'] = df_defense[team_col_defense].map(team_name_map)
    df_defense = df_defense[df_defense['TEAM_ABBR_Mapped'].isin(nba_teams_abbr)].copy()
    
    # Renomeia colunas de defesa
    cols_rename_defense = {
        'TEAM_ABBR_Mapped': 'OPPONENT', 'PTS': 'OPP_PTS_PER_G', 'FG_PCT': 'OPP_FG_PCT',
        'FG3_PCT': 'OPP_FG3_PCT', 'AST': 'OPP_AST_PER_G', 'REB': 'OPP_REB_PER_G',
        'STL': 'OPP_STL_PER_G', 'BLK': 'OPP_BLK_PER_G'
    }
    cols_to_rename_existing = {k: v for k, v in cols_rename_defense.items() if k in df_defense.columns}
    df_defense_renamed = df_defense.rename(columns=cols_to_rename_existing)
    
    
    # Selecionar explicitamente as colunas finais para df_defense_final
    colunas_defensivas_desejadas = list(cols_to_rename_existing.values()) 
    colunas_para_manter = ['Season', 'OPPONENT'] + colunas_defensivas_desejadas
    
    # Filtra colunas que realmente existem em df_defense_renamed
    colunas_para_manter_existentes = [col for col in colunas_para_manter if col in df_defense_renamed.columns]
    
    # Remove duplicatas da LISTA de colunas
    colunas_para_manter_unicas = list(dict.fromkeys(colunas_para_manter_existentes)) 

    df_defense_final = df_defense_renamed[colunas_para_manter_unicas]
    
    logging.info(f"Colunas de defesa selecionadas para merge: {colunas_para_manter_unicas}")

    # Cria a chave 'Season' no df_limpo
    df_limpo['Season'] = df_limpo['GAME_DATE'].apply(get_season_from_date)
    
    # Executa o Merge
    df_merged = pd.merge(df_limpo, df_defense_final, on=['Season', 'OPPONENT'], how='left')

    # Engenharia de Features Finais 
    logging.info("Calculando médias móveis e features de descanso...")
    
    # Médias Móveis (NB02, Célula 2f2d667b)
    stats_cols_ma = ['MIN', 'PTS', 'AST', 'REB', 'FG3M', 'FGM', 'FGA', 'FTM', 'FTA', 
                     'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS', 'STL', 'BLK']
    stats_cols_ma_existentes = [col for col in stats_cols_ma if col in df_merged.columns]
    
    grouped = df_merged.groupby('Player_ID')
    for window in [5, 10]:
        for col in stats_cols_ma_existentes:
            df_merged[f'{col}_MA_{window}'] = grouped[col].shift(1).rolling(window, min_periods=1).mean()
            
    # Trata NaNs das MAs
    cols_medias_moveis = [col for col in df_merged.columns if '_MA_' in col]
    df_merged[cols_medias_moveis] = df_merged[cols_medias_moveis].fillna(0)

    # Features de Descanso (NB02, Células ca83cd29, 8a0fa305, 195371b6)
    df_merged['DAYS_REST'] = df_merged.groupby('Player_ID')['GAME_DATE'].diff().dt.days - 1
    df_merged['DAYS_REST'] = df_merged['DAYS_REST'].fillna(7) # Preenche primeiro jogo com 7
    
    df_merged['IS_B2B'] = (df_merged['DAYS_REST'] == 0).astype(int)
    
    df_merged['WIN'] = (df_merged['WL'] == 'W').astype(int)
    df_merged['WIN_LAST_GAME'] = df_merged.groupby('Player_ID')['WIN'].shift(1)
    df_merged['WIN_LAST_GAME'] = df_merged['WIN_LAST_GAME'].fillna(0) # Preenche primeiro jogo com 0

    # Limpeza final e salvamento
    logging.info("Limpando colunas finais e salvando...")
    
    # Remove colunas que não são features ou alvos (NB02, Célula 4d9b1f2e)
    cols_to_drop = ['MATCHUP', 'WL', 'WIN']
    df_final_features = df_merged.drop(columns=cols_to_drop)

    # Garante que o diretório de saída exista
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Salva o arquivo
    df_final_features.to_csv(PROCESSED_FILE_PATH, index=False)
    
    logging.info(f"--- Script de features concluído! Dados salvos em {PROCESSED_FILE_PATH} ---")

if __name__ == '__main__':
    main()