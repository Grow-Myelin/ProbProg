import pandas as pd
import numpy as np
from jax import numpy as jnp
from typing import Tuple, Dict, List
import time


def normalize_counts(pitch_counts: np.ndarray) -> np.ndarray:
    """
    Normalize the pitch counts to get a distribution proportion.

    Parameters:
    - pitch_counts: Array of pitch counts.

    Returns:
    - np.ndarray: Normalized distribution of pitch counts.
    """
    total_pitches = pitch_counts.sum()
    return pitch_counts / total_pitches if total_pitches > 0 else pitch_counts


# def efficient_pitch_distribution(df: pd.DataFrame, pitch_types: List[str], filter_conditions: Dict[str, str]) -> np.ndarray:
#     """
#     Calculate and normalize the distribution of pitch types efficiently using multiple filter conditions.

#     Parameters:
#     - df: DataFrame containing pitch data.
#     - pitch_types: List of all possible pitch types.
#     - filter_conditions: Conditions to filter the DataFrame for specific calculations.

#     Returns:
#     - np.ndarray: Normalized distribution of pitch counts as proportions.
#     """
#     # Apply filter conditions dynamically
#     filtered_df = df
#     for condition, value in filter_conditions.items():
#         filtered_df = filtered_df[filtered_df[condition] == value]
#     print(filtered_df)
#     pitches = filtered_df['pitch_type']

#     # Map pitch types to their indices
#     pitch_to_index = {pitch: i for i, pitch in enumerate(pitch_types)}
    
#     # Convert pitches to their corresponding indices
#     pitch_indices = [pitch_to_index.get(pitch, -1) for pitch in pitches]
#     pitch_indices = [i for i in pitch_indices if i >= 0]  # Filter out -1 values
    
#     # Count occurrences using np.bincount and ensure the array size matches pitch_types
#     pitch_counts = np.bincount(pitch_indices, minlength=len(pitch_types))
    
#     # Normalize the counts
#     return normalize_counts(pitch_counts)

def efficient_pitch_distribution(df: pd.DataFrame, pitch_types: List[str], filter_conditions: Dict[str, str]) -> np.ndarray:
    """
    Calculate and normalize the distribution of pitch types, excluding the current event.
    """
    # Convert game_date to datetime for comparison
    df['game_date'] = pd.to_datetime(df['game_date'])
    filter_date = pd.to_datetime(filter_conditions.get("game_date"))
    # Apply chronological filtering
    if len(list(filter_conditions.keys())) == 1:
        df = df[df['game_date'] == filter_date]
    elif len(list(filter_conditions.keys())) == 2:
        df = df[df['game_date'] <= filter_date]
    if "at_bat_number" in filter_conditions and "game_date" in filter_conditions and "at_bat_pitch_num" in filter_conditions:
        df = df[df['game_date'] == filter_date]
        df = df[df['at_bat_number'] == filter_conditions["at_bat_number"]]
        df = df[df['pitch_number'] <= filter_conditions['at_bat_pitch_num']]
#        df = df[(df['game_date'] == filter_date) & (df['at_bat_number'] == filter_conditions["at_bat_number"]) | (df['game_date'] <= filter_date)]
    if "pitch_number" in filter_conditions and "at_bat_number" in filter_conditions:
        df = df[df['game_date'] == filter_date]
        df = df[df['at_bat_number'] == filter_conditions["at_bat_number"]]
        df = df[df['pitch_number'] == filter_conditions["pitch_number"]]

    pitches = df['pitch_type']

    # Map, convert, count, and normalize as before
    pitch_to_index = {pitch: i for i, pitch in enumerate(pitch_types)}
    pitch_indices = [pitch_to_index.get(pitch, -1) for pitch in pitches]
    pitch_indices = [i for i in pitch_indices if i >= 0]
    pitch_counts = np.bincount(pitch_indices, minlength=len(pitch_types))
    
    return normalize_counts(pitch_counts)



df = pd.read_csv('savant_data.csv')
#use_columns = ['pitch_type','pitch_name','pitch_number','at_bat_number','release_speed','game_date','pitcher','batter','home_team','away_team','zone','balls','strikes','on_3b','on_2b','on_1b','home_score','away_score','outs_when_up','p_throws','spin_axis']
use_columns = ['game_date','pitch_type','pitcher','at_bat_number','pitch_number']
df = df[use_columns].sort_values(['game_date','at_bat_number','pitch_number'])
df = df[df['pitcher']==675911]
pitch_types = df['pitch_type'].unique()
#print(df['pitch_type'].unique())
#df.to_csv('parsed_df.csv')
# Example usage:
# Calculate the distribution for the last pitch
# start = time.time()
# last_at_bat_number = df['at_bat_number'].iloc[-1]
# last_game_date = df['game_date'].iloc[-1]
# last_pitch_number = df['pitch_number'].iloc[-1]
def return_pitch_distributions(df_row):
    last_at_bat_number = df_row['at_bat_number']
    last_game_date = df_row['game_date']
    last_pitch_number = df_row['pitch_number']
    # Example usage:
    # Last pitch distribution (no specific filter condition needed, so we pass an empty dict)
    last_pitch_filter = {
        "game_date": last_game_date,
        "at_bat_number": last_at_bat_number,
        "pitch_number": last_pitch_number-1
    }
    last_pitch_distribution_efficient = efficient_pitch_distribution(df, pitch_types, last_pitch_filter)

    #last_pitch_distribution_efficient = efficient_pitch_distribution(df, pitch_types, {"at_bat_number": last_at_bat_number,"game_date": last_game_date,"pitch_number":last_pitch_number})

    # At-bat distribution for the last at-bat
    at_bat_filter = {
        "game_date": last_game_date,
        "at_bat_number": last_at_bat_number,
        "at_bat_pitch_num": last_pitch_number-1
    }
    at_bat_distribution_efficient = efficient_pitch_distribution(df, pitch_types, at_bat_filter)

    #at_bat_distribution_efficient = efficient_pitch_distribution(df, pitch_types, {"at_bat_number": last_at_bat_number,"game_date": last_game_date})

    # Game distribution for the last game
    game_filter = {
        "game_date": last_game_date  # The function will exclude pitches from the current game date
    }
    game_distribution_efficient = efficient_pitch_distribution(df, pitch_types, game_filter)

    #game_distribution_efficient = efficient_pitch_distribution(df, pitch_types, {"game_date": last_game_date})
    current_pitcher = df['pitcher'].iloc[-1]
    season_filter = {
        "game_date": last_game_date,  # Exclude the current game
        "pitcher": current_pitcher
    }
    season_distribution_efficient = efficient_pitch_distribution(df, pitch_types, season_filter)
    print(pitch_types)
    print(last_pitch_distribution_efficient,at_bat_distribution_efficient,game_distribution_efficient,season_distribution_efficient)

for index,row in df.iterrows():
    print(row)
    return_pitch_distributions(row)