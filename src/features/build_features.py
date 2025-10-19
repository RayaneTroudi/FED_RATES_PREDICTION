import pandas as pd

def build_align_all_features_at_correct_meeting_date():
  
    # --- Chargement des données ---
    df_dff = pd.read_csv('./data/processed/DFF_PROCESSED.csv')
    df_vix = pd.read_csv('./data/raw/VIXCLS.csv')
    df_unrate = pd.read_csv('./data/raw/UNRATE.csv')
    df_spread = pd.read_csv('./data/raw/T10Y2Y.csv')
    df_cpi = pd.read_csv('./data/raw/CPIAUCSL.csv')

    # --- Conversion des dates ---
    for df in [df_dff, df_vix, df_unrate, df_spread, df_cpi]:
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        df.sort_values('observation_date', inplace=True)

    # --- Alignement temporel (strictement antérieur à la réunion) ---
    df_merged = pd.merge_asof(
        df_dff, df_vix,
        on='observation_date',
        direction='backward', allow_exact_matches=False
    )
    df_merged = pd.merge_asof(
        df_merged, df_unrate,
        on='observation_date',
        direction='backward', allow_exact_matches=False
    )
    df_merged = pd.merge_asof(
        df_merged, df_spread,
        on='observation_date',
        direction='backward', allow_exact_matches=False
    )
    df_merged = pd.merge_asof(
        df_merged, df_cpi,
        on='observation_date',
        direction='backward', allow_exact_matches=False
    )

    # --- Nettoyage final ---
    df_merged.rename(columns={
        'VIXCLS': 'VIX',
        'UNRATE': 'UNRATE',
        'T10Y2Y': 'SPREAD10Y2Y',
        'CPIAUCSL': 'CPI'
    }, inplace=True)

    df_merged = df_merged[[
        'observation_date', 'DFF', 'VIX', 'UNRATE', 'SPREAD10Y2Y', 'CPI'
    ]]
    
    return df_merged

df_merged = build_align_all_features_at_correct_meeting_date()
print(df_merged.head(10))
    
    
