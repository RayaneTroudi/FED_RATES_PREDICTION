import pandas as pd
from src.features.features_config_meeting import get_feature_config
from src.features.transformations import apply_transformations


# ======================================================================
#  BUILD DATASET — VERSION DÉFINITIVE (SUPPRESSION DES NIVEAUX SAUF DFF)
# ======================================================================

def build_dataset_meeting(config):
    """
    Construit un dataset aligné sur les dates de meeting FOMC.
    - Prend la dernière valeur connue de chaque indicateur à la date du meeting.
    - Corrige le décalage temporel pour les séries mensuelles (CPI, UNRATE).
    - Supprime toutes les variables brutes sauf DFF, utilisée pour le labelling.
    """

    # === Étape 1 : chargement des dates de meeting (avec DFF)
    df_meetings = pd.read_csv('./data/processed/DFF_PROCESSED.csv')
    df_meetings['observation_date'] = pd.to_datetime(df_meetings['observation_date'])

    all_features = []

    # === Étape 2 : alignement meeting par meeting
    for feature_name in config.keys():
        print(f"[INFO] Collecting raw values for {feature_name} ...")

        df_feature = pd.read_csv(f'./data/raw/{feature_name}.csv')
        df_feature['observation_date'] = pd.to_datetime(df_feature['observation_date'])

        # Correction du timing pour les séries mensuelles
        if feature_name in ["CPIAUCSL", "UNRATE"]:
            df_feature['observation_date'] = df_feature['observation_date'] + pd.offsets.MonthEnd(1)

        feature_rows = []
        for date in df_meetings['observation_date']:
            subset = df_feature[df_feature['observation_date'] < date]
            if subset.empty:
                continue
            last_row = subset.iloc[-1]

            row_data = {'observation_date': date}
            for col in df_feature.columns:
                if col != 'observation_date':
                    row_data[col] = last_row[col]
            feature_rows.append(row_data)

        df_var_features = pd.DataFrame(feature_rows)
        all_features.append(df_var_features)

    # === Étape 3 : fusion de toutes les features alignées
    dataset = all_features[0]
    for df_next in all_features[1:]:
        dataset = pd.merge(dataset, df_next, on='observation_date', how='inner')

    dataset.ffill(inplace=True)

    # === Étape 4 : transformations meeting-based
    for feature_name, conf in config.items():
        print(f"[INFO] Applying transformations (meeting-based) for {feature_name} ...")
        dataset = apply_transformations(dataset, feature_name, conf)

    # === Étape 5 : suppression des variables brutes sauf DFF
    cols_to_drop = [
        c for c in dataset.columns
        if (
            not c.startswith("observation_date")
            and not c.startswith("DFF")
            and "_" not in c  # garde seulement les dérivées (ex: _DIFF, _MA)
        )
    ]
    dataset.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # === Étape 6 : nettoyage final
    if 'DFF_DIFF1' in dataset.columns:
        dataset.drop('DFF_DIFF1', axis=1, inplace=True)

    # === Étape 7 : sauvegarde
    dataset.to_csv('./data/processed/MEETING_BASED.csv', index=False)
    print("[INFO] Dataset built successfully — all raw variables dropped except DFF.")
    return dataset


# ======================================================================
#  EXECUTION
# ======================================================================

if __name__ == "__main__":
    dataset = build_dataset_meeting(config=get_feature_config())
