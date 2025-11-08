import pandas as pd
from src.features.features_config_meeting import get_feature_config
from src.features.transformations import apply_transformations


# --------------------------- FOR MARKOV ------------------------------


# thi snew version of building dataset , build the dataset by add only the last known valu eof feature at the the date 
# of the fomc meeting. It's mean that we get th evalue of the vix the day before the meeting and we do that for each fomc
# date then we build new features (diff, ma and so on) from meeting to meeting whereas in the last version we buil ddirectly
# the features daily whereas we want to predict a decision from meeting to meeting so we must align the data meeting by meeting too

def build_dataset_meeting(config):

    # Lecture des dates de meeting FOMC
    df_meetings = pd.read_csv('./data/processed/DFF_PROCESSED.csv')
    df_meetings['observation_date'] = pd.to_datetime(df_meetings['observation_date'])

    all_features = []

    # Étape 1 : alignement brut sur les dates de meeting
    for feature_name in config.keys():
        print(f"[INFO] Collecting raw values for {feature_name} ...")

        df_feature = pd.read_csv(f'./data/raw/{feature_name}.csv')
        df_feature['observation_date'] = pd.to_datetime(df_feature['observation_date'])

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

    # Étape 2 : fusion de toutes les features alignées
    dataset = all_features[0]
    for df_next in all_features[1:]:
        dataset = pd.merge(dataset, df_next, on='observation_date', how='inner')

    dataset.ffill(inplace=True)
    if 'DFF_DIFF1' in dataset.columns:
        dataset.drop('DFF_DIFF1', axis=1, inplace=True)

    # Étape 3 : application des transformations sur le dataset “meeting-based”
    for feature_name, conf in config.items():
        print(f"[INFO] Applying transformations (meeting-based) for {feature_name} ...")
        dataset = apply_transformations(dataset, feature_name, conf)

    dataset.to_csv('./data/processed/MEETING_BASED.csv', index=False)
    print("[INFO] Dataset built successfully with meeting-to-meeting transformations.")
    return dataset


      

    
    
    
