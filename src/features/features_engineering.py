import pandas as pd
from src.features.features_config import get_feature_config
from src.features.transformations import apply_transformations

# this function is used to build the dataset that 
# the models will use by compute basic transformations of each features like MA or DIFF
# /!\ the transformation are computed by a daily it's mean that for a fomc date, we compute for instance, the EMA7 for vix 
# 7 day sbefore the meeting date. 

def build_dataset_daily_transform(config):

    # beginning by reading fomc meeting date because we only select data at the meeting date not at each day of the year as explain before
    df_meetings = pd.read_csv('./data/processed/DFF_PROCESSED.csv')
    df_meetings['observation_date'] = pd.to_datetime(df_meetings['observation_date'])

    all_features = []

    for feature_name, conf in config.items():
        print(f"[INFO] Building features for {feature_name} ...")

        # contains all the value of the feature , it's a raw dataset
        df_feature = pd.read_csv(f'./data/raw/{feature_name}.csv')
        df_feature['observation_date'] = pd.to_datetime(df_feature['observation_date'])

        # compute the transformation of the dataset as defined on the file features_config
        df_feature = apply_transformations(df_feature, feature_name, conf)

        # extract all the last values of the feature transformed at the current date which is read in the loop
        feature_rows = []
        
        for date in df_meetings['observation_date']:
            # select all the values of the features before the meeting date
            subset = df_feature[df_feature['observation_date'] < date]
            if subset.empty:
                continue
            # but get only the last one because it's the last update of the feature, the most recent
            last_row = subset.iloc[-1]
            
            # build the dataset at each meeting point with the transformations of the features known at this date
            row_data = {'observation_date': date}
            for col in df_feature.columns:
                if col != 'observation_date':
                    row_data[col] = last_row[col]
            feature_rows.append(row_data)

        df_var_features = pd.DataFrame(feature_rows)
        all_features.append(df_var_features)

    # merge all the variables (UNRATE , VIX, and so on at the correct meeting )
    dataset = all_features[0]
    for df_next in all_features[1:]:
        dataset = pd.merge(dataset, df_next, on='observation_date', how='inner')


    # Targeting - Cleaning NA 
    dataset.ffill(inplace=True)
    dataset.drop('DFF_DIFF1',axis=1,inplace=True)
    dataset.to_csv('./data/processed/DATASET_FINAL_DAILY_BASED.csv',index=False)
    
    print("[INFO] Dataset built successfully.")
    
    return dataset
      
      
      
      
# ___________________________________________________________________________________________________________

      


# thi snew version of building dataset , build the dataset by add only the last known valu eof feature at the the date 
# of the fomc meeting. It's mean that we get th evalue of the vix the day before the meeting and we do that for each fomc
# date then we build new features (diff, ma and so on) from meeting to meeting whereas in the last version we buil ddirectly
# the features daily whereas we want to predict a decision from meeting to meeting so we must align the data meeting by meeting too

def build_dataset_meeting_transform(config):

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

    dataset.to_csv('./data/processed/DATASET_FINAL_MEETING_BASED.csv', index=False)
    print("[INFO] Dataset built successfully with meeting-to-meeting transformations.")
    return dataset


    
        
daily = 0
if (daily):
    build_dataset_daily_transform(config=get_feature_config())
else:
    build_dataset_meeting_transform(config=get_feature_config())   
    
  
    
    
    
    
    
