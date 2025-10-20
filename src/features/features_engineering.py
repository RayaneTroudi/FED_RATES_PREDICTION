import pandas as pd
from transformations import apply_transformations
from features_config import get_feature_config

# this function is used to build the dataset that 
# the models will use by compute basic transformations of each features like MA or DIFF

def build_dataset(config):

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

    dataset.sort_values('observation_date', inplace=True)
    dataset.dropna(axis=0, how='any', inplace=True)

    print("[INFO] Dataset built successfully.")
    return dataset
            
        
        
    
build_dataset(config=get_feature_config())
    
    
  
    
    
    
    
    
