import pandas as pd
from src.features.features_config_daily import get_feature_config
from src.features.transformations import apply_transformations

# this function is used to build the dataset that 
# the models will use by compute basic transformations of each features like MA or DIFF

def build_dataset_daily(config):

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
    dataset.fillna(dataset.mean(), inplace=True)
    dataset.drop('DFF_DIFF1',axis=1,inplace=True)
    dataset.to_csv('./data/processed/DAILY_BASED.csv',index=False)
    
    print("[INFO] Dataset built successfully.")
    
    return dataset
            
build_dataset_daily(config=get_feature_config())

    
    
    
    
    