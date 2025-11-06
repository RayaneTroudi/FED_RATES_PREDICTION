import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("./data/processed/DATASET_FINAL.csv")
df = df.sort_values("observation_date").reset_index(drop=True)
print(df.head(1))

