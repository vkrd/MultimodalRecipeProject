import pandas as pd
import numpy as np

df = pd.read_csv("../data/preprocessed_data/data.csv")

# Randomly sample 200 recipes
df_init = df[df["split"] == "train"]

df_init = df_init.sample(n=200, random_state=1)

rec1 = []
rec2 = []

for i in range(100):
    rec1.append(df_init.iloc[i]["id"])
    rec2.append(df_init.iloc[i+100]["id"])

recipe_comparison = pd.DataFrame({"id": [i+1 for i in range(100)], "rec1": rec1, "rec2": rec2})

recipe_comparison.to_csv("recipe_comparison.csv", index=False)