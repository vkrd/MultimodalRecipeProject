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

# Save in pretty format
with open("human_readable_recipes.txt".format(i+1), "w") as f:
    for i in range(100):
        rec1_instructions = df[df["id"] == recipe_comparison.iloc[i]["rec1"]]["instructions"].values[0]
        rec2_instructions = df[df["id"] == recipe_comparison.iloc[i]["rec2"]]["instructions"].values[0]

        rec1_ingredients = df[df["id"] == recipe_comparison.iloc[i]["rec1"]]["ingredients"].values[0]
        rec2_ingredients = df[df["id"] == recipe_comparison.iloc[i]["rec2"]]["ingredients"].values[0]

        rec1_title = df[df["id"] == recipe_comparison.iloc[i]["rec1"]]["food_title"].values[0]
        rec2_title = df[df["id"] == recipe_comparison.iloc[i]["rec2"]]["food_title"].values[0]

        recipe1 = "\n".join([
            "RECIPE 1 - " + rec1_title,
            "Ingredients: ",
            "\n".join(rec1_ingredients.split("/t")),
            "\nInstructions: ",
            "\n".join(rec1_instructions.split("/t"))
        ])

        recipe2 = "\n".join([
            "RECIPE 2 - " + rec2_title,
            "Ingredients: ",
            "\n".join(rec2_ingredients.split("/t")),
            "\nInstructions: ",
            "\n".join(rec2_instructions.split("/t"))
        ])

        f.write("RECIPE COMPARISON {}\n\n".format(i+1))
        f.write(recipe1)
        f.write("\n\n\n")
        f.write(recipe2)
        f.write("\n" + "-"*100 + "\n\n\n")