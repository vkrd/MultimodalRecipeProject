import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

np.random.seed(0)

ALL_MEATS = [
    "beef",
    "pork",
    "chicken",
    "breast",
    "turkey",
    "lamb",
    "duck",
    "fillet",
    "bacon",
    "sausage",
    " ham",
    "meat",
    "steak",
    "fish",
    "sirloin",
    "tenderloin",
    "venison",
    "veal",
    "goat",
    "rabbit",
    "quail",
    "pheasant",
    "bison",
    "ribeye",
    "prosciutto",
    "pastrami",
    "salami",
    "pepperoni",
    "jerky",
    "chorizo",
    "tuna",
    "salmon",
    "shrimp",
    "prawn",
    "crab",
    "lobster",
    "mussel",
    "scallop",
    "squid",
    "octopus",
    "clam",
    "anchovy",
    "sardine",
    "trout",
    "cod",
    "haddock",
    "mackerel",
    "tilapia",
    "sea bass",
    "hake",
    "pollock",
    "catfish",
    "perch",
    "halibut",
    "sole",
    "swordfish",
    "snapper",
    "grouper",
    "carp",
    "fowl",
    "oyster",
    "crabmeat",
    "hot dog"
]

df = pd.read_csv("human_eval.csv")

# with open('../../data/recipe1M_layers/layer1.json') as json_file:
#     data = json.load(json_file)
#     recipe_keys = np.random.choice(len(data), 100, replace=False)
#     automated = []
#     for idx, i in enumerate(tqdm(recipe_keys)):
#         flag = False
#         ingredients = ' '.join([j['text'] for j in data[i]['ingredients']])
#         for word in ALL_MEATS:
#             if word in ingredients.lower():
#                 automated.append(0)
#                 flag = True
#                 break
#         if not flag:
#             automated.append(1)
#         if df["Human"][idx] != automated[-1]:
#             print(idx, "-"*45)
#             print("INGREDIENTS: ")
#             for j in data[i]['ingredients']:
#                 print(j['text'])
#             print("-"*50)
#             print()

#     df["Automated"] = automated
#     df.to_csv("human_eval.csv", index=False)
#     # display confusion matrix
#     print(confusion_matrix(df["Human"], df["Automated"]).ravel())
#     print("Accuracy: ", np.sum(df["Human"] == df["Automated"])/len(df))
#     print()

with open('../../data/recipe1M_layers/layer1.json') as json_file:
    data = json.load(json_file)
    recipe_keys = np.arange(len(data))
    automated = []
    for idx, i in enumerate(tqdm(recipe_keys)):
        flag = False
        ingredients = ' '.join([j['text'] for j in data[i]['ingredients']])
        for word in ALL_MEATS:
            if word in ingredients.lower():
                automated.append(0)
                flag = True
                break
        if not flag:
            automated.append(1)

    # display confusion matrix
    print(np.mean(automated))
    print(len(automated))
    print()
    