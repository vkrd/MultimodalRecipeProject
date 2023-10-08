import json
import os
import random
import numpy as np

np.random.seed(0)

number_of_samples = 100

with open('../../data/recipe1M_layers/layer1.json') as json_file:
    data = json.load(json_file)
    recipe_keys = np.random.choice(len(data), number_of_samples, replace=False)

    for idx, i in enumerate(recipe_keys):
        print(idx, "-"*45)
        print("INGREDIENTS: ")
        for j in data[i]['ingredients']:
            print(j['text'])
            
        print("-"*50)
        print()