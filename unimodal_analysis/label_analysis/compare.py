import boto3
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# Create a Bedrock client
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

# Specify the model identifier
modelId = 'anthropic.claude-v2'

# Specify the content type and accept type
contentType = 'application/json'
accept = 'application/json'


def call_LLM(ing1, rec1, ing2, rec2):
    prompt = """
    \n\nHuman: 
    Closely take a look at these two recipes and ingredients list. After reading them, determine which one is more nutritious and healthy overall. While their may be nuances, please use your best judgement and only output 1 for recipe 1 and 2 for recipe 2.
    
    Recipe 1:
    Ingredients: """ + ing1 + """
    Recipe: """ + rec1 + """

    Recipe 2:
    Ingredients: """ + ing2 + """
    Recipe: """ + rec2 + """
    \n\nAssistant:
    """

    # Define the prompt
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 100,
        "temperature": 0.1,
        "top_p": 0.3
    })

    # Invoke the model
    response = bedrock.invoke_model(
        body=body,
        modelId=modelId,
        contentType=contentType,
        accept=accept
    )

    s = b""

    for strr in response["body"]:
        s += strr

    model_output = json.loads(s)

    #return model_output["completions"][0]["data"]["text"]
    return model_output["completion"]

np.random.seed(0)

number_of_samples = 25

def make_ingredients_list(ingredients):
    ingredients_list = ""
    for j in ingredients:
        ingredients_list += j['text'] + "\n"
    return ingredients_list

def make_instruction_list(instructions):
    instructions_list = ""
    for idx, j in enumerate(instructions):
        instructions_list += str(idx+1) + ") " + j['text'] + "\n"
    return instructions_list

def evaluate_judgement(judgement):
    # Does 1 or 2 appear first?
    ind1 = judgement.find("1")
    ind2 = judgement.find("2")
    if ind1 == -1:
        return 2
    elif ind2 == -1:
        return 1
    return 1 if ind1 < ind2 else 2

verbose = False

with open('../../data/recipe1M_layers/layer1.json') as json_file:
    data = json.load(json_file)
    recipe_keys = np.random.choice(len(data), number_of_samples*2, replace=False)
    ind1 = []
    ind2 = []
    ing1s = []
    ing2s = []
    rec1s = []
    rec2s = []
    raw_judgement = []
    all_judgements = []

    iterator = tqdm(range(number_of_samples)) if not verbose else range(number_of_samples)
    for i in iterator:
        ind1.append(recipe_keys[i*2])
        ind2.append(recipe_keys[i*2 + 1])

        ing1 = make_ingredients_list(data[recipe_keys[i*2]]['ingredients'])
        ing2 = make_ingredients_list(data[recipe_keys[i*2 + 1]]['ingredients'])
        ing1s.append(ing1)
        ing2s.append(ing2)

        rec1 = make_instruction_list(data[recipe_keys[i*2]]['instructions'])
        rec2 = make_instruction_list(data[recipe_keys[i*2 + 1]]['instructions'])
        rec1s.append(rec1)
        rec2s.append(rec2)
        if verbose:
            print(i, "-"*45)
            print("INGREDIENTS 1: ")
            print(ing1)
            print("RECIPE 1: ")
            print(rec1)

            print("INGREDIENTS 2: ")
            print(ing2)
            print("RECIPE 2: ")
            print(rec2)

            print()
        
        raw_judgement.append(call_LLM(ing1, rec1, ing2, rec2))
        all_judgements.append(evaluate_judgement(raw_judgement[-1]))

        if verbose:
            print("RECIPE", all_judgements[-1], "IS HEALTHIER")
                
            print("-"*50)
            print()

df = pd.DataFrame({
    "Index 1": ind1,
    "Index 2": ind2,
    "Ingredients 1": ing1s,
    "Ingredients 2": ing2s,
    "Recipe 1": rec1s,
    "Recipe 2": rec2s,
    "Raw Judgement": raw_judgement,
    "Healthier": all_judgements
})

df.to_csv("compare.csv", index=False)