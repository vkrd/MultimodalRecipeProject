import pandas as pd
from tqdm import tqdm
from transformers import (
    LogitsProcessor,
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    AutoModelForCausalLM,
)
import os
import torch
import time
import ctranslate2
from transformers import AutoTokenizer
import numpy as np
import wandb




SOURCE_LAMBDA = 0.0
QUERY_LAMBA = 0.0
CSV_NAME = "zero_shot_clip_" + str(SOURCE_LAMBDA) + "/" + str(QUERY_LAMBA) + "_image_importance_results.csv"

wandb.init(
    # set the wandb project where this run will be logged
    project="CLIP_Eval",
    name="source_{}_query_{}".format(SOURCE_LAMBDA, QUERY_LAMBA),
    # track hyperparameters and run metadata
    config={
        "source_lambda":SOURCE_LAMBDA,
        "query_lambda": QUERY_LAMBA,
        "model": "CLIP"
    }
)


dataframe = pd.read_csv("../../data/preprocessed_data/data.csv")
dataframe.head()

df_pool = dataframe[dataframe['split']=='val']
df_query = dataframe[dataframe['split']=='test']

df_result = pd.read_csv("../../data/index_outputs/" + CSV_NAME)
df_result = df_result.rename(columns={'input_id': 'id_query', 'predicted_id': 'id_pool', 'distance': 'similarity'})

prompts = []
for index, row in tqdm(df_result.iterrows(), total=df_result.shape[0]):
    id_query = row['id_query']
    id_pool = row['id_pool']

    # Match id_query to df_query
    query_match = df_query[df_query['id'] == id_query].iloc[0]

    # Match id_pool to df_pool's 'id'
    pool_match = df_pool[df_pool['id'] == id_pool].iloc[0]

    # Process the 'instructions' column
    instructions_1 = query_match['instructions'].split('/t')
    instructions_list_1 = ""
    for idx, inst in enumerate(instructions_1):
        instructions_list_1 += str(idx + 1) + ") " + inst + " "

    instructions_2 = pool_match['instructions'].split('/t')
    instructions_list_2 = ""
    for idx, inst in enumerate(instructions_2):
        instructions_list_2 += str(idx + 1) + ") " + inst + " "

    # print(instructions_list)

    # Process the 'ingredients' column
    ingredients_1 = query_match['ingredients'].replace(' /t', ',')
    ingredients_2 = pool_match['ingredients'].replace(' /t', ',')
    # print(ingredients)

    # construct prompt
    prompt = """
    ### Question:
    Closely take a look at the following two recipes, which have a ingredients list (seperated by ",") and Recipe Instructions (Listed in order). After reading them, determine which one is more nutritious and healthy overall. While their may be nuances, please use your best judgement and only output 1 for recipe 1 and 2 for recipe 2. Please make your answers concise and you must make a decision. Begin your answer with "Recipe {} is more nutritious and healthy overall".

    Recipe 1:
    Ingredients: """ + ingredients_1 + """
    Recipe Instructions: """ + instructions_list_1 + """

    Recipe 2:
    Ingredients: """ + ingredients_2 + """
    Recipe Instructions: """ + instructions_list_2 + """

    ### Answer:
    """
    prompts.append(prompt)
    # print(prompt)

base_path = "meta-llama/Llama-2-7b-chat-hf"       #for 7B
path = "content/drive/MyDrive/llama7b_ct2/"     #for 7B

tokenizer = AutoTokenizer.from_pretrained(base_path, token="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")

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = ctranslate2.Generator(path, device=device)

generation_config = {
    "no_repeat_ngram_size": 10,
    "min_length": 10,
    "max_length": 50,
    "length_penalty": -2.0,
    "beam_size": 1,
    "sampling_temperature": 0.0,
    "repetition_penalty": 1.05,
    "include_prompt_in_result": False,
    "sampling_topp": 0.1
}

raw_answers = []

batch_size = 1

# generate answers
# for prompt in tqdm(prompts):
#   inputs = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt, truncation=False))
#   output = generator.generate_batch([inputs], **generation_config, return_scores=True,)
#   result = tokenizer.decode(output[0].sequences_ids[0])
#   raw_answers.append(result)

for i in tqdm(range(0, len(prompts), batch_size)):
  batch_prompts = prompts[i:min(i+batch_size, len(prompts))]
  batch_ids = [tokenizer.encode(prompt, truncation=False) for prompt in batch_prompts]
  batch_inputs = [tokenizer.convert_ids_to_tokens(ids) for ids in batch_ids] 

  output = generator.generate_batch(batch_inputs, **generation_config, return_scores=True,)

  for j in range(batch_size):
      result = tokenizer.decode(output[j].sequences_ids[0])
      raw_answers.append(result)
      
def evaluate_judgement(judgement):
    # Does 1 or 2 appear first?
    ind1 = judgement.find("1")
    ind2 = judgement.find("2")
    if ind1 == -1:
        return 2
    elif ind2 == -1:
        return 1
    return 1 if ind1 < ind2 else 2

final_answers = []

for raw_answer in raw_answers:
  final_answer = evaluate_judgement(raw_answer)
  final_answers.append(final_answer)

df_results = pd.DataFrame({"final_answers": final_answers, "raw_answer": raw_answer})

os.makedirs("../../data/results", exist_ok=True)
df_results.to_csv("../../data/results/source_{}_query_{}.csv".format(SOURCE_LAMBDA, QUERY_LAMBA), index=False)

print("SOURCE LAMBDA", SOURCE_LAMBDA)
print("QUERY LAMBDA", QUERY_LAMBA)
print("IMPROVEMENT", sum([i == 2 for i in final_answers])/len(final_answers))
print("AVG SIM", np.mean(df_result['similarity']))
print("DIVERSITY", (len(set(df_pool)))/4372)

wandb.log({
    "improvement": sum([i == 2 for i in final_answers])/len(final_answers),
    "avg_sim": np.mean(df_result['similarity']),
    "diversity": (len(set(df_result["id_pool"])))/4372
    })