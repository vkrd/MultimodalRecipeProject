import torch
from PIL import Image
from tqdm import tqdm
import faiss
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


EXPERIMENT_NAME = "zero_shot_clip_1.0"
IMAGE_IMPORTANCE = 1.0
IMAGE_FILE_LOCATIONS = "../../data/preprocessed_data/"
SPLIT = "test"

def get_embedding(image_paths, recipe, ingredients, image_importance=0.5):
    def split_long_text_into_chunks(text, max_length=77):
        tokenized_text = processor(text, truncation=False, padding='max_length', max_length=77, return_tensors='pt')
        text_length = tokenized_text['input_ids'].shape[1]
        batches = text_length//max_length if text_length%max_length == 0 else text_length//max_length + 1
        tokenized_text = processor(text, truncation=False, padding='max_length', max_length=77*batches, return_tensors='pt')

        return_dict = dict()
        return_dict['input_ids'] = tokenized_text['input_ids'].reshape(batches, max_length).to(device)
        return_dict['attention_mask'] = tokenized_text['attention_mask'].reshape(batches, max_length).to(device)

        weight = [return_dict['attention_mask'][i].sum().item() for i in range(batches)]
        weight = torch.tensor(weight).to(device)
        return return_dict, weight/torch.sum(weight)

    with torch.no_grad():
        if image_importance == 0.0:
            image_feature = torch.zeros(512).to(device)
        else:
            images = [Image.open(IMAGE_FILE_LOCATIONS + image_path) for image_path in image_paths]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(inputs['pixel_values'])
            image_feature = image_features.mean(dim=0)

        if image_importance == 1.0:
            text_feature = torch.zeros(512).to(device)
        else:
            text = "".join([
                "Ingredients: \n",
                ingredients,
                "\n\nRecipe: \n",
                recipe
            ])

            input_tokens, weight = split_long_text_into_chunks(text)
            text_features = model.get_text_features(**input_tokens)

            text_feature = torch.matmul(weight, text_features)
        
        return image_importance * image_feature + (1 - image_importance) * text_feature

save_folder = "../../data/index_outputs/{}/".format(EXPERIMENT_NAME)

index = faiss.read_index(save_folder + "index.faiss")

df = pd.read_csv("../../data/preprocessed_data/data.csv")

# Only keep set we care about
df_init = df[df["split"] == SPLIT]

print("Making index with {} images and {} image importance".format(len(df_init), IMAGE_IMPORTANCE))

new_df = pd.DataFrame(columns=["id", "image_path", "recipe", "ingredients"])
seen = set()

print("Combining images with same id")

for row in tqdm(df_init.itertuples(), total=len(df_init)):
    if row.id not in seen:
        new_row = {"id": row.id, "image_path": row.image_path, "recipe": row.instructions, "ingredients": row.ingredients}
        new_df = pd.concat([new_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        seen.add(row.id)
    else:
        new_df.loc[new_df["id"] == row.id, "image_path"] += "," + row.image_path

X_reference = new_df

ids = ''.join([line for line in open(save_folder + "ids.arr")]).split(",")[:-1]

predicted_ids = []
distances = []

print("Finding closest target")

with torch.no_grad():
    input_tokens = processor("healthy", truncation=False, padding='max_length', max_length=77, return_tensors='pt')
    
    input_tokens["input_ids"] = input_tokens["input_ids"].to(device)
    input_tokens["attention_mask"] = input_tokens["attention_mask"].to(device)
    modification = model.get_text_features(**input_tokens).cpu().detach()[0]
    # modification = modification / np.linalg.norm(modification)

average_distance = 0.0

for row in tqdm(X_reference.itertuples(), total=len(X_reference)):
    image_paths = row.image_path.split(",")
    recipe = row.recipe
    ingredients = row.ingredients
    embedding = get_embedding(image_paths, recipe, ingredients, IMAGE_IMPORTANCE).cpu().detach()
    target_embedding = embedding + modification
    target_embedding = target_embedding / np.linalg.norm(target_embedding)
    D, I = index.search(np.expand_dims(target_embedding, axis=0), 1)
    
    predicted_ids.append(ids[I[0][0]])
    average_distance += (1 - D[0][0])
    distances.append(1 - D[0][0])

final_df = pd.DataFrame({"input_id": X_reference["id"], "predicted_id": predicted_ids, "distance": distances})
final_df.to_csv("../../data/index_outputs/{}/{}_image_importance_results.csv".format(EXPERIMENT_NAME, IMAGE_IMPORTANCE), index=False)
print("Average distance: {}".format(average_distance/len(X_reference)))