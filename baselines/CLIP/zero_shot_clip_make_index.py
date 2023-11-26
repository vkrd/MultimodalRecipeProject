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

EXPERIMENT_NAME = "zero_shot_clip"
IMAGE_IMPORTANCE = 1.0
IMAGE_FILE_LOCATIONS = "../../data/preprocessed_data/"
SPLIT = "val"

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

# X_reference, X_query = train_test_split(new_df, test_size=0.2, random_state=42)

X_reference = new_df

print("Indexing reference dataset")

index = faiss.IndexFlatIP(512)
ids = []

for row in tqdm(X_reference.itertuples(), total=len(X_reference)):
    image_paths = row.image_path.split(",")
    recipe = row.recipe
    ingredients = row.ingredients
    embedding = get_embedding(image_paths, recipe, ingredients, image_importance=IMAGE_IMPORTANCE)
    embedding = embedding.cpu().detach()
    embedding = embedding / np.linalg.norm(embedding)
    index.add(np.expand_dims(embedding, axis=0))
    ids.append(row.id)

print("Saving index")

os.makedirs("../../data/index_outputs/", exist_ok=True)
save_folder = "../../data/index_outputs/{}/".format(EXPERIMENT_NAME + "_" + str(IMAGE_IMPORTANCE))
os.makedirs(save_folder, exist_ok=True)
faiss.write_index(index, save_folder + "/index.faiss")
with open(save_folder + "/ids.arr", "w") as f:
    for id in ids:
        f.write("{},".format(id))