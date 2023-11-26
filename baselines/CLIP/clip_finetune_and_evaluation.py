# -*- coding: utf-8 -*-
"""
Reference:
    https://github.com/openai/CLIP
    https://github.com/openai/CLIP/issues/83
    https://github.com/shashnkvats/Indofashionclip/
"""

!pip install transformers
!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git

import pandas as pd
from tqdm import tqdm
import numpy as np

import json
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision
import torchvision.transforms as transforms

import clip
from transformers import CLIPProcessor, CLIPModel

from joblib import dump
import matplotlib.pyplot as plt

from IPython.display import display, Markdown

dataframe = pd.read_csv("data.csv")
dataframe.head()

dataframe['combined'] = "Ingredients: \n" + dataframe['ingredients'] + "\n\nRecipe: \n" + dataframe['instructions']

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and processor
model_hug = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model, preprocess = clip.load("ViT-B/32", device=device)

model = model.to(device)

clip.available_models()

"""# Finetune"""

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None, split="train"):
        if split=="train":
          dataframe = dataframe[(dataframe['split'] == split) & (dataframe.index % 4 == 0)]
        else:
          dataframe = dataframe[dataframe['split'] == split]

        self.image_files = dataframe["image_path"].to_list()
        self.transform = transform
        self.dataframe = dataframe
        self.texts = clip.tokenize(dataframe['combined'], context_length=77, truncate=True)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = preprocess(Image.open(img_path))

        if self.transform:
            image = self.transform(image)
        text = self.texts[idx]


        return image,text

train_data = CustomDataset(dataframe=dataframe, split="train")
val_data = CustomDataset(dataframe=dataframe, split="val")
test_data = CustomDataset(dataframe=dataframe, split="test")

config = {
    'epochs'        : 10,
    'batch_size'    : 256,
}

train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    num_workers = 4,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = True
)

val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    num_workers = 2,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False
)

test_loader = torch.utils.data.DataLoader(
    dataset     = test_data,
    num_workers = 2,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False
)


print("Batch size     : ", config['batch_size'])

print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Validation dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the smaller lr, safer for fine tuning to new dataset
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*config['epochs'])
# specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

best_te_loss = 1e5
best_ep = -1

def train_val(model, train_loader, val_loader, optimizer, loss_img, loss_txt, best_te_loss, best_ep, begin_epoch=0):

    # Train the model
    num_epochs = config['epochs']
    for epoch in range(begin_epoch, num_epochs):
        print(f"running epoch {epoch}, best test loss {best_te_loss} after epoch {best_ep}")
        pbar = tqdm(train_loader, total=len(train_loader))
        model.train()
        total_train_loss = 0
        step = 0
        for batch in pbar:
            step += 1
            optimizer.zero_grad()

            images,texts = batch

            images= images.to(device)
            texts = texts.to(device)

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # Compute batch loss
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device).to(device)
            loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

            # Backward pass
            loss.backward()

            total_train_loss += loss.item()
            if device == "cpu":
                optimizer.step()
            else :
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Train Batch Loss: {loss.item():.4f}")

            ### Release memory
            del images, texts, logits_per_image, logits_per_text
            torch.cuda.empty_cache()

        ave_train_loss = total_train_loss / step # average training loss for the epoch


        step = 0
        total_val_loss = 0 # total val loss for the epoch
        with torch.no_grad():
            model.eval()
            val_pbar = tqdm(val_loader, leave=False)
            for batch in val_pbar:
                step += 1
                images,texts = batch

                images= images.to(device)
                texts = texts.to(device)

                logits_per_image, logits_per_text = model(images, texts)
                ground_truth = torch.arange(len(images),dtype=torch.long,device=device).to(device)

                loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                total_val_loss += loss.item()
                val_pbar.set_description(f"Val Batch Loss: {loss.item()}", refresh=True)

            ave_val_loss = total_val_loss / step # average val loss for the epoch

        if ave_val_loss < best_te_loss:
            best_te_loss = ave_val_loss
            best_ep = epoch
            # torch.save(model.state_dict(), f"best_model_epoch_{best_ep}.pt")
            torch.save({
              'best_epoch': best_ep,
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'ave_train_loss': ave_train_loss,
              'ave_val_loss': ave_val_loss,
              },  f"best_model_epoch_{best_ep}.pt")
            print(f"best_model_saved")

        print(f"epoch {epoch}, tr_loss {ave_train_loss}, te_loss {ave_val_loss}")

train_val(model, train_loader, val_loader, optimizer, loss_img, loss_txt,best_te_loss, best_ep)

# model.load_state_dict(torch.load("best_model_epoch_4.pt"))
# Load the model state from the saved checkpoint
checkpoint = torch.load("best_model_epoch_4.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Optional: Load other information from the checkpoint
best_epoch = checkpoint['best_epoch']
epoch = checkpoint['epoch']
ave_train_loss = checkpoint['ave_train_loss']
ave_val_loss = checkpoint['ave_val_loss']

"""# Evaluation"""

IMAGE_IMPORTANCE = 1

# val split: pool
df_init = dataframe[dataframe["split"] == "val"]

print("Making index with {} images and {} image importance".format(len(df_init), IMAGE_IMPORTANCE))

new_df = pd.DataFrame(columns=["id", "image_path", "recipe", "ingredients"])
# recipe id; image_path; recipe_instruction; ingredients
seen = set()

print("Combining images with same id")

for row in tqdm(df_init.itertuples(), total=len(df_init)):
    if row.id not in seen:
        new_row = {"id": row.id, "image_path": row.image_path, "recipe": row.instructions, "ingredients": row.ingredients}
        new_df = pd.concat([new_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        seen.add(row.id)
    else:
        new_df.loc[new_df["id"] == row.id, "image_path"] += "," + row.image_path

X_pool = new_df

# test split: test retrieval
df_init = dataframe[dataframe["split"] == "test"]

print("Making index with {} images and {} image importance".format(len(df_init), IMAGE_IMPORTANCE))

new_df = pd.DataFrame(columns=["id", "image_path", "recipe", "ingredients"])
# recipe id; image_path; recipe_instruction; ingredients
seen = set()

print("Combining images with same id")

for row in tqdm(df_init.itertuples(), total=len(df_init)):
    if row.id not in seen:
        new_row = {"id": row.id, "image_path": row.image_path, "recipe": row.instructions, "ingredients": row.ingredients}
        new_df = pd.concat([new_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        seen.add(row.id)
    else:
        new_df.loc[new_df["id"] == row.id, "image_path"] += "," + row.image_path

X_test = new_df

# calculate embedding function: no need to change image but need to change text
def get_embedding(image_paths, recipe, ingredients, image_importance=1):
    def split_long_text_into_chunks(text, max_length=77):
        tokenized_text = processor(text, truncation=False, padding='max_length', max_length=77, return_tensors='pt')
        text_length = tokenized_text['input_ids'].shape[1]
        batches = text_length//max_length if text_length%max_length == 0 else text_length//max_length + 1
        tokenized_text = processor(text, truncation=False, padding='max_length', max_length=77*batches, return_tensors='pt')

        return_dict = dict()
        return_dict['input_ids'] = tokenized_text['input_ids'].reshape(batches, max_length)
        return_dict['attention_mask'] = tokenized_text['attention_mask'].reshape(batches, max_length)

        weight = [return_dict['attention_mask'][i].sum().item() for i in range(batches)]
        return return_dict, np.array(weight)/sum(weight) # the tokenizor is the same as the clip.tokenizor (but can tackle with longer input)


    with torch.no_grad():
        if image_importance == 0.0:
            image_feature = torch.zeros(512).to(device)
        else:
            image_embedding_list = []
            for image_path in image_paths:
              image_em = preprocess(Image.open(image_path)).unsqueeze(0).to(device) # get one
              image_em = model.encode_image(image_em)
              image_embedding_list.append(image_em)

            image_features_all = torch.cat(image_embedding_list, dim=0)
            image_feature = image_features_all.mean(axis=0)

        if image_importance == 1.0:
            text_feature = torch.zeros(512).to(device)
        else:
            text = "".join([
                "Ingredients: \n",
                ingredients,
                "\n\nRecipe: \n",
                recipe
            ])

            input_tokens, weight = split_long_text_into_chunks(text) # unchanged: as only related to the tokenizor
            n = input_tokens['input_ids'].shape[0]  # Get the number of samples
            text_features_list = []

            for i in range(n):
                input_token_sequence = (input_tokens['input_ids'][i].unsqueeze(0)).to(device)
                text_features = model.encode_text(input_token_sequence)
                text_features_list.append(text_features)

            text_features_tensor = torch.cat(text_features_list, dim=0)
            weight_tensor = torch.tensor(weight, device=text_features_tensor.device, dtype=text_features_tensor.dtype)
            text_feature = torch.matmul(text_features_tensor.T, weight_tensor)

        return (image_importance * image_feature + (1 - image_importance) * text_feature).detach().cpu().numpy().flatten()

# calcultae embedding for X_pool
pool_embeddings = []
for row in tqdm(X_pool.itertuples(), total=len(X_pool)):
    image_paths = row.image_path.split(",")
    recipe = row.recipe
    ingredients = row.ingredients
    embedding = get_embedding(image_paths, recipe, ingredients, IMAGE_IMPORTANCE) # (512,)
    pool_embeddings.append(embedding)

# # Saving the pool_embeddings to a compressed file
# dump(pool_embeddings, 'pool_embeddings_image_1.pkl', compress=('gzip', 6))

# Loading the pool_embeddings from the compressed file
loaded_pool_embeddings = load('pool_embeddings_image_0.pkl')
(len(loaded_pool_embeddings), loaded_pool_embeddings[0].shape)

text_modification = clip.tokenize("healthy").to(device)

with torch.no_grad():
    modification = model.encode_text(text_modification)

modification = modification.detach().cpu().numpy().flatten()

# calcultae embedding for X_test (need a further step to + add modification)
query_embeddings = []
for row in tqdm(X_test.itertuples(), total=len(X_test)):
    image_paths = row.image_path.split(",")
    recipe = row.recipe
    ingredients = row.ingredients
    embedding = get_embedding(image_paths, recipe, ingredients, IMAGE_IMPORTANCE) # (512,)
    target_embedding = embedding + modification
    query_embeddings.append(target_embedding)

# # Saving the pool_embeddings to a compressed file
# dump(pool_embeddings, 'pool_embeddings.pkl', compress=('gzip', 6))

# Saving the compress=('gzip', 6) to a compressed file
dump(query_embeddings, 'query_embeddings_image_0.pkl', compress=('gzip', 6))

from sklearn.metrics.pairwise import cosine_similarity

def find_highest_similarity(list_of_embeddings, test_embeddings):
    result_data = []

    # Stack the list_of_embeddings into a 2D array
    array_of_pool_embeddings = np.vstack(list_of_embeddings)

    for idx, test_embedding in tqdm(enumerate(test_embeddings), total=len(test_embeddings)):

        # Calculate cosine similarities
        similarities = cosine_similarity([test_embedding], array_of_pool_embeddings)

        # Find the index with the highest similarity
        highest_similarity_index = np.argmax(similarities)
        highest_similarity_value = similarities[0, highest_similarity_index]

        # Store the result in the DataFrame
        result_data.append({'idx': highest_similarity_index, 'similarity': highest_similarity_value})

    # Create a DataFrame from the result_data list
    result_df = pd.DataFrame(result_data)

    return result_df

result_df = find_highest_similarity(loaded_pool_embeddings, query_embeddings)

# Concatenate result_df to the right side of X_pool
combined_df = pd.concat([X_test[['id']], result_df], axis=1)
combined_df

combined_df = combined_df.rename(columns={"id": "id_query"})
combined_df['id_pool'] = combined_df['idx'].apply(lambda x: X_pool.loc[x, 'id'])
combined_df = combined_df[['id_query', 'id_pool', 'similarity']]

# Save the combined DataFrame to a CSV file
combined_df.to_csv('finetuned_result_image_0.csv', index=False)

"""# Sanity Check"""

def imshow(img):

    plt.figure(figsize=(15,15))
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

check_df = combined_df.loc[:5]
check_df

def get_first_image_path(image_path_str):
    """
    Extracts the first image path from a comma-separated string.
    """
    return image_path_str.split(',')[0]

# Iterate through each row in check_df
all_images = []

for _, row in check_df.iterrows():
    # Fetch image paths
    img_path_query = get_first_image_path(X_test[X_test['id'] == row['id_query']]['image_path'].values[0])
    img_path_pool = get_first_image_path(X_pool[X_pool['id'] == row['id_pool']]['image_path'].values[0])

    # Load and preprocess images
    img_query = preprocess(Image.open(img_path_query))
    img_pool = preprocess(Image.open(img_path_pool))

    # Stack the images side by side (horizontally)
    pair_tensor = torch.cat((img_query, img_pool), dim=-1)
    all_images.append(pair_tensor)

# Stack all pairs vertically
final_image_tensor = torch.cat(all_images, dim=1) # Concatenate vertically

# Show the final image
imshow(final_image_tensor)

def get_first_image_path(image_path_str):
    """
    Extracts the first image path from a comma-separated string.
    """
    return image_path_str.split(',')[0]

def display_ingredients_comparison(ingredients_query, ingredients_pool):
    """
    Display ingredients of two recipes side-by-side for comparison in a markdown table.
    """
    # Replace tabs with commas for better visual representation
    ingredients_query = ingredients_query.replace("/t ", ", ")
    ingredients_pool = ingredients_pool.replace("/t ", ", ")

    # Create a markdown table string
    comparison_str = f"| **Query Ingredients** | **Pool Ingredients** |\n"
    comparison_str += "|:----------------------:|:---------------------:|\n"
    comparison_str += f"| {ingredients_query} | {ingredients_pool} |"

    # Display using markdown
    display(Markdown(comparison_str))

for _, row in check_df.iterrows():
    # Fetch ingredients
    ingredients_query = X_test[X_test['id'] == row['id_query']]['ingredients'].values[0]
    ingredients_pool = X_pool[X_pool['id'] == row['id_pool']]['ingredients'].values[0]

    # Display the ingredients comparison
    display_ingredients_comparison(ingredients_query, ingredients_pool)

