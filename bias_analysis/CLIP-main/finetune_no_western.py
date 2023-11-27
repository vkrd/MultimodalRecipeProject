import torch
import clip
from PIL import Image
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm
import sys, os
import pandas as pd
import numpy as np
import json

def get_embedding(preprocessed_image, recipe, ingredients, model, image_preprocessor=None, device="cuda"):
        preprocessed_image = image_preprocessor(Image.open("/home/ubuntu/data/preprocessed_data/" + preprocessed_image)).unsqueeze(0)
        pre = preprocessed_image.to(device)
        image_feature = model.encode_image(pre)[0]

        ingredients = ingredients.split('/t')
        instructions = recipe.split('/t')
        instructions_list = ""
        for idx, inst in enumerate(instructions):
            instructions_list += str(idx + 1) + ") " + inst + "\n"
        
        recipe = "\n".join([
            "Ingredients: ",
            "\n".join(ingredients),
            "\nInstructions: ",
            instructions_list
        ])

        text_input = clip.tokenize([recipe], context_length=10_000, truncate=False)
        
        total_cnt = 0
        inputs = []
        for i in range(text_input.shape[1]//77):
            value = (text_input[0, i*77:(i+1)*77] != 0).sum().item()
            if value == 0:
                break
            inputs.append(text_input[:, i*77:(i+1)*77])
            total_cnt += 1
            
        inputs = torch.cat(inputs).to(device)
        
        text_feature = model.encode_text(inputs).mean(dim=0)

        return image_feature, text_feature
    
if __name__ == "__main__":
    # Config that should only change once
    DATA_PATH = "/home/ubuntu/data/"
    DEBUG = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and freeze appropriate layers
    model, preprocess = clip.load("ViT-B/32", device = device)

    for name, param in model.named_parameters():
        if name in ['text_projection', 'logit_scale']:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Load data
    df = pd.read_csv(DATA_PATH  + "preprocessed_data/data.csv")

    with open('data/id_to_cuisine_llama.json') as f:
        id_to_label = json.load(f)

    train_split = "val" if DEBUG else "train"
    train_dataset = RecipeDataset(df, split = train_split, device = device, data_path = DATA_PATH, image_preprocessor = preprocess)
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, weight_decay=0.1, eps=0.000001)
    criteron = torch.nn.CrossEntropyLoss()

    batch_size = 32

    for epoch in range(5):
        print("Starting epoch", epoch + 1)
        running_loss = 0.0
        model.train()
        for rows in tqdm(train_dataloader):
            this_batch_size = len(rows["image_path"])
            image_features, text_features = torch.zeros((this_batch_size, 512)).to(device), torch.zeros((this_batch_size, 512)).to(device)
            for i in range(this_batch_size):
                # pre = rows["image"][i]
                pre = rows["image_path"][i]
                recipe = rows["recipe"][i]
                ingredients = rows["ingredients"][i]

                image_embeddings, text_embeddings = get_embedding(pre, recipe, ingredients, model, image_preprocessor=preprocess, device=device)
                image_features[i, :] = image_embeddings
                text_features[i, :] = text_embeddings

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            labels = torch.arange(this_batch_size).long().to(device)

            image_loss = criteron(logits_per_image, labels)
            text_loss = criteron(logits_per_text, labels)

            loss = (image_loss + text_loss)/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if DEBUG:
                break
        
        print("Epoch", epoch + 1, "train loss:", running_loss/len(train_dataloader))
        torch.save(model.state_dict(), "full_finetune_" + str(epoch) + "_model.pt")
        if DEBUG:
            break