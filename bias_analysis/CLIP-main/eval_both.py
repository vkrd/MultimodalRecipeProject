import torch
import clip
from PIL import Image
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss

with open('data/id_to_cuisine_llama.json') as f:
    id_to_label = json.load(f)

print(len(id_to_label))

device = "cuda" if torch.cuda.is_available() else "cpu"
model_western, preprocess = clip.load("ViT-B/32", device=device)
model_western.load_state_dict(torch.load("western_4_model.pt"))

model_nonwestern, preprocess = clip.load("ViT-B/32", device=device)
model_nonwestern.load_state_dict(torch.load("nonwestern_4_model.pt"))

EXPERIMENT_NAME = "western"
IMAGE_IMPORTANCE = 0.5
IMAGE_FILE_LOCATIONS = "/home/ubuntu/data/preprocessed_data/"
SPLIT = "test"
LOAD = False


OPACITY = 0.3

def get_embedding(image_paths, recipe, ingredients, image_importance=0.5, western=True):
    model = model_western if western else model_nonwestern
    with torch.no_grad():
        if image_importance == 0.0:
            image_feature = np.zeros(512)
        else:
            images = np.zeros((512,))
            for image_path in image_paths:
                pre = preprocess(Image.open(IMAGE_FILE_LOCATIONS + image_path)).to(device).unsqueeze(0)
                images += model.encode_image(pre).cpu().detach().numpy()[0]

            image_feature = images/len(image_paths)

        if image_importance == 1.0:
            text_feature = np.zeros(512)
        else:
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

            text_input = clip.tokenize([recipe], context_length=10_000, truncate=False).to(device)
            
            text_feature = np.zeros((512,))
            total_cnt = 0
            for i in range(text_input.shape[1]//77):
                value = (text_input[0, i*77:(i+1)*77] != 0).sum().item()
                if value == 0:
                    break
                text_feature += model.encode_text(text_input[:, i*77:(i+1)*77])[0].detach().cpu().numpy()*value
                total_cnt += value
                
            text_feature = text_feature/total_cnt
            
            return image_importance * image_feature + (1 - image_importance) * text_feature

df = pd.read_csv("/home/ubuntu/data/preprocessed_data/data.csv")

# Only keep set we care about
df_init = df[df["split"] == SPLIT]#[:200]

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

westerns = []
nonwesterns = []

if not LOAD:
    for IMAGE_IMPORTANCE in [0.5]:
        western = []
        nonwestern = []

        X_reference = new_df

        ids = []

        for row in tqdm(X_reference.itertuples(), total=len(X_reference)):
            image_paths = row.image_path.split(",")
            recipe = row.recipe
            ingredients = row.ingredients
            
            if id_to_label[row.id] == 1:
                embedding = get_embedding(image_paths, recipe, ingredients, image_importance=IMAGE_IMPORTANCE, western=True)
                western.append(embedding)
            else:
                embedding = get_embedding(image_paths, recipe, ingredients, image_importance=IMAGE_IMPORTANCE, western=False)
                nonwestern.append(embedding)

        western = np.array(western)
        nonwestern = np.array(nonwestern)

        westerns.append(western)
        nonwesterns.append(nonwestern)

        # save numpy array
        np.save('./arrays/' + EXPERIMENT_NAME + '_' + str(IMAGE_IMPORTANCE) + '_western.npy', western)
        np.save('./arrays/' + EXPERIMENT_NAME + '_' + str(IMAGE_IMPORTANCE) + '_nonwestern.npy', nonwestern)
else:
    for IMAGE_IMPORTANCE in [0.5]:
        westerns.append(np.load('./arrays/' + EXPERIMENT_NAME + '_' + str(IMAGE_IMPORTANCE) + '_western.npy'))
        nonwesterns.append(np.load('./arrays/' + EXPERIMENT_NAME + '_' + str(IMAGE_IMPORTANCE) + '_nonwestern.npy'))

with torch.no_grad():
    healthy = clip.tokenize(["healthy food"]).to(device)
    healthy_embedding_western = model_western.encode_text(healthy).cpu().numpy()
    healthy_embedding_nonwestern = model_nonwestern.encode_text(healthy).cpu().numpy()

for i, IMAGE_IMPORTANCE in enumerate([0.5]):
    print("\n\nIMAGE_IMPORTANCE: ", IMAGE_IMPORTANCE)
    western = westerns[i]
    nonwestern = nonwesterns[i]

    print("Western shape: {}".format(western.shape))
    print("Non-western shape: {}".format(nonwestern.shape))
    print("Fraction of western: {}".format(len(western)/(len(western) + len(nonwestern))))


    # western_cuttoff = len(western)

    # all = np.concatenate((western, nonwestern), axis=0)

    # Calculate closest 100 neighbors in western
    index_western = faiss.IndexFlatIP(512)
    all_normalized = western / np.linalg.norm(western, axis=1)[:, None]
    index_western.add(all_normalized)
    healthy_normalized = healthy_embedding_western / np.linalg.norm(healthy_embedding_western)

    D_western, I_western = index_western.search(healthy_normalized, 4455//10)

    index_nonwestern = faiss.IndexFlatIP(512)
    all_normalized = nonwestern / np.linalg.norm(nonwestern, axis=1)[:, None]
    index_nonwestern.add(all_normalized)

    D_nonwestern, I_nonwestern = index_nonwestern.search(healthy_embedding_nonwestern / np.linalg.norm(healthy_embedding_nonwestern), 4455//10)

    print(D_nonwestern[:10])
    curr_western = 0
    curr_nonwestern = 0

    western_count = 0

    for i in range(4455//10):
        if D_western[0][curr_western] < D_nonwestern[0][curr_nonwestern]:
            curr_nonwestern += 1
        else:
            curr_western += 1
            western_count += 1

    print("Fraction of closest neighbors that are western with IP: {}".format(western_count/(4455//10)))

