import torch
import clip
from PIL import Image
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

with open('data/id_to_cuisine_llama.json') as f:
    id_to_label = json.load(f)

print(len(id_to_label))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

EXPERIMENT_NAME = "zero_shot_clip"
IMAGE_IMPORTANCE = 0.0
IMAGE_FILE_LOCATIONS = "../../data/preprocessed_data/"
SPLIT = "val"
LOAD = True


OPACITY = 0.3

def get_embedding(image_paths, recipe, ingredients, image_importance=0.5):
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

            text_input = clip.tokenize([recipe], context_length=10_000, truncate=False)
            
            text_feature = np.zeros((512,))
            total_cnt = 0
            for i in range(text_input.shape[1]//77):
                value = (text_input[0, i*77:(i+1)*77] != 0).sum().item()
                if value == 0:
                    break
                text_feature += model.encode_text(text_input[:, i*77:(i+1)*77])[0].detach().numpy()*value
                total_cnt += value
                
            text_feature = text_feature/total_cnt
        
        return image_importance * image_feature + (1 - image_importance) * text_feature

df = pd.read_csv("../../data/preprocessed_data/data.csv")

# Only keep set we care about
df_init = df[df["split"] == SPLIT]#[:1000]

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
    for IMAGE_IMPORTANCE in [0.0, 0.5, 1.0]:
        western = []
        nonwestern = []

        X_reference = new_df

        ids = []

        for row in tqdm(X_reference.itertuples(), total=len(X_reference)):
            image_paths = row.image_path.split(",")
            recipe = row.recipe
            ingredients = row.ingredients
            embedding = get_embedding(image_paths, recipe, ingredients, image_importance=IMAGE_IMPORTANCE)
            if id_to_label[row.id] == 1:
                western.append(embedding)
            else:
                nonwestern.append(embedding)

        western = np.array(western)
        nonwestern = np.array(nonwestern)

        westerns.append(western)
        nonwesterns.append(nonwestern)

        # save numpy array
        np.save('./arrays/' + EXPERIMENT_NAME + '_' + str(IMAGE_IMPORTANCE) + '_western.npy', western)
        np.save('./arrays/' + EXPERIMENT_NAME + '_' + str(IMAGE_IMPORTANCE) + '_nonwestern.npy', nonwestern)
else:
    for IMAGE_IMPORTANCE in [0.0, 0.5, 1.0]:
        westerns.append(np.load('./arrays/' + EXPERIMENT_NAME + '_' + str(IMAGE_IMPORTANCE) + '_western.npy'))
        nonwesterns.append(np.load('./arrays/' + EXPERIMENT_NAME + '_' + str(IMAGE_IMPORTANCE) + '_nonwestern.npy'))

with torch.no_grad():
    healthy = clip.tokenize(["healthy"]).to(device)
    healthy_embedding = model.encode_text(healthy).cpu().numpy()

for i, IMAGE_IMPORTANCE in enumerate([0.0, 0.5, 1.0]):
    print("\n\nIMAGE_IMPORTANCE: ", IMAGE_IMPORTANCE)
    western = westerns[i]
    nonwestern = nonwesterns[i]

    print("Western shape: {}".format(western.shape))
    print("Non-western shape: {}".format(nonwestern.shape))
    print("Fraction of western: {}".format(len(western)/(len(western) + len(nonwestern))))

    # Calculate average distance to healthy
    western_dist = np.linalg.norm(western - healthy_embedding, axis=1)
    nonwestern_dist = np.linalg.norm(nonwestern - healthy_embedding, axis=1)

    print("Western mean distance: {}".format(np.mean(western_dist)))
    print("Non-western mean distance: {}".format(np.mean(nonwestern_dist)))


    pca = PCA(n_components=2)
    western_cuttoff = len(western)

    all = np.concatenate((western, nonwestern), axis=0)

    pca_data = pca.fit_transform(all)

    pca_western = pca_data[:western_cuttoff]
    pca_nonwestern = pca_data[western_cuttoff:]

    plt.scatter(pca_western[:,0], pca_western[:,1], label="Western", alpha=OPACITY)
    plt.scatter(pca_nonwestern[:,0], pca_nonwestern[:,1], label="Non-western", alpha=OPACITY)

    pca_healthy = pca.transform(healthy_embedding)
    plt.scatter(pca_healthy[:,0], pca_healthy[:,1], label="Healthy", c="black")

    plt.legend()

    plt.title("Principal Component Analysis")
    plt.savefig("PCA_{}.png".format(IMAGE_IMPORTANCE), bbox_inches='tight')
    plt.clf()

    # Calculate t-SNE
    from sklearn.manifold import TSNE

    tsne = TSNE(n_iter=3000, random_state=23)

    actual_all = np.concatenate((western, nonwestern, healthy_embedding), axis=0)
    tsne_data = tsne.fit_transform(actual_all)

    tsne_western = tsne_data[:western_cuttoff]
    tsne_nonwestern = tsne_data[western_cuttoff:-1]

    tsne_western = tsne_western[::10]
    tsne_nonwestern = tsne_nonwestern[::10]

    plt.scatter(tsne_western[:,0], tsne_western[:,1], label="Western", alpha=OPACITY)
    plt.scatter(tsne_nonwestern[:,0], tsne_nonwestern[:,1], label="Non-western", alpha=OPACITY)

    tsne_healthy = tsne_data[-1:]
    plt.scatter(tsne_healthy[:,0], tsne_healthy[:,1], label="Healthy", c="black")

    plt.legend()
    map = {0.0: "Image", 0.5: "Image + Text", 1.0: "Text"}
    plt.title(map[IMAGE_IMPORTANCE] + " Embedding t-SNE")
    plt.savefig("t-SNE_{}.png".format(IMAGE_IMPORTANCE), bbox_inches='tight')
    plt.clf()

    # Calculate closest 100 neighbors to healthy
    index = faiss.IndexFlatIP(512)
    all_normalized = all / np.linalg.norm(all, axis=1)[:, None]
    index.add(all_normalized)
    healthy_normalized = healthy_embedding / np.linalg.norm(healthy_embedding)

    D, I = index.search(healthy_normalized, 100)

    print("Fraction of closest neighbors that are western with cosine sim: {}".format(np.sum(I < western_cuttoff)/len(I)))


    index2 = faiss.IndexFlatL2(512)
    index2.add(all)
    D, I = index2.search(healthy_embedding, 100)

    print("Fraction of closest neighbors that are western with L2: {}".format(np.sum(I < western_cuttoff)/len(I)))

# tsne_healthy = tsne_data[-1:]
# plt.scatter(tsne_healthy[:,0], tsne_healthy[:,1], label="Healthy", c="black")

# plt.legend()

# plt.title("Text Embedding t-SNE")
# plt.savefig("t-SNE.png", bbox_inches='tight')
