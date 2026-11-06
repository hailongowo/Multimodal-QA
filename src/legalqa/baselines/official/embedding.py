"""Official corpus encoding over the baseline's preprocessed law data."""

import re

import numpy as np
from tqdm import tqdm


def embedding_articles(law_db, img_path, model):
    model.eval()
    lst_id = []
    embedding = None

    for law in law_db:
        law_id = law["id"]
        print(law_id)
        print(law["title"])
        print("---------=====---------")
        for d in tqdm(law["articles"]):
            images_pattern = re.finditer(r"<<IMAGE: \s*(.*?)\s* \/IMAGE>>", d["text"])
            if re.search(r"<<IMAGE: \s*(.*?)\s* /IMAGE>>", d["text"]) is not None:
                print("text+image")
                lst_imgs = []
                for img in images_pattern:
                    image_file = img.group(1).strip()
                    lst_imgs.append(str(img_path) + "/" + image_file)
                    article = d["text"].replace(img.group(0), "")
                    d["text"] = article

                for i in range(0, len(lst_imgs)):
                    lst_id.append("{}-{}_{}".format(d["id"], str(i + 1), law_id))
                    candidates = model.encode(image=lst_imgs[i], text=d["text"])
                    if embedding is None:
                        embedding = candidates.cpu().detach().numpy()
                    else:
                        embedding = np.append(embedding, candidates.cpu().detach().numpy(), axis=0)
            else:
                print("text")
                candidates = model.encode(text=d["text"])
                lst_id.append(d["id"] + "_" + law_id)
                if embedding is None:
                    embedding = candidates.cpu().detach().numpy()
                else:
                    embedding = np.append(embedding, candidates.cpu().detach().numpy(), axis=0)

    embedding_id = np.array(lst_id)
    print(embedding.shape)
    print(embedding_id.shape)
    return embedding_id, embedding
