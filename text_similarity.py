import torch
import csv
import numpy as np
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

model, preprocess = clip.load('ViT-B/32', device)

model.to(device)

def get_embeddings(texts):
    text_input = clip.tokenize(texts).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(text_input)
    return embeddings.cpu().numpy()

with open("movies_dataset.csv", newline='',encoding="utf-8") as csvfile:
    dataset = list(csv.reader(csvfile, delimiter=','))

dataset = np.array(dataset)

descriptions = dataset[1:, 9].tolist()

user_input = input()

try:
    descriptions_embeddings = np.load("embeddings.npy")
except OSError:
    descriptions_embeddings = get_embeddings(descriptions)
    np.save("embeddings.npy", descriptions_embeddings)
for i in range(len(descriptions_embeddings)):
    descriptions_embeddings[i] /= np.linalg.norm(descriptions_embeddings[i])

user_embedding = get_embeddings([user_input])[0]
user_embedding /= np.linalg.norm(user_embedding)

similarity_matrix = user_embedding @ descriptions_embeddings.T

top_indices = similarity_matrix.argsort()[-5:][::-1]
for i in top_indices:
    print(dataset[i+1])



