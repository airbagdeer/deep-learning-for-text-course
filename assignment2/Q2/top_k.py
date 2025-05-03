import numpy as np
import torch
from typing import Dict

VOCAB = r"../embeddings/vocab.txt"
WORD_VECTORS = r"../embeddings/wordVectors.txt"

def load_pretrained_embeddings(VOCAB_FILE_PATH: str, WORD_VECTORS_FILE_PATH: str) -> Dict[str, torch.Tensor]:
    embeddings = {}

    with open(VOCAB_FILE_PATH, "r", encoding="utf-8") as vocab, open(WORD_VECTORS_FILE_PATH, "r",
                                                                     encoding="utf-8") as wordVectors:
        for word, vector in zip(vocab.readlines(), wordVectors.readlines()):
            embeddings[word.replace("\n", "")] = torch.from_numpy(
                np.fromstring(vector.replace(" \n", ""), sep=" ", dtype=np.float32))

    embeddings["<pad>"] = torch.zeros(len(list(embeddings.values())[0]), dtype=torch.float32)
    embeddings["<unk>"] = torch.stack(list(embeddings.values()), dim=1).mean(dim=1)

    return embeddings

embeddings = load_pretrained_embeddings(VOCAB, WORD_VECTORS)

def most_similar(word, k):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    word_embedding = embeddings[word]
    similarity = []
    for embedding in embeddings.values():
        similarity.append(cos(embedding, word_embedding))

    values, indices = torch.topk(torch.stack(similarity), k+1)
    return [{list(embeddings.keys())[indices[i+1]]: values[i+1].item()} for i in range(len(indices)-1)]

print('dog', most_similar('dog', 5))
print('england',most_similar('england', 5))
print('john',most_similar('john', 5))
print('explode',most_similar('explode', 5))
print('office',most_similar('office', 5))