from typing import Dict

import numpy as np
import torch

from pos import process_pos_data
from utils import torch_to_numpy_dtype, DTYPE, POS_TRAIN

VOCAB = r"./embeddings/vocab.txt"
WORD_VECTORS = r"./embeddings/wordVectors.txt"

POS_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS = r"./pos_processed/train_context_embeddings.pt"
POS_PROCESSED_DEV_CONTEXT_EMBEDDINGS = r"./pos_processed/dev_context_embeddings.pt"
POS_PROCESSED_TEST_CONTEXT_EMBEDDINGS = r"./pos_processed/test_context_embeddings.pt"

def load_pretrained_embeddings(VOCAB_FILE_PATH: str, WORD_VECTORS_FILE_PATH: str) -> Dict[str, torch.Tensor]:
    embeddings = {}

    with open(VOCAB_FILE_PATH, "r", encoding="utf-8") as vocab, open(WORD_VECTORS_FILE_PATH, "r",
                                                                     encoding="utf-8") as wordVectors:
        for word, vector in zip(vocab.readlines(), wordVectors.readlines()):
            embeddings[word.replace("\n", "")] = torch.from_numpy(
                np.fromstring(vector.replace(" \n", ""), sep=" ", dtype=torch_to_numpy_dtype[DTYPE]))

    embeddings["<pad>"] = torch.zeros(len(list(embeddings.values())[0]), dtype=DTYPE)
    embeddings["<unk>"] = torch.stack(list(embeddings.values()), dim=1).mean(dim=1)

    return embeddings

pretrained_embeddings = load_pretrained_embeddings(VOCAB, WORD_VECTORS)

process_pos_data(pretrained_embeddings, POS_TRAIN, POS_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS, POS_PROCESSED_TRAIN_LABELS, POS_DEV, POS_PROCESSED_DEV_CONTEXT_EMBEDDINGS, POS_PROCESSED_DEV_LABELS)

