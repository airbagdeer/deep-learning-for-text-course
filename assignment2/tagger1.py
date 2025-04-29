import string
from typing import Dict

import torch

from ner import train_NER
from pos import train_POS, process_pos_data
from utils import DTYPE, POS_TRAIN, POS_DEV, NER_TRAIN, NER_DEV

NER_MODEL = r"./ner-processed/q1-model.pt"

POS_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS = r"./pos_processed/train_context_embeddings.pt"
POS_PROCESSED_DEV_CONTEXT_EMBEDDINGS = r"./pos_processed/dev_context_embeddings.pt"
POS_PROCESSED_TEST_CONTEXT_EMBEDDINGS = r"./pos_processed/test_context_embeddings.pt"


POS_PROCESSED_TRAIN_CONTEXT_RANDOM_EMBEDDINGS = r"./pos_processed/train_context_random_embeddings.pt"
POS_PROCESSED_TRAIN_LABELS = r"./pos_processed/train_labels.pt"
POS_PROCESSED_DEV_CONTEXT_RANDOM_EMBEDDINGS = r"./pos_processed/dev_context_random_embeddings.pt"
POS_PROCESSED_DEV_LABELS = r"./pos_processed/dev_labels.pt"
POS_PROCESSED_TEST_LABELS = r"./pos_processed/test_labels.pt"
Q1_POS_MODEL = r"./pos_processed/q1-model.pt"

def load_pos_random_embeddings(TRAIN_FILE, embeddings_size = 50) -> Dict[str, torch.Tensor]:
    embeddings = {}

    def is_punctuation(s):
        return all(char in string.punctuation for char in s) and bool(s)

    with open(TRAIN_FILE, "r", encoding="utf-8") as train:
        for line in train.readlines():
            words = line.split(" ")
            actual_word = words[0].lower()
            if not is_punctuation(actual_word) and actual_word != "\n" and actual_word != "\t" and actual_word != "\r" and actual_word != " ":
                if actual_word not in embeddings:
                    embeddings[actual_word] = torch.randn(embeddings_size, dtype=DTYPE)

    embeddings["<pad>"] = torch.zeros(len(list(embeddings.values())[0]), dtype=DTYPE)
    embeddings["<unk>"] = torch.stack(list(embeddings.values()), dim=1).mean(dim=1)

    return embeddings

def load_ner_random_embeddings(TRAIN_FILE, embeddings_size = 50) -> Dict[str, torch.Tensor]:
    embeddings = {}

    with open(TRAIN_FILE, "r", encoding="utf-8") as train:
        for line in train.readlines():
            words = line.split("\t")
            actual_word = words[0].lower()
            if actual_word != "\n" and actual_word != "\t" and actual_word != "\r" and actual_word != " ":
                if actual_word not in embeddings:
                    embeddings[actual_word] = torch.randn(embeddings_size, dtype=DTYPE)

    embeddings["<pad>"] = torch.zeros(len(list(embeddings.values())[0]), dtype=DTYPE)
    embeddings["<unk>"] = torch.stack(list(embeddings.values()), dim=1).mean(dim=1)

    return embeddings


torch.manual_seed(42)

# pos_random_embeddings = load_pos_random_embeddings(POS_TRAIN)
# process_pos_data(pos_random_embeddings, POS_TRAIN, POS_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS, POS_PROCESSED_TRAIN_LABELS, POS_DEV, POS_PROCESSED_DEV_CONTEXT_EMBEDDINGS, POS_PROCESSED_DEV_LABELS, save_labels=True)

# train_POS(POS_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS, POS_PROCESSED_TRAIN_LABELS, POS_PROCESSED_DEV_CONTEXT_EMBEDDINGS, POS_PROCESSED_DEV_LABELS)


# TODO: add batch size 128/256, look at torch DataLoader.
# TODO: each batch should have examples from not only the o class if possible
ner_random_embeddings = load_ner_random_embeddings(NER_TRAIN)
train_NER(ner_random_embeddings, NER_TRAIN, NER_DEV)