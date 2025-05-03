import string
from typing import Dict

import torch

from ner import train_NER, evaluate_ner_file_with_context
from pos import train_POS, process_pos_data, evaluate_pos_file_with_context
from utils import DTYPE, POS_TRAIN, POS_DEV, NER_TRAIN, NER_DEV, NER_TEST, POS_TEST

def load_ner_one_hot_words(TRAIN_FILE):
    embeddings = {}
    embeddings["<pad>"] = 0
    embeddings["<unk>"] = 1
    amount_of_words = 2

    with open(TRAIN_FILE, "r", encoding="utf-8") as train:
        for line in train.readlines():
            words = line.split("\t")
            actual_word = words[0].lower()
            if actual_word != "\n" and actual_word != "\t" and actual_word != "\r" and actual_word != " ":
                if actual_word not in embeddings:
                    embeddings[actual_word] = amount_of_words
                    amount_of_words += 1


    return embeddings, amount_of_words


def load_pos_one_hot_embeddings(TRAIN_FILE):
    embeddings = {}
    embeddings["<pad>"] = 0
    embeddings["<unk>"] = 1
    amount_of_words = 2

    def is_punctuation(s):
        return all(char in string.punctuation for char in s) and bool(s)

    with open(TRAIN_FILE, "r", encoding="utf-8") as train:
        for line in train.readlines():
            words = line.split(" ")
            actual_word = words[0].lower()
            if not is_punctuation(actual_word) and actual_word != "\n" and actual_word != "\t" and actual_word != "\r" and actual_word != " ":
                if actual_word not in embeddings:
                    embeddings[actual_word] = amount_of_words
                    amount_of_words += 1


    return embeddings, amount_of_words

torch.manual_seed(42)

ner_one_hot_embeddings, vocab_size = load_ner_one_hot_words(NER_TRAIN)
ner_trained_model = train_NER(ner_one_hot_embeddings, NER_TRAIN, NER_DEV, vocab_size)
evaluate_ner_file_with_context(ner_trained_model, NER_TEST, ner_one_hot_embeddings, "test1.ner")

# TODO: Move everything to a Q1 folder, Arrange Q2 folder, and create Q3 folder, tagger3 already exists.
# TODO: Use AI Chat (you have a free trial dont waste it)
# pos_one_hot_embeddings, vocab_size = load_pos_one_hot_embeddings(POS_TRAIN)
# pos_trained_model = train_POS(POS_TRAIN, POS_DEV, pos_one_hot_embeddings, vocab_size)
# evaluate_pos_file_with_context(pos_trained_model, POS_TEST, pos_one_hot_embeddings, "test1.pos")
