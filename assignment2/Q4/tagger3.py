import string
from typing import Dict

import numpy as np
import torch

from ner import train_NER, evaluate_ner_file_with_context
from pos import train_POS, process_pos_data, evaluate_pos_file_with_context
from utils import DTYPE, POS_TRAIN, POS_DEV, NER_TRAIN, NER_DEV, NER_TEST, POS_TEST

with open("../embeddings/vocab.txt", "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]

vocab.insert(0, "<pad>")
vocab.insert(1, "<unk>")

word_to_index = {word: idx for idx, word in enumerate(vocab)}

with open("../embeddings/wordVectors.txt", "r", encoding="utf-8") as f:
    vectors = [list(map(float, line.strip().split())) for line in f]

unk_vector = np.mean(np.array(vectors, dtype=np.float32), axis=0).tolist()
vectors.insert(0, [0.0] * len(vectors[0]))
vectors.insert(1, unk_vector)

embedding_matrix = torch.tensor(vectors, dtype=torch.float32)

def load_ner_one_hot_words_prefix_and_suffix(TRAIN_FILE):
    prefix_embeddings = {}
    prefix_embeddings["<pad>"] = 0
    prefix_embeddings["<unk>"] = 1
    prefix_embeddings["<short>"] = 2

    suffix_embeddings = {}
    suffix_embeddings["<pad>"] = 0
    suffix_embeddings["<unk>"] = 1
    suffix_embeddings["<short>"] = 2

    prefix_amount_of_words = 3
    suffix_amount_of_words = 3

    with open(TRAIN_FILE, "r", encoding="utf-8") as train:
        for line in train.readlines():
            words = line.split("\t")
            actual_word = words[0].lower()
            if actual_word != "\n" and actual_word != "\t" and actual_word != "\r" and actual_word != " ":
                if len(actual_word)>=3:
                    prefix = actual_word[:3]
                    suffix = actual_word[-3:]

                    if prefix not in prefix_embeddings:
                        prefix_embeddings[prefix] = prefix_amount_of_words
                        prefix_amount_of_words += 1

                    if suffix not in suffix_embeddings:
                        suffix_embeddings[suffix] = suffix_amount_of_words
                        suffix_amount_of_words += 1


    return prefix_embeddings, prefix_amount_of_words, suffix_embeddings, suffix_amount_of_words


def load_pos_one_hot_words_prefix_and_suffix(TRAIN_FILE):
    prefix_embeddings = {}
    prefix_embeddings["<pad>"] = 0
    prefix_embeddings["<unk>"] = 1
    prefix_embeddings["<short>"] = 2

    suffix_embeddings = {}
    suffix_embeddings["<pad>"] = 0
    suffix_embeddings["<unk>"] = 1
    suffix_embeddings["<short>"] = 2

    prefix_amount_of_words = 3
    suffix_amount_of_words = 3

    with open(TRAIN_FILE, "r", encoding="utf-8") as train:
        for line in train.readlines():
            words = line.split(" ")
            actual_word = words[0].lower()
            if actual_word != "\n" and actual_word != "\t" and actual_word != "\r" and actual_word != " ":
                if len(actual_word)>=3:
                    prefix = actual_word[:3]
                    suffix = actual_word[-3:]

                    if prefix not in prefix_embeddings:
                        prefix_embeddings[prefix] = prefix_amount_of_words
                        prefix_amount_of_words += 1

                    if suffix not in suffix_embeddings:
                        suffix_embeddings[suffix] = suffix_amount_of_words
                        suffix_amount_of_words += 1


    return prefix_embeddings, prefix_amount_of_words, suffix_embeddings, suffix_amount_of_words


torch.manual_seed(42)

vocab_size = len(vocab)

# TODO: fix evaluate function, evaluate on non pretrained as well
# TODO: understand what the best model is and evaluate on it
ner_prefix_embeddings, ner_prefix_amount_of_words, ner_suffix_embeddings, ner_suffix_amount_of_words = load_ner_one_hot_words_prefix_and_suffix(NER_TRAIN)
ner_trained_model = train_NER(word_to_index, NER_TRAIN, NER_DEV, vocab_size, embedding_matrix, ner_prefix_embeddings, ner_prefix_amount_of_words, ner_suffix_embeddings, ner_suffix_amount_of_words)
# evaluate_ner_file_with_context(ner_trained_model, NER_TEST, word_to_index, "test4.ner")

pos_prefix_embeddings, pos_prefix_amount_of_words, pos_suffix_embeddings, pos_suffix_amount_of_words = load_pos_one_hot_words_prefix_and_suffix(NER_TRAIN)
pos_trained_model = train_POS(POS_TRAIN, POS_DEV, word_to_index, vocab_size, embedding_matrix, pos_prefix_embeddings, pos_prefix_amount_of_words, pos_suffix_embeddings, pos_suffix_amount_of_words)
# evaluate_pos_file_with_context(pos_trained_model, POS_TEST, word_to_index, "test4.pos")
