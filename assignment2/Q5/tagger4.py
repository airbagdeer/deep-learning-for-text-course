import string
from typing import Dict

import numpy as np
import torch

from ner import train_NER, evaluate_ner_file_with_context
from pos import train_POS, process_pos_data, evaluate_pos_file_with_context
from utils import DTYPE, POS_TRAIN, POS_DEV, NER_TRAIN, NER_DEV, NER_TEST, POS_TEST

def get_pretrained_embeddings():
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

    vocab_size = len(vocab)

    return torch.tensor(vectors, dtype=torch.float32), word_to_index, vocab_size

embedding_matrix, word_to_index, vocab_size = get_pretrained_embeddings()


def create_char_dictionary(file_path):
    char_to_idx = {
        '<pad>': 0,
        '<unk>': 1
    }
    idx = 2

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line or '\t' not in line:
                    continue

                word = line.split('\t')[0]

                for char in word:
                    if char not in char_to_idx:
                        char_to_idx[char] = idx
                        idx += 1

    except Exception as e:
        print(f"Error reading file: {e}")
        return {}, 0

    unique_chars = len(char_to_idx)

    return char_to_idx, unique_chars


char_to_index, num_chars = create_char_dictionary(NER_TRAIN)

torch.manual_seed(42)
max_word_size = 70

# TODO: NER is working, do the evaluate function.
ner_trained_model = train_NER(word_to_index, NER_TRAIN, NER_DEV, vocab_size, embedding_matrix, char_to_index, num_chars, max_word_size)
# evaluate_ner_file_with_context(ner_trained_model, NER_TEST, word_to_index, "test3.ner")

# pos_trained_model = train_POS(POS_TRAIN, POS_DEV, word_to_index, vocab_size, embedding_matrix)
# evaluate_pos_file_with_context(pos_trained_model, POS_TEST, word_to_index, "test3.pos")
