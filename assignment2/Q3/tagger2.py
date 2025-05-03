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

torch.manual_seed(42)

vocab_size = len(vocab)
ner_trained_model = train_NER(word_to_index, NER_TRAIN, NER_DEV, vocab_size, embedding_matrix)
evaluate_ner_file_with_context(ner_trained_model, NER_TEST, word_to_index, "test3.ner")

pos_trained_model = train_POS(POS_TRAIN, POS_DEV, word_to_index, vocab_size, embedding_matrix)
evaluate_pos_file_with_context(pos_trained_model, POS_TEST, word_to_index, "test3.pos")
