import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict

NER_TRAIN = r"./ner/train"
POS_TRAIN = r"./pos/train"
VOCAB = r"./embeddings/vocab.txt"
WORD_VECTORS = r"./embeddings/wordVectors.txt"

NER_labels = {
    "PER": 0,
    "LOC": 1,
    "ORG": 2,
    "MISC": 3,
    "O": 4,
}

def load_embeddings(VOCAB_FILE_PATH: str, WORD_VECTORS_FILE_PATH: str) -> Dict[str, torch.Tensor]:
    embeddings = {}

    with open(VOCAB_FILE_PATH, "r", encoding="utf-8") as vocab, open(WORD_VECTORS_FILE_PATH, "r", encoding="utf-8") as wordVectors:
        for word, vector in zip(vocab.readlines(), wordVectors.readlines()):
            embeddings[word.replace("\n", "")] = torch.from_numpy(np.fromstring(vector.replace(" \n", ""), sep=" "))
    return embeddings


def load_NER_train_data(TRAIN_PATH: str)-> Dict[str, str]:
    train_data = {}
    with open(TRAIN_PATH, "r", encoding="utf-8") as train_file:
        for raw_line in train_file.readlines():
            line = raw_line.replace("\n", "").split("\t")
            if len(line) == 2:
                [word, type] = raw_line.replace("\n", "").split("\t")
                train_data[word] = NER_labels[type]

    return train_data

class NER(nn.Module):
    def __init__(self):
        super().__init__()

        self.w00 = nn.Parameter(torch.randn((50,5), dtype=torch.float64), requires_grad=True)
        self.b00 = nn.Parameter(torch.randn((5), dtype=torch.float64), requires_grad=True)

    def forward(self, input):
        input_to_tanh = torch.matmul(input, self.w00) + self.b00
        tanh_output = F.tanh(input_to_tanh)

        output = tanh_output
        # TODO: remove:
        # output = F.softmax(tanh_output, dim=1)

        return output

# TODO: understand how to backpropogation

def arrange_ner_training_data(ner_labels: Dict[str, str], embeddings: Dict[str, torch.Tensor]):
    train_embeddings_in_order = []
    labels_in_order = []

    for word, label in ner_labels.items():
        if word in embeddings:
            train_embeddings_in_order.append(embeddings[word])
            labels_in_order.append(label)

    return torch.stack(train_embeddings_in_order, dim=0), torch.tensor(labels_in_order)


embeddings = load_embeddings(VOCAB, WORD_VECTORS)
ner_raw_train_data = load_NER_train_data(NER_TRAIN)

ner_train_data, labels = arrange_ner_training_data(ner_raw_train_data, embeddings)

model = NER()

optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    output = model(ner_train_data)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")