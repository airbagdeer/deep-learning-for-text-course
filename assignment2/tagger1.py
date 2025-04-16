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

        self.w00 = nn.Parameter(torch.randn((5,50), dtype=torch.float64), requires_grad=True)
        self.b00 = nn.Parameter(torch.randn((5,1), dtype=torch.float64), requires_grad=True)

    def forward(self, input):
        input_to_tanh = torch.matmul(self.w00, input) + self.b00
        tanh_output = F.tanh(input_to_tanh)

        output = tanh_output
        # TODO: remove:
        # output = F.softmax(tanh_output, dim=0)

        return output

# TODO: understand how to backpropogation

def arrange_ner_training_data(ner_labels: Dict[str, str], embeddings: Dict[str, torch.Tensor]):
    train_embeddings_in_order = []
    labels_in_order = []

    for word, label in ner_labels.items():
        if word in embeddings:
            train_embeddings_in_order.append(embeddings[word])
            labels_in_order.append(label)

    return train_embeddings_in_order, labels_in_order


embeddings = load_embeddings(VOCAB, WORD_VECTORS)
ner_raw_train_data = load_NER_train_data(NER_TRAIN)

ner_train_data, labels = arrange_ner_training_data(ner_raw_train_data, embeddings)

model = NER()

optimizer = optim.Adam(model.parameters(), lr=0.1)
loss = nn.CrossEntropyLoss()

# model(torch.randn(50, 50, dtype=torch.float64))
# model(embeddings["hello"])

print(torch.cat((embeddings["hello"], embeddings["world"]), dim=0).shape)

# for epoch in range(100):
#     total_loss = 0
#
#     output = model(ner_train_data)
#
#
#     # for iteration in range(len(ner_train_data)):
#     #     input_i = ner_train_data[iteration]
#     #     labels_i = labels[iteration]
#     #
#     #     output_i = model(input_i)
#     #     print(output_i)
#     #
#     #     loss = F.cross_entropy(output_i, labels_i)
#     #     # loss = (output_i - labels_i).pow(2)
#     #
#     #     loss.backward()
#     #
#     #     total_loss += float(loss)
#
#     optimizer.step()
#     optimizer.zero_grad()
#     print(f"epoch {epoch}, loss: {total_loss}, weight: {model.w00.data}, bias: {model.b00.data}")