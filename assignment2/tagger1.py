import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

NER_TRAIN = r"./ner/train.txt"
POS_TRAIN = r"./pos/train.txt"
VOCAB = r"./embeddings/vocab.txt"
WORD_VECTORS = r"./embeddings/wordVectors.txt"

embeddings = {}

def load_embeddings(VOCAB: str, WORD_VECTORS: str):
    embeddings = {}

    with open(VOCAB, "r", encoding="utf-8") as vocab, open(WORD_VECTORS, "r", encoding="utf-8") as wordVectors:
        for word, vector in zip(vocab.readlines(), wordVectors.readlines()):
            embeddings[word.replace("\n", "")] = np.fromstring(vector.replace(" \n", ""), sep=" ")
    return embeddings


# embeddings = load_embeddings(VOCAB, WORD_VECTORS)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.w00 = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.b00 = nn.Parameter(torch.tensor(-1.), requires_grad=True)

    def forward(self, input):
        input_to_tanh = self.w00 * input + self.b00
        tanh_output = F.tanh(input_to_tanh)

        output = F.softmax(tanh_output, dim=0)

        return output

model = MLP()
input = torch.tensor([0., 0.5, 1.])
labels= torch.tensor([0., 1., 0.])

optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    total_loss = 0

    for iteration in range(len(input)):
        input_i = input[iteration]
        labels_i = labels[iteration]

        output_i = model(input_i)

        # loss = F.cross_entropy(output_i, labels_i)
        loss = (output_i - labels_i).pow(2)

        loss.backward()

        total_loss += float(loss)

    optimizer.step()
    optimizer.zero_grad()
    print(f"epoch {epoch}, loss: {total_loss}, weight: {model.w00.data}, bias: {model.b00.data}")