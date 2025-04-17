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

    embeddings["<pad>"] = torch.zeros(50, dtype=torch.float64)

    return embeddings


def arrange_ner_training_data(ner_raw_train_data: [[str, str]], embeddings: Dict[str, torch.Tensor]) -> [(str, torch.Tensor), [torch.Tensor]]:
    ner_training_data = []
    # labels = torch.empty(0, dtype=torch.int64)
    # ner_training_data_words_embeddings = torch.empty(0, dtype=torch.float64)


    for [label, words] in ner_raw_train_data:
        embedded_words = torch.empty(0, dtype=torch.float64)
        for word in words:
            if word.lower() in embeddings:
                embedded_words = torch.cat((embedded_words, embeddings[word.lower()]), dim=0)
                # embedded_words.append(embeddings[word.lower()])
            else:
                embedded_words = torch.cat((embedded_words, embeddings["<pad>"]), dim=0)
                # embedded_words.append(embeddings["<pad>"])

        # ner_training_data_words_embeddings = torch.cat((ner_training_data_words_embeddings, embedded_words), dim=0)
        # labels = torch.cat((labels, torch.tensor([NER_labels[label]])), dim=0)
        ner_training_data.append((words[2], torch.tensor(NER_labels[label]), embedded_words))

    # return ner_training_data, ner_training_data_words_embeddings, labels
    return ner_training_data


class NER(nn.Module):
    def __init__(self):
        super().__init__()

        self.w00 = nn.Parameter(torch.randn((250,5), dtype=torch.float64), requires_grad=True)
        self.b00 = nn.Parameter(torch.randn((5), dtype=torch.float64), requires_grad=True)

    def forward(self, input):
        input_to_tanh = torch.matmul(input, self.w00) + self.b00
        tanh_output = F.tanh(input_to_tanh)

        output = tanh_output
        # TODO: remove:
        # output = F.softmax(tanh_output, dim=1)

        return output


def parse_ner_file(filepath):
    result = []
    filler = "<pad>"

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    document = []  # current document (list of [word, label])

    def process_document(doc):
        for i, (word, label) in enumerate(doc):
            context = []
            for j in range(i - 2, i + 3):
                if 0 <= j < len(doc):
                    context.append(doc[j][0])
                else:
                    context.append(filler)
            result.append([label, context])

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("-DOCSTART-"):
            if document:
                process_document(document)
                document = []
        else:
            parts = line.split()
            if len(parts) == 2:
                word, label = parts
                document.append((word, label))

    # Process the last document
    if document:
        process_document(document)

    return result


def train(model: nn.Module, train_data, labels):

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


embeddings = load_embeddings(VOCAB, WORD_VECTORS)
ner_raw_train_data = parse_ner_file(NER_TRAIN)

ner_train_data_organized = arrange_ner_training_data(ner_raw_train_data, embeddings)

context_embeddings = torch.stack([context_embedding[2] for context_embedding in ner_train_data_organized], dim=0)
labels = torch.stack([context_embedding[1] for context_embedding in ner_train_data_organized], dim=0)

model = NER()

train(model, context_embeddings, labels)
