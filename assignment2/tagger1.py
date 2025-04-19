import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import Accuracy
import numpy as np
from typing import Dict, Tuple

NER_TRAIN = r"./ner/train"
NER_DEV = r"./ner/dev"
NER_TEST = r"./ner/test"
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

POS_labels = {
    "CC": 0,
    "CD": 1,
    "DT": 2,
    "EX": 3,
    "FW": 4,
    "IN": 5,
    "JJ": 6,
    "JJR": 7,
    "JJS": 8,
    "LS": 9,
    "MD": 10,
    "NN": 11,
    "NNS": 12,
    "NNP": 13,
    "NNPS": 14,
    "PDT": 15,
    "POS": 16,
    "PRP": 17,
    "PRP$": 18,
    "RB": 19,
    "RBR": 20,
    "RBS": 21,
    "RP": 22,
    "SYM": 23,
    "TO": 24,
    "UH": 25,
    "VB": 26,
    "VBD": 27,
    "VBG": 28,
    "VBN": 29,
    "VBP": 30,
    "VBZ": 31,
    "WDT": 32,
    "WP": 33,
    "WP$": 34,
    "WRB": 35
}


def load_embeddings(VOCAB_FILE_PATH: str, WORD_VECTORS_FILE_PATH: str) -> Dict[str, torch.Tensor]:
    embeddings = {}

    with open(VOCAB_FILE_PATH, "r", encoding="utf-8") as vocab, open(WORD_VECTORS_FILE_PATH, "r",
                                                                     encoding="utf-8") as wordVectors:
        for word, vector in zip(vocab.readlines(), wordVectors.readlines()):
            embeddings[word.replace("\n", "")] = torch.from_numpy(np.fromstring(vector.replace(" \n", ""), sep=" "))

    embeddings["<pad>"] = torch.stack(list(embeddings.values()), dim=1).mean(dim=1)

    return embeddings


def arrange_data(raw_data: [[str, str]], embeddings: Dict[str, torch.Tensor], true_labels: Dict[str, int]):
    words_in_order = []
    labels_in_order = []
    context_embeddings = []

    for [label, words] in raw_data:
        embedded_words = torch.empty(0, dtype=torch.float64)
        for word in words:
            if word.lower() in embeddings:
                embedded_words = torch.cat((embedded_words, embeddings[word.lower()]), dim=0)
            else:
                embedded_words = torch.cat((embedded_words, embeddings["<pad>"]), dim=0)

        words_in_order.append(words[2])
        labels_in_order.append(torch.tensor(true_labels[label]))
        context_embeddings.append(embedded_words)

    return words_in_order, torch.stack(labels_in_order, dim=0), torch.stack(context_embeddings, dim=0)


def parse_file(filepath):
    result = []
    filler = "<pad>"

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    document = []

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

    if document:
        process_document(document)

    return result


class NER(nn.Module):
    def __init__(self):
        super().__init__()

        self.w00 = nn.Parameter(torch.randn((250, 5), dtype=torch.float64), requires_grad=True)
        self.b00 = nn.Parameter(torch.randn((5), dtype=torch.float64), requires_grad=True)

    def forward(self, input):
        input_to_tanh = torch.matmul(input, self.w00) + self.b00

        output = F.tanh(input_to_tanh)

        return output


class NER(nn.Module):
    def __init__(self):
        super().__init__()

        self.w00 = nn.Parameter(torch.randn((250, 5), dtype=torch.float64), requires_grad=True)
        self.b00 = nn.Parameter(torch.randn((5), dtype=torch.float64), requires_grad=True)

    def forward(self, input):
        input_to_tanh = torch.matmul(input, self.w00) + self.b00

        output = F.tanh(input_to_tanh)

        return output

class POS(nn.Module):
    def __init__(self):
        super().__init__()

        self.w00 = nn.Parameter(torch.randn((250, 36), dtype=torch.float64), requires_grad=True)
        self.b00 = nn.Parameter(torch.randn((36), dtype=torch.float64), requires_grad=True)

    def forward(self, input):
        input_to_tanh = torch.matmul(input, self.w00) + self.b00

        output = F.tanh(input_to_tanh)

        return output

def train(model: nn.Module, train_data, labels, epochs: int = 100, lr: int = 0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


torch.manual_seed(42)
embeddings = load_embeddings(VOCAB, WORD_VECTORS)

def train_NER():
    ner_raw_train_data = parse_file(NER_TRAIN)
    ner_raw_dev_data = parse_file(NER_DEV)

    train_words, train_labels, train_context_embeddings = arrange_data(ner_raw_train_data, embeddings, NER_labels)
    dev_words, dev_labels, dev_context_embeddings = arrange_data(ner_raw_dev_data, embeddings, NER_labels)

    model = NER()

    train(model, train_context_embeddings, train_labels)

    raw_predictions = model(dev_context_embeddings)
    probabilities = F.softmax(raw_predictions, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    print(predictions.shape, dev_labels.shape)

    accuracy = Accuracy(task='multiclass', num_classes=5)
    accuracy.update(predictions, dev_labels)
    print(accuracy.compute())


    def remove_correct_class_o(true_labels: torch.Tensor, pred_labels: torch.Tensor, o_class_number) -> Tuple[
        torch.Tensor, torch.Tensor]:
        correct_class_o_mask = (true_labels == o_class_number) & (pred_labels == o_class_number)
        keep_mask = ~correct_class_o_mask
        return true_labels[keep_mask], pred_labels[keep_mask]


    predictions, dev_labels = remove_correct_class_o(predictions, dev_labels, NER_labels["O"])

    print(predictions.shape, dev_labels.shape)

    accuracy = Accuracy(task='multiclass', num_classes=5)
    accuracy.update(predictions, dev_labels)
    print(accuracy.compute())


def train_POS():
    def remove_punctuation(pos_train_data):
        return [item for item in pos_train_data if item[0] in POS_labels]

    pos_raw_train_data = parse_file(POS_TRAIN)
    pos_train_data = remove_punctuation(pos_raw_train_data)
    train_words, train_labels, train_context_embeddings = arrange_data(pos_train_data, embeddings, POS_labels)


    pos_raw_dev_data = parse_file(POS_TRAIN)
    pos_dev_data = remove_punctuation(pos_raw_dev_data)
    dev_words, dev_labels, dev_context_embeddings = arrange_data(pos_dev_data, embeddings, POS_labels)

    model = POS()
    train(model, train_context_embeddings, train_labels, epochs=100, lr=0.01)

    raw_predictions = model(dev_context_embeddings)
    probabilities = F.softmax(raw_predictions, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    print(predictions.shape, dev_labels.shape)

    accuracy = Accuracy(task='multiclass', num_classes=36)
    accuracy.update(predictions, dev_labels)
    print(accuracy.compute())

train_POS()