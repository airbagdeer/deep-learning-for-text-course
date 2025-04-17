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


def load_embeddings(VOCAB_FILE_PATH: str, WORD_VECTORS_FILE_PATH: str) -> Dict[str, torch.Tensor]:
    embeddings = {}

    with open(VOCAB_FILE_PATH, "r", encoding="utf-8") as vocab, open(WORD_VECTORS_FILE_PATH, "r",
                                                                     encoding="utf-8") as wordVectors:
        for word, vector in zip(vocab.readlines(), wordVectors.readlines()):
            embeddings[word.replace("\n", "")] = torch.from_numpy(np.fromstring(vector.replace(" \n", ""), sep=" "))

    embeddings["<pad>"] = torch.stack(list(embeddings.values()), dim=1).mean(dim=1)

    return embeddings


def arrange_ner_data(ner_raw_data: [[str, str]], embeddings: Dict[str, torch.Tensor]) -> [(str, torch.Tensor),
                                                                                          [torch.Tensor]]:
    ner_training_data = []

    for [label, words] in ner_raw_data:
        embedded_words = torch.empty(0, dtype=torch.float64)
        for word in words:
            if word.lower() in embeddings:
                embedded_words = torch.cat((embedded_words, embeddings[word.lower()]), dim=0)
                # embedded_words.append(embeddings[word.lower()])
            else:
                embedded_words = torch.cat((embedded_words, embeddings["<pad>"]), dim=0)
                # embedded_words.append(embeddings["<pad>"])

        ner_training_data.append((words[2], torch.tensor(NER_labels[label]), embedded_words))

    return ner_training_data


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


class NER(nn.Module):
    def __init__(self):
        super().__init__()

        self.w00 = nn.Parameter(torch.randn((250, 5), dtype=torch.float64), requires_grad=True)
        self.b00 = nn.Parameter(torch.randn((5), dtype=torch.float64), requires_grad=True)

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
ner_raw_train_data = parse_ner_file(NER_TRAIN)
ner_raw_dev_data = parse_ner_file(NER_DEV)

ner_train_data_organized = arrange_ner_data(ner_raw_train_data, embeddings)
ner_dev_data_organized = arrange_ner_data(ner_raw_dev_data, embeddings)

train_context_embeddings = torch.stack(
    [train_organized_sample[2] for train_organized_sample in ner_train_data_organized], dim=0)
train_labels = torch.stack([train_organized_sample[1] for train_organized_sample in ner_train_data_organized], dim=0)
train_words = [train_organized_sample[0] for train_organized_sample in ner_train_data_organized]

dev_context_embeddings = torch.stack([dev_organized_sample[2] for dev_organized_sample in ner_dev_data_organized],
                                     dim=0)
dev_labels = torch.stack([dev_organized_sample[1] for dev_organized_sample in ner_dev_data_organized], dim=0)
dev_words = [dev_organized_sample[0] for dev_organized_sample in ner_dev_data_organized]

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
