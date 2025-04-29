from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy

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

class POS(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(in_features=250, out_features=500, bias=True)
        self.fc1 = nn.Linear(in_features=500, out_features=500, bias=True)
        self.output_layer = nn.Linear(in_features=500, out_features=36, bias=True)

        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)


    def forward(self, input):
        input_layer_output = self.input_layer(input)
        hidden_layer_output = F.tanh(self.fc1(input_layer_output))
        output_layer_output = self.output_layer(hidden_layer_output)

        return output_layer_output


def train(model: nn.Module, train_data, train_labels, dev_data, dev_labels, epochs: int, lr: float = 0.01, num_of_labels: int = None):
    history = {
        'dev_loss': [],
        'dev_accuracy': [],
        'train_loss': [],
        'train_accuracy': []
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    accuracy = Accuracy(task='multiclass', num_classes=num_of_labels)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_data)
        history['train_accuracy'].append(accuracy(output, train_labels))
        train_loss = criterion(output, train_labels)
        train_loss.backward()
        optimizer.step()

        history['train_loss'].append(train_loss.item())

        model.eval()
        with torch.no_grad():
            dev_raw_output = model(dev_data)
            dev_loss = criterion(dev_raw_output, dev_labels)
            prediction = torch.argmax(F.softmax(dev_raw_output, dim=1), dim=1)
            current_accuracy = accuracy(prediction, dev_labels)
            history['dev_accuracy'].append(current_accuracy)
            history['dev_loss'].append(dev_loss.item())
        model.train()

        scheduler.step(current_accuracy)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {train_loss:.4f}, Accuracy = {current_accuracy:.4f}")

    return history



def train_POS(TRAIN_CONTEXT_EMBEDDINGS, TRAIN_LABELS, DEV_CONTEXT_EMBEDDINGS, DEV_LABELS, POS_MODEL=None):
    train_labels = torch.load(TRAIN_LABELS)
    train_context_embeddings = torch.load(TRAIN_CONTEXT_EMBEDDINGS)

    dev_labels = torch.load(DEV_LABELS)
    dev_context_embeddings = torch.load(DEV_CONTEXT_EMBEDDINGS)

    model = POS()

    history = train(model, train_context_embeddings, train_labels, dev_context_embeddings, dev_labels, epochs=100, num_of_labels=len(list(POS_labels.values())))

    model.eval()

    if POS_MODEL:
        torch.save(model.state_dict(), POS_MODEL)

    # TODO: add test predictions later
    # raw_predictions = model(dev_context_embeddings)
    # probabilities = F.softmax(raw_predictions, dim=1)
    # predictions = torch.argmax(probabilities, dim=1)
    #
    # accuracy = Accuracy(task='multiclass', num_classes=len(list(POS_labels.values())))

    plot(history['dev_accuracy'], 'POS Dev Accuracy')
    plot(history['train_accuracy'], 'POS Train Accuracy')
    plot(history['dev_loss'], 'POS Dev Loss')
    plot(history['train_loss'], 'POS Train Loss')

def process_pos_data(embeddings, TRAIN_DATA, TRAIN_CONTEXT_EMBEDDINGS, TRAIN_LABELS, DEV_DATA, DEV_CONTEXT_EMBEDDINGS, DEV_LABELS, save_labels = False):
    def remove_punctuation(pos_train_data):
        return [item for item in pos_train_data if item[0] in POS_labels]

    pos_raw_train_data = parse_file(TRAIN_DATA)
    pos_train_data = remove_punctuation(pos_raw_train_data)
    train_words, train_labels, train_context_embeddings = arrange_data(pos_train_data, embeddings, POS_labels)

    torch.save(train_context_embeddings, TRAIN_CONTEXT_EMBEDDINGS)
    if save_labels:
        torch.save(train_labels, TRAIN_LABELS)

    pos_raw_dev_data = parse_file(DEV_DATA)
    pos_dev_data = remove_punctuation(pos_raw_dev_data)
    dev_words, dev_labels, dev_context_embeddings = arrange_data(pos_dev_data, embeddings, POS_labels)

    torch.save(dev_context_embeddings, DEV_CONTEXT_EMBEDDINGS)
    if save_labels:
        torch.save(dev_labels, DEV_LABELS)

def plot(data, title, save_file=None):
    plt.plot(data, marker='o')
    plt.title(f'{title}')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.grid(True)
    plt.show()
    if save_file:
        plt.savefig(save_file)

def arrange_data(raw_data: [[str, str]], embeddings: Dict[str, torch.Tensor], true_labels: Dict[str, int]):
    words_in_order = []
    labels_in_order = []
    context_embeddings = []

    for [label, words] in raw_data:
        embedded_words = torch.cat([embeddings[word] if word in embeddings else embeddings["<unk>"] for word in words], dim=0)
        words_in_order.append(words[2])
        labels_in_order.append(torch.tensor(true_labels[label], dtype=torch.long))
        context_embeddings.append(embedded_words)

    return words_in_order, torch.stack(labels_in_order, dim=0), torch.stack(context_embeddings, dim=0)
