from typing import Dict, Tuple
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy

NER_labels = {
    "PER": 0,
    "LOC": 1,
    "ORG": 2,
    "MISC": 3,
    "O": 4,
}

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
                document.append((word.lower(), label))

    if document:
        process_document(document)

    return result

class NER(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(in_features=250, out_features=100, bias=True)
        self.fc1=nn.Linear(in_features=100, out_features=100, bias=True)
        self.output_layer = nn.Linear(in_features=100, out_features=5, bias=True)

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
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
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
            prediction_without_o, dev_labels_without_o = remove_correct_class_o(prediction, dev_labels, NER_labels["O"])
            current_accuracy = accuracy(prediction_without_o, dev_labels_without_o)
            history['dev_accuracy'].append(current_accuracy)
            history['dev_loss'].append(dev_loss.item())
        model.train()

        # scheduler.step(current_accuracy)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {train_loss:.4f}, Accuracy = {current_accuracy:.4f}")

    return history

def remove_correct_class_o(true_labels: torch.Tensor, pred_labels: torch.Tensor, o_class_number) -> Tuple[
    torch.Tensor, torch.Tensor]:
    correct_class_o_mask = (true_labels == o_class_number) & (pred_labels == o_class_number)
    keep_mask = ~correct_class_o_mask
    return true_labels[keep_mask], pred_labels[keep_mask]

def train_NER(embeddings, TRAIN_DATA, DEV_DATA, NER_MODEL_FILE=None):
    ner_raw_train_data = parse_file(TRAIN_DATA)
    ner_raw_dev_data = parse_file(DEV_DATA)

    train_words, train_labels, train_context_embeddings = arrange_data(ner_raw_train_data, embeddings, NER_labels)
    dev_words, dev_labels, dev_context_embeddings = arrange_data(ner_raw_dev_data, embeddings, NER_labels)

    model = NER()

    history = train(model, train_context_embeddings, train_labels, dev_context_embeddings, dev_labels, epochs=1000, num_of_labels=len(list(NER_labels.values())))
    if NER_MODEL_FILE:
        torch.save(model.state_dict(), NER_MODEL_FILE)

    raw_predictions = model(dev_context_embeddings)
    probabilities = F.softmax(raw_predictions, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

    accuracy = Accuracy(task='multiclass', num_classes=len(list(NER_labels.values())))

    print('before before remove_correct_class_o', accuracy(predictions, dev_labels))

    predictions, dev_labels = remove_correct_class_o(predictions, dev_labels, NER_labels["O"])

    accuracy = Accuracy(task='multiclass', num_classes=len(list(NER_labels.values())))
    print('accuracy after remove_correct_class_o',accuracy(predictions, dev_labels))

    plot(history['dev_accuracy'], 'NER Dev Accuracy without O\'s')
    plot(history['train_accuracy'], 'NER Train Accuracy')
    plot(history['dev_loss'], 'NER Dev Loss')
    plot(history['train_loss'], 'NER Train Loss')

def plot(data, title):
    plt.plot(data, marker='o')
    plt.title(f'{title}')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.grid(True)
    plt.show()