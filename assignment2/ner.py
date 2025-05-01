from typing import Dict, Tuple

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy

from utils import DTYPE

NER_labels_to_one_hot = {
    "PER": 0,
    "LOC": 1,
    "ORG": 2,
    "MISC": 3,
    "O": 4,
}

NER_one_hot_to_label = {v: k for k, v in NER_labels_to_one_hot.items()}

def arrange_data(raw_data: [[str, str]], embeddings: Dict[str, int], true_labels: Dict[str, int] = None):
    words_in_order = []
    labels_in_order = []
    context_embeddings = []

    for [label, words] in raw_data:
        embedded_words = torch.tensor([embeddings[word] if word in embeddings else embeddings["<unk>"] for word in words], dtype=torch.long)
        words_in_order.append(words[2])
        if true_labels:
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

        self.embeddings = nn.Embedding(num_embeddings=21013, embedding_dim=50, padding_idx=0)

        self.fc1=nn.Linear(in_features=250, out_features=100, bias=True)
        self.output_layer = nn.Linear(in_features=100, out_features=5, bias=True)

        self.dropout = nn.Dropout(0.3)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)


    def forward(self, input):
        embedded = self.embeddings(input)
        if embedded.dim()==3:
            flattened = embedded.view(input.shape[0], -1)
        else:
            flattened = embedded.view(-1)
        hidden_layer_output = F.tanh(self.fc1(flattened))
        hidden_layer_output = self.dropout(hidden_layer_output)
        output_layer_output = self.output_layer(hidden_layer_output)

        return output_layer_output

def train(model: nn.Module, epochs: int, lr: float = 0.0001, num_of_labels: int = None, train_dataloader=None, dev_dataloader=None):
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
    early_stopping_counter = 0

    for epoch in range(epochs):
        total_train_loss = []
        total_train_accuracy = []
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            output = model(inputs)
            total_train_accuracy.append(accuracy(output, labels))
            train_loss = criterion(output, labels)
            total_train_loss.append(train_loss.item())
            train_loss.backward()
            optimizer.step()


        history['train_accuracy'].append(np.mean(total_train_accuracy))
        history['train_loss'].append(np.mean(total_train_loss))

        model.eval()
        with torch.no_grad():
            total_dev_accuracy = []
            total_dev_loss = []
            for inputs, labels in dev_dataloader:
                dev_raw_output = model(inputs)
                dev_loss = criterion(dev_raw_output, labels)
                prediction = torch.argmax(F.softmax(dev_raw_output, dim=1), dim=1)
                current_accuracy = accuracy(prediction, labels)
                prediction_without_o, dev_labels_without_o = remove_correct_class_o(prediction, labels, NER_labels_to_one_hot["O"])
                current_accuracy_without_o = accuracy(prediction_without_o, dev_labels_without_o)
                total_dev_accuracy.append(current_accuracy_without_o)
                total_dev_loss.append(dev_loss.item())
        model.train()

        dev_loss = np.mean(total_dev_loss)
        history['dev_loss'].append(dev_loss)
        current_accuracy_without_o = np.mean(total_dev_accuracy)
        history['dev_accuracy'].append(current_accuracy_without_o)

        # scheduler.step(current_accuracy_without_o)
        train_loss = sum(total_train_loss) / len(total_train_loss)
        # if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Dev Loss = {dev_loss:.4f}, Accuracy without o\'s = {current_accuracy_without_o:.4f}, Accuracy with o\'s = {current_accuracy:.4f}")

        if epoch > 5 and history['dev_loss'][epoch-5] < dev_loss:
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0

        if early_stopping_counter >= 5:
            break


    return history

def remove_correct_class_o(true_labels: torch.Tensor, pred_labels: torch.Tensor, o_class_number) -> Tuple[
    torch.Tensor, torch.Tensor]:
    correct_class_o_mask = (true_labels == o_class_number) & (pred_labels == o_class_number)
    keep_mask = ~correct_class_o_mask
    return true_labels[keep_mask], pred_labels[keep_mask]

def train_NER(one_hot_embedding, TRAIN_DATA, DEV_DATA, vocab_size):
    ner_raw_train_data = parse_file(TRAIN_DATA)
    ner_raw_dev_data = parse_file(DEV_DATA)

    train_words, train_labels, train_context_embeddings = arrange_data(ner_raw_train_data, one_hot_embedding, NER_labels_to_one_hot)
    dev_words, dev_labels, dev_context_embeddings = arrange_data(ner_raw_dev_data, one_hot_embedding, NER_labels_to_one_hot)

    train_dataset = TensorDataset(train_context_embeddings, train_labels)
    train_data = DataLoader(train_dataset, batch_size=128, shuffle=True)

    dev_dataset = TensorDataset(dev_context_embeddings, dev_labels)
    dev_data = DataLoader(dev_dataset, batch_size=128, shuffle=True)

    model = NER()

    history = train(model, epochs=100, num_of_labels=len(list(NER_labels_to_one_hot.values())), train_dataloader=train_data, dev_dataloader=dev_data)

    raw_predictions = model(dev_context_embeddings)
    probabilities = F.softmax(raw_predictions, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

    accuracy = Accuracy(task='multiclass', num_classes=len(list(NER_labels_to_one_hot.values())))

    print('before before remove_correct_class_o', accuracy(predictions, dev_labels))

    predictions, dev_labels = remove_correct_class_o(predictions, dev_labels, NER_labels_to_one_hot["O"])

    accuracy = Accuracy(task='multiclass', num_classes=len(list(NER_labels_to_one_hot.values())))
    print('accuracy after remove_correct_class_o',accuracy(predictions, dev_labels))

    plot(history['dev_accuracy'], 'NER Dev Accuracy without O\'s')
    plot(history['train_accuracy'], 'NER Train Accuracy')
    plot(history['dev_loss'], 'NER Dev Loss')
    plot(history['train_loss'], 'NER Train Loss')

    return model

def plot(data, title):
    plt.plot(data, marker='o')
    plt.title(f'{title}')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.grid(True)
    plt.show()
    plt.savefig("./images/" + title + ".png")

def evaluate_test_file(model, TEST_DATA, one_hot_words):
    model.eval()

    ner_raw_test_data = parse_file(TEST_DATA)

    test_words, _, test_context_embeddings = arrange_data(ner_raw_test_data, one_hot_words, NER_labels_to_one_hot)

    test_dataset = TensorDataset(test_context_embeddings)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    for data in test_dataloader:
        predictions = model(data)

    raw_predictions = model(test_context_embeddings)
    probabilities = F.softmax(raw_predictions, dim=1)
    predictions = torch.argmax(probabilities, dim=1)


def evaluate_ner_file_with_context(model, filepath, one_hot_encoding, output_path):
    PAD = one_hot_encoding.get('<pad>')
    UNK = one_hot_encoding.get('<unk>')

    def encode(word):
        return one_hot_encoding.get(word, UNK)

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    output_lines = []
    sentence = []

    for line in lines:
        word = line.strip()
        if word == '' or word == '-DOCSTART-':
            if sentence:
                padded = ['<pad>', '<pad>'] + sentence + ['<pad>', '<pad>']
                for i in range(2, len(padded) - 2):
                    window = padded[i - 2:i + 3]
                    encoded_window = torch.tensor([encode(w) for w in window])
                    raw_prediction = model(encoded_window)
                    probabilities = F.softmax(raw_prediction, dim=0)
                    prediction = torch.argmax(probabilities, dim=0)
                    output_lines.append(f"{padded[i]} {NER_one_hot_to_label[prediction.item()]}")
                sentence = []
            output_lines.append(word)
        else:
            word=word.lower()
            sentence.append(word)

    if sentence:
        padded = ['<pad>', '<pad>'] + sentence + ['<pad>', '<pad>']
        for i in range(2, len(padded) - 2):
            window = padded[i - 2:i + 3]
            encoded_window = torch.tensor([encode(w) for w in window])
            raw_prediction = model(encoded_window)
            probabilities = F.softmax(raw_prediction, dim=0)
            prediction = torch.argmax(probabilities,dim=0)

            output_lines.append(f"{padded[i]}\t{NER_one_hot_to_label[prediction.item()]}")

    with open(output_path, 'w', encoding='utf-8') as out:
        for line in output_lines:
            out.write(line + '\n')

