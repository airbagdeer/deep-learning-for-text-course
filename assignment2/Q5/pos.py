import string
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy

POS_labels_to_one_hot = {
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

POS_one_hot_to_label = {v: k for k, v in POS_labels_to_one_hot.items()}


def is_punctuation(s):
    return all(char in string.punctuation for char in s) and bool(s)

def parse_file(filepath):
    result = []
    filler = "<pad>"

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    document = []

    def process_document(doc):
        doc = [(word,label) for (word,label) in doc if not is_punctuation(word)]

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


class POS(nn.Module):
    def __init__(self, embedding_matrix, char_vocab_size, max_word_size, window_size):
        super().__init__()

        chars_embeddings_size = 32
        out_channels = window_size

        self.embeddings = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
        self.char_embeddings = nn.Embedding(char_vocab_size, chars_embeddings_size, padding_idx=0)

        self.char_conv = nn.Conv1d(in_channels=chars_embeddings_size, out_channels=out_channels, kernel_size=3)

        self.fc1 = nn.Linear(in_features=(50 + out_channels) * 5, out_features=100, bias=True)
        self.output_layer = nn.Linear(in_features=100, out_features=36, bias=True)

        self.dropout = nn.Dropout(0.3)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, input):
        batch_size = input.shape[0]

        concatenated_features = []

        for example_idx in range(batch_size):
            example_features = []
            example = input[example_idx]

            for word_idx in range(5):
                word = example[word_idx]

                word_embedding = self.embeddings(word[0])

                chars = word[1:]
                char_embeddings = self.char_embeddings(chars)

                char_embeddings = char_embeddings.transpose(0, 1)

                conv_output = self.char_conv(char_embeddings.unsqueeze(0)).squeeze(0)

                max_pooled, _ = torch.max(conv_output, dim=1)

                combined = torch.cat([word_embedding, max_pooled], dim=0)
                example_features.append(combined)

            example_tensor = torch.stack(example_features)
            concatenated_features.append(example_tensor)

        all_features = torch.stack(concatenated_features)

        flattened = all_features.view(batch_size, -1)

        hidden_layer_output = F.tanh(self.fc1(flattened))
        hidden_layer_output = self.dropout(hidden_layer_output)

        output_layer_output = self.output_layer(hidden_layer_output)

        return output_layer_output


def train(model: nn.Module, epochs: int, lr: float = 0.0001, num_of_labels: int = None, train_dataloader = None, dev_dataloader = None):
    device = torch.device("cuda")
    model = model.to(device)

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
    early_stopping_counter= 0

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

        model.eval()
        with torch.no_grad():
            total_dev_accuracy = []
            total_dev_loss = []
            for inputs, labels in dev_dataloader:
                dev_raw_output = model(inputs)
                dev_loss = criterion(dev_raw_output, labels)
                prediction = torch.argmax(F.softmax(dev_raw_output, dim=1), dim=1)
                current_accuracy = accuracy(prediction, labels)
                total_dev_accuracy.append(current_accuracy)
                total_dev_loss.append(dev_loss.item())
        model.train()

        train_loss = np.mean(total_train_loss)
        train_accuracy = np.mean(total_train_accuracy)
        dev_loss = np.mean(total_dev_loss)
        dev_accuracy = np.mean(total_dev_accuracy)

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['dev_loss'].append(dev_loss)
        history['dev_accuracy'].append(dev_accuracy)

        scheduler.step(dev_accuracy)
        # if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, Dev Accuracy = {dev_accuracy:.4f}, Dev Loss = {dev_loss:.4f}")

        if epoch > 5 and history['dev_loss'][epoch-2] < dev_loss:
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0

        if early_stopping_counter >= 5:
            break


    return history



def train_POS(TRAIN_DATA, DEV_DATA, one_hot_embeddings, vocab_size, embedding_matrix, char_to_index, char_vocab_size, max_word_size, window_size):
    pos_raw_train_data = parse_file(TRAIN_DATA)
    train_words, train_labels, train_context_embeddings = arrange_data(pos_raw_train_data, one_hot_embeddings, POS_labels_to_one_hot, char_to_index, max_word_size)

    pos_raw_dev_data = parse_file(DEV_DATA)
    dev_words, dev_labels, dev_context_embeddings = arrange_data(pos_raw_dev_data, one_hot_embeddings, POS_labels_to_one_hot, char_to_index, max_word_size)

    train_dataset = TensorDataset(train_context_embeddings, train_labels)
    train_data = DataLoader(train_dataset, batch_size=128, shuffle=True)

    dev_dataset = TensorDataset(dev_context_embeddings, dev_labels)
    dev_data = DataLoader(dev_dataset, batch_size=128, shuffle=True)

    model = POS(embedding_matrix, char_vocab_size, max_word_size, window_size)

    history = train(model, epochs=100, num_of_labels=len(list(POS_labels_to_one_hot.values())), train_dataloader=train_data, dev_dataloader=dev_data)

    model.eval()

    plot(history['dev_accuracy'], 'POS Dev Accuracy', window_size)
    plot(history['train_accuracy'], 'POS Train Accuracy', window_size)
    plot(history['dev_loss'], 'POS Dev Loss', window_size)
    plot(history['train_loss'], 'POS Train Loss', window_size)

    return model

def remove_punctuation(pos_train_data):
    return [item for item in pos_train_data if item[0] in POS_labels_to_one_hot]

def process_pos_data(embeddings, TRAIN_DATA, TRAIN_CONTEXT_EMBEDDINGS, TRAIN_LABELS, DEV_DATA, DEV_CONTEXT_EMBEDDINGS, DEV_LABELS, save_labels = False):

    pos_raw_train_data = parse_file(TRAIN_DATA)
    pos_train_data = remove_punctuation(pos_raw_train_data)
    train_words, train_labels, train_context_embeddings = arrange_data(pos_train_data, embeddings, POS_labels_to_one_hot)

    pos_raw_dev_data = parse_file(DEV_DATA)
    pos_dev_data = remove_punctuation(pos_raw_dev_data)
    dev_words, dev_labels, dev_context_embeddings = arrange_data(pos_dev_data, embeddings, POS_labels_to_one_hot)

def plot(data, title, window_size):
    plt.plot(data, marker='o')
    plt.title(f'{title}, Final: {data[-1]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.grid(True)
    plt.savefig("./images/" + title + "_" + str(window_size) + ".png")
    plt.show()

def arrange_data(raw_data: [[str, str]], embeddings: Dict[str, int], true_labels: Dict[str, int] = None, char_to_index=None, max_word_length=None):
    words_in_order = []
    labels_in_order = []
    context_embeddings = []

    for [label, words] in raw_data:
        if label not in true_labels:
            continue

        embedded_words = []
        for word in words:
            embedded_word = []
            if word in embeddings:
                embedded_word.append(embeddings[word])
            else:
                embedded_word.append(embeddings["<unk>"])

            embedded_chars = []
            if word == "<pad>":
                embedded_chars = [char_to_index["<pad>"]]*max_word_length
            else:
                for char in word:
                    if char in char_to_index:
                        embedded_chars.append(char_to_index[char])
                    else:
                        embedded_chars.append(char_to_index["<unk>"])
                if len(embedded_chars) < max_word_length:
                    embedded_chars = embedded_chars + [char_to_index["<pad>"]]*(max_word_length - len(embedded_chars))

            embedded_words.append(embedded_word + embedded_chars)

        words_in_order.append(words[2])
        if true_labels:
            labels_in_order.append(torch.tensor(true_labels[label], dtype=torch.long))
        context_embeddings.append(torch.tensor(embedded_words, dtype=torch.long))

    return words_in_order, torch.stack(labels_in_order, dim=0), torch.stack(context_embeddings, dim=0)


def evaluate_pos_file_with_context(model, filepath, word_to_index, char_to_index, max_word_length, output_path):
    PAD_WORD = word_to_index.get('<pad>', 0)
    UNK_WORD = word_to_index.get('<unk>', 1)
    PAD_CHAR = char_to_index.get('<pad>', 0)
    UNK_CHAR = char_to_index.get('<unk>', 1)

    # Function to check if a string is punctuation
    def is_punctuation(s):
        return all(char in string.punctuation for char in s) and bool(s)

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    output_lines = []
    sentence = []

    for line in lines:
        word = line.strip()
        if word == '' or word.startswith('-DOCSTART-'):
            if sentence:
                padded = ['<pad>', '<pad>'] + sentence + ['<pad>', '<pad>']
                for i in range(2, len(padded) - 2):
                    if not is_punctuation(padded[i]):
                        # Create a context window of size 5
                        window = padded[i - 2:i + 3]

                        # Create embedded representation with word and character features
                        embedded_window = []
                        for word in window:
                            # Word embedding
                            word_idx = word_to_index.get(word, UNK_WORD)

                            # Character embeddings
                            char_indices = []
                            if word == "<pad>":
                                char_indices = [PAD_CHAR] * max_word_length
                            else:
                                for char in word:
                                    if char in char_to_index:
                                        char_indices.append(char_to_index[char])
                                    else:
                                        char_indices.append(UNK_CHAR)

                                # Pad or truncate to max_word_length
                                if len(char_indices) > max_word_length:
                                    char_indices = char_indices[:max_word_length]
                                elif len(char_indices) < max_word_length:
                                    char_indices = char_indices + [PAD_CHAR] * (max_word_length - len(char_indices))

                            # Combine word and character features
                            embedded_word = [word_idx] + char_indices
                            embedded_window.append(embedded_word)

                        # Convert to tensor and add batch dimension
                        input_tensor = torch.tensor(embedded_window, dtype=torch.long).unsqueeze(0)

                        # Get model prediction
                        with torch.no_grad():
                            raw_prediction = model(input_tensor)
                            probabilities = torch.softmax(raw_prediction, dim=1)
                            prediction = torch.argmax(probabilities, dim=1).item()

                        # Add to output
                        output_lines.append(f"{padded[i]} {POS_one_hot_to_label[prediction]}")
                    else:
                        # For punctuation, use the punctuation itself as the tag
                        output_lines.append(f"{padded[i]} {padded[i]}")

                # End of sentence
                output_lines.append("")
                sentence = []

            # Add empty line or DOCSTART marker
            if word:
                output_lines.append(word)
                output_lines.append("")
        else:
            word = word.lower()
            sentence.append(word)

    # Process any remaining sentence
    if sentence:
        padded = ['<pad>', '<pad>'] + sentence + ['<pad>', '<pad>']
        for i in range(2, len(padded) - 2):
            if not is_punctuation(padded[i]):
                # Create a context window of size 5
                window = padded[i - 2:i + 3]

                # Create embedded representation with word and character features
                embedded_window = []
                for word in window:
                    # Word embedding
                    word_idx = word_to_index.get(word, UNK_WORD)

                    # Character embeddings
                    char_indices = []
                    if word == "<pad>":
                        char_indices = [PAD_CHAR] * max_word_length
                    else:
                        for char in word:
                            if char in char_to_index:
                                char_indices.append(char_to_index[char])
                            else:
                                char_indices.append(UNK_CHAR)

                        # Pad or truncate to max_word_length
                        if len(char_indices) > max_word_length:
                            char_indices = char_indices[:max_word_length]
                        elif len(char_indices) < max_word_length:
                            char_indices = char_indices + [PAD_CHAR] * (max_word_length - len(char_indices))

                    # Combine word and character features
                    embedded_word = [word_idx] + char_indices
                    embedded_window.append(embedded_word)

                # Convert to tensor and add batch dimension
                input_tensor = torch.tensor(embedded_window, dtype=torch.long).unsqueeze(0)

                # Get model prediction
                with torch.no_grad():
                    raw_prediction = model(input_tensor)
                    probabilities = torch.softmax(raw_prediction, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()

                # Add to output
                output_lines.append(f"{padded[i]} {POS_one_hot_to_label[prediction]}")
            else:
                # For punctuation, use the punctuation itself as the tag
                output_lines.append(f"{padded[i]} {padded[i]}")

    # Write output file
    with open(output_path, 'w', encoding='utf-8') as out:
        for line in output_lines:
            out.write(line + '\n')