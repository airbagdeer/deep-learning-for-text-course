import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from typing import List
import random

# Set random seeds for reproducibility
torch.manual_seed(0)
random.seed(0)

# Parameters
WINDOW_SIZE = 2
EMBEDDING_DIM = 50
HIDDEN_DIM = 100
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.01

SPECIAL_TOKENS = ["<PAD>", "<UNK>"]

# Helper functions
def read_ner_file(filepath: str):
    sentences = []
    sentence = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            if line.startswith("-DOCSTART-"):
                continue
            word, tag = line.split()
            sentence.append((word, tag))
    if sentence:
        sentences.append(sentence)
    return sentences

def build_vocab(sentences: List[List[tuple]]):
    word_to_idx = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    idx = len(word_to_idx)
    tag_to_idx = {}
    tag_idx = 0

    for sentence in sentences:
        for word, tag in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
            if tag not in tag_to_idx:
                tag_to_idx[tag] = tag_idx
                tag_idx += 1

    idx_to_tag = {i: t for t, i in tag_to_idx.items()}
    return word_to_idx, tag_to_idx, idx_to_tag

def prepare_window(sentence: List[str], word_to_idx: dict):
    padded = ["<PAD>"] * WINDOW_SIZE + sentence + ["<PAD>"] * WINDOW_SIZE
    idxs = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in padded]
    windows = []
    for i in range(WINDOW_SIZE, len(idxs) - WINDOW_SIZE):
        window = idxs[i - WINDOW_SIZE:i + WINDOW_SIZE + 1]
        windows.append(window)
    return windows

def prepare_batch(sentences, word_to_idx, tag_to_idx):
    X = []
    y = []
    for sentence in sentences:
        words = [w for w, _ in sentence]
        tags = [t for _, t in sentence]
        windows = prepare_window(words, word_to_idx)
        labels = [tag_to_idx[t] for t in tags]
        X.extend(windows)
        y.extend(labels)
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Model class

class WindowMLPTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.linear1 = nn.Linear((2 * WINDOW_SIZE + 1) * EMBEDDING_DIM, HIDDEN_DIM)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(HIDDEN_DIM, tagset_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.view(embeds.size(0), -1)
        hidden = self.tanh(self.linear1(embeds))
        out = self.linear2(hidden)
        return out

# Evaluation ignoring correct 'O' predictions
def evaluate(model, sentences, word_to_idx, tag_to_idx, idx_to_tag):
    model.eval()
    X_dev, y_dev = prepare_batch(sentences, word_to_idx, tag_to_idx)
    with torch.no_grad():
        outputs = model(X_dev)
        preds = outputs.argmax(dim=1)

    total = correct = 0
    for pred, gold in zip(preds, y_dev):
        pred_tag = idx_to_tag[pred.item()]
        gold_tag = idx_to_tag[gold.item()]
        if gold_tag == 'O' and pred_tag == 'O':
            continue
        total += 1
        if pred_tag == gold_tag:
            correct += 1

    return correct / total if total > 0 else 0

# Main training loop
def train(train_file, dev_file):
    train_data = read_ner_file(train_file)
    dev_data = read_ner_file(dev_file)

    word_to_idx, tag_to_idx, idx_to_tag = build_vocab(train_data)

    model = WindowMLPTagger(len(word_to_idx), len(tag_to_idx))
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    X_train, y_train = prepare_batch(train_data, word_to_idx, tag_to_idx)

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        total_loss = 0
        for i in range(0, X_train.size(0), BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, dev_data, word_to_idx, tag_to_idx, idx_to_tag)
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f} Dev Accuracy={acc:.4f}")

    return model, word_to_idx, idx_to_tag

# Prediction for test set
def predict(model, sentences, word_to_idx, idx_to_tag, output_file):
    model.eval()
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            words = [w for w, _ in sentence]
            windows = prepare_window(words, word_to_idx)
            X = torch.tensor(windows, dtype=torch.long)
            with torch.no_grad():
                outputs = model(X)
                preds = outputs.argmax(dim=1)

            for word, pred in zip(words, preds):
                tag = idx_to_tag[pred.item()]
                f.write(f"{word} {tag}\n")
            f.write("\n")

# Example usage
if __name__ == "__main__":
    model, word_to_idx, idx_to_tag = train("./ner/train", "./ner/dev")
    test_data = read_ner_file("./ner/test")
    predict(model, test_data, word_to_idx, idx_to_tag, "test1.ner")
