import string
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import Accuracy
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt

NER_TRAIN = r"./ner/train"
NER_DEV = r"./ner/dev"
NER_TEST = r"./ner/test"

POS_TRAIN = r"./pos/train"
POS_DEV = r"./pos/dev"
POS_TEST = r"./pos/test"

# NER_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS = r"./ner_processed/train_context_embeddings.pt"
# NER_PROCESSED_TRAIN_CONTEXT_RANDOM_EMBEDDINGS = r"./ner_processed/train_context_random_embeddings.pt"
# NER_PROCESSED_TRAIN_LABELS = r"./ner_processed/train_labels.pt"
# NER_PROCESSED_DEV_CONTEXT_EMBEDDINGS = r"./ner_processed/dev_context_embeddings.pt"
# NER_PROCESSED_DEV_CONTEXT_RANDOM_EMBEDDINGS = r"./ner_processed/dev_context_random_embeddings.pt"
# NER_PROCESSED_DEV_LABELS = r"./ner_processed/dev_labels.pt"
# NER_PROCESSED_TEST_CONTEXT_EMBEDDINGS = r"./ner_processed/test_context_embeddings.pt"
# NER_PROCESSED_TEST_LABELS = r"./ner_processed/test_labels.pt"

POS_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS = r"./pos_processed/train_context_embeddings.pt"
POS_PROCESSED_TRAIN_CONTEXT_RANDOM_EMBEDDINGS = r"./pos_processed/train_context_random_embeddings.pt"
POS_PROCESSED_TRAIN_LABELS = r"./pos_processed/train_labels.pt"
POS_PROCESSED_DEV_CONTEXT_EMBEDDINGS = r"./pos_processed/dev_context_embeddings.pt"
POS_PROCESSED_DEV_CONTEXT_RANDOM_EMBEDDINGS = r"./pos_processed/dev_context_random_embeddings.pt"
POS_PROCESSED_DEV_LABELS = r"./pos_processed/dev_labels.pt"
POS_PROCESSED_TEST_CONTEXT_EMBEDDINGS = r"./pos_processed/test_context_embeddings.pt"
POS_PROCESSED_TEST_LABELS = r"./pos_processed/test_labels.pt"

VOCAB = r"./embeddings/vocab.txt"
WORD_VECTORS = r"./embeddings/wordVectors.txt"

DTYPE = torch.float32

torch_to_numpy_dtype = {
    torch.float32: np.float32,
    torch.float64: np.float64
}

NER_labels = {
    "PER": 0,
    "LOC": 1,
    "ORG": 2,
    "MISC": 3,
    "O": 4,
}


def load_pretrained_embeddings(VOCAB_FILE_PATH: str, WORD_VECTORS_FILE_PATH: str) -> Dict[str, torch.Tensor]:
    embeddings = {}

    with open(VOCAB_FILE_PATH, "r", encoding="utf-8") as vocab, open(WORD_VECTORS_FILE_PATH, "r",
                                                                     encoding="utf-8") as wordVectors:
        for word, vector in zip(vocab.readlines(), wordVectors.readlines()):
            embeddings[word.replace("\n", "")] = torch.from_numpy(
                np.fromstring(vector.replace(" \n", ""), sep=" ", dtype=torch_to_numpy_dtype[DTYPE]))

    embeddings["<pad>"] = torch.zeros(len(list(embeddings.values())[0]), dtype=DTYPE)
    embeddings["<unk>"] = torch.stack(list(embeddings.values()), dim=1).mean(dim=1)

    return embeddings

def load_random_embeddings(TRAIN_FILE, embeddings_size = 50) -> Dict[str, torch.Tensor]:
    embeddings = {}

    def is_punctuation(s):
        return all(char in string.punctuation for char in s) and bool(s)

    with open(TRAIN_FILE, "r", encoding="utf-8") as train:
        for line in train.readlines():
            words = line.split(" ")
            actual_word = words[0]
            if not is_punctuation(actual_word) and actual_word != "\n" and actual_word != "\t" and actual_word != "\r" and actual_word != " ":
                if actual_word not in embeddings:
                    embeddings[actual_word] = torch.randn(embeddings_size, dtype=DTYPE)

    embeddings["<pad>"] = torch.zeros(len(list(embeddings.values())[0]), dtype=DTYPE)
    embeddings["<unk>"] = torch.stack(list(embeddings.values()), dim=1).mean(dim=1)

    return embeddings


def arrange_data(raw_data: [[str, str]], embeddings: Dict[str, torch.Tensor], true_labels: Dict[str, int]):
    words_in_order = []
    labels_in_order = []
    context_embeddings = []

    for [label, words] in raw_data:
        embedded_words = torch.empty(0, dtype=DTYPE)
        for word in words:
            if word.lower() in embeddings:
                embedded_words = torch.cat((embedded_words, embeddings[word.lower()]), dim=0)
            else:
                embedded_words = torch.cat((embedded_words, embeddings["<unk>"]), dim=0)

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
                document.append((word, label))

    if document:
        process_document(document)

    return result


class NER(nn.Module):
    def __init__(self):
        super().__init__()

        # self.w00 = nn.Parameter(torch.empty((250, 5), dtype=DTYPE), requires_grad=True)
        # self.b00 = nn.Parameter(torch.empty((5), dtype=DTYPE), requires_grad=True)

        self.input_layer = nn.Linear(in_features= 250, out_features=500, bias=True)
        self.fc1=nn.Linear(in_features=500, out_features=250, bias=True)
        self.output_layer = nn.Linear(in_features=250, out_features=5, bias=True)

        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)


    def forward(self, input):
        # input_to_tanh = torch.matmul(input, self.w00) + self.b00
        #
        # output = F.tanh(input_to_tanh)
        #
        # return output
        input_layer_output = self.input_layer(input)
        hidden_layer_output = F.tanh(self.fc1(input_layer_output))
        output_layer_output = self.output_layer(hidden_layer_output)

        return output_layer_output

class POS(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(in_features=250, out_features=250, bias=True)
        self.fc1 = nn.Linear(in_features=250, out_features=250, bias=True)
        self.output_layer = nn.Linear(in_features=250, out_features=36, bias=True)

        # self.w00 = nn.Parameter(torch.empty((250, 36), dtype=DTYPE), requires_grad=True)
        # self.b00 = nn.Parameter(torch.empty((36), dtype=DTYPE), requires_grad=True)
        # nn.init.xavier_uniform_(self.w00)
        # nn.init.xavier_uniform_(self.b00)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)


    def forward(self, input):
        # input_to_tanh = torch.matmul(input, self.w00) + self.b00
        input_layer_output = self.input_layer(input)
        hidden_layer_output = F.tanh(self.fc1(input_layer_output))
        output_layer_output = self.output_layer(hidden_layer_output)

        return output_layer_output


def train(model: nn.Module, train_data, train_labels, dev_data, dev_labels, epochs: int, lr: float = None):
    history = {
        'dev_loss': [],
        'dev_accuracy': [],
        'train_loss': [],
        'train_accuracy': []
    }

    criterion = nn.CrossEntropyLoss()
    if lr:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    accuracy = Accuracy(task='multiclass', num_classes=len(list(POS_labels.values())))

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_data)
        history['train_accuracy'].append(accuracy(output, train_labels))
        # dev_loss = criterion(model(dev_data), dev_labels)
        # history['dev_loss'].append(dev_loss.item())
        train_loss = criterion(output, train_labels)
        train_loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #     dev_output = model(dev_data)
        #     dev_loss = criterion(dev_output, dev_labels)
        #     history['dev_loss'].append(dev_loss.item())

        if not lr:
            scheduler.step(current_accuracy)
        history['train_loss'].append(train_loss.item())

        model.eval()  # set to eval mode
        with torch.no_grad():
            dev_raw_output = model(dev_data)
            prediction = torch.argmax(F.softmax(dev_raw_output, dim=1), dim=1)
            current_accuracy = accuracy(prediction, dev_labels)
            dev_loss = criterion(dev_raw_output, dev_labels)
            history['dev_accuracy'].append(accuracy(prediction, dev_labels))
            history['dev_loss'].append(dev_loss.item())
        model.train()  # switch back to training mode

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {train_loss:.4f}, Accuracy = {current_accuracy:.4f}")

    return history

def remove_correct_class_o(true_labels: torch.Tensor, pred_labels: torch.Tensor, o_class_number) -> Tuple[
    torch.Tensor, torch.Tensor]:
    correct_class_o_mask = (true_labels == o_class_number) & (pred_labels == o_class_number)
    keep_mask = ~correct_class_o_mask
    return true_labels[keep_mask], pred_labels[keep_mask]

def train_NER(embeddings):
    ner_raw_train_data = parse_file(NER_TRAIN)
    ner_raw_dev_data = parse_file(NER_DEV)

    train_words, train_labels, train_context_embeddings = arrange_data(ner_raw_train_data, embeddings, NER_labels)
    dev_words, dev_labels, dev_context_embeddings = arrange_data(ner_raw_dev_data, embeddings, NER_labels)

    model = NER()

    print(train_context_embeddings[0].shape, train_labels.shape, dev_context_embeddings[0].shape, dev_labels.shape)

    history = train(model, train_context_embeddings, train_labels, dev_context_embeddings, dev_labels, epochs=100, lr=0.001)

    raw_predictions = model(dev_context_embeddings)
    probabilities = F.softmax(raw_predictions, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

    accuracy = Accuracy(task='multiclass', num_classes=len(list(NER_labels.values())))

    print('before after remove_correct_class_o', accuracy(predictions, dev_labels))

    predictions, dev_labels = remove_correct_class_o(predictions, dev_labels, NER_labels["O"])

    accuracy = Accuracy(task='multiclass', num_classes=len(list(NER_labels.values())))
    print('accuracy after remove_correct_class_o',accuracy(predictions, dev_labels))

    plot(history['dev_accuracy'], 'Dev Accuracy')
    plot(history['train_accuracy'], 'Train Accuracy')
    plot(history['dev_loss'], 'Dev Loss')
    plot(history['train_loss'], 'Train Loss')


def process_pos_data(embeddings, POS_TRAIN_FILE, POS_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS_FILE, POS_PROCESSED_TRAIN_LABELS_FILE, POS_DEV_FILE, POS_PROCESSED_DEV_CONTEXT_EMBEDDINGS_FILE, POS_PROCESSED_DEV_LABELS_FILE, save_labels = False):
    def remove_punctuation(pos_train_data):
        return [item for item in pos_train_data if item[0] in POS_labels]

    pos_raw_train_data = parse_file(POS_TRAIN_FILE)
    pos_train_data = remove_punctuation(pos_raw_train_data)
    train_words, train_labels, train_context_embeddings = arrange_data(pos_train_data, embeddings, POS_labels)

    torch.save(train_context_embeddings, POS_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS_FILE)
    if save_labels:
        torch.save(train_labels, POS_PROCESSED_TRAIN_LABELS_FILE)

    pos_raw_dev_data = parse_file(POS_DEV_FILE)
    pos_dev_data = remove_punctuation(pos_raw_dev_data)
    dev_words, dev_labels, dev_context_embeddings = arrange_data(pos_dev_data, embeddings, POS_labels)

    torch.save(dev_context_embeddings, POS_PROCESSED_DEV_CONTEXT_EMBEDDINGS_FILE)
    if save_labels:
        torch.save(dev_labels, POS_PROCESSED_DEV_LABELS_FILE)


def train_POS(POS_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS_FILE, POS_PROCESSED_TRAIN_LABELS_FILE, POS_PROCESSED_DEV_CONTEXT_EMBEDDINGS_FILE, POS_PROCESSED_DEV_LABELS_FILE):
    train_labels = torch.load(POS_PROCESSED_TRAIN_LABELS_FILE)
    train_context_embeddings = torch.load(POS_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS_FILE)

    dev_labels = torch.load(POS_PROCESSED_DEV_LABELS_FILE)
    dev_context_embeddings = torch.load(POS_PROCESSED_DEV_CONTEXT_EMBEDDINGS_FILE)

    model = POS()

    history = train(model, train_context_embeddings, train_labels, dev_context_embeddings, dev_labels, epochs=100, lr=0.01)

    model.eval()

    raw_predictions = model(dev_context_embeddings)
    probabilities = F.softmax(raw_predictions, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

    accuracy = Accuracy(task='multiclass', num_classes=len(list(POS_labels.values())))

    plot(history['dev_accuracy'], 'Dev Accuracy')
    plot(history['train_accuracy'], 'Train Accuracy')
    plot(history['dev_loss'], 'Dev Loss')
    plot(history['train_loss'], 'Train Loss')

def plot(data, title):
    plt.plot(data, marker='o')
    plt.title(f'Model {title}')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.grid(True)
    plt.show()

torch.manual_seed(42)
# embeddings = load_pretrained_embeddings(VOCAB, WORD_VECTORS)
# process_pos_data(embeddings, POS_TRAIN, POS_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS, POS_PROCESSED_TRAIN_LABELS, POS_DEV, POS_PROCESSED_DEV_CONTEXT_EMBEDDINGS, POS_PROCESSED_DEV_LABELS)
# train_POS(POS_PROCESSED_TRAIN_CONTEXT_EMBEDDINGS, POS_PROCESSED_TRAIN_LABELS, POS_PROCESSED_DEV_CONTEXT_EMBEDDINGS, POS_PROCESSED_DEV_LABELS)


# random_embeddings = load_random_embeddings(POS_TRAIN)

# process_pos_data(random_embeddings, POS_TRAIN, POS_PROCESSED_TRAIN_CONTEXT_RANDOM_EMBEDDINGS, POS_PROCESSED_TRAIN_LABELS, POS_DEV, POS_PROCESSED_DEV_CONTEXT_RANDOM_EMBEDDINGS, POS_PROCESSED_DEV_LABELS)
train_POS(POS_PROCESSED_TRAIN_CONTEXT_RANDOM_EMBEDDINGS, POS_PROCESSED_TRAIN_LABELS, POS_PROCESSED_DEV_CONTEXT_RANDOM_EMBEDDINGS, POS_PROCESSED_DEV_LABELS)

# train_NER(embeddings=random_embeddings)