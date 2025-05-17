import numpy as np
import torch.nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics.classification import BinaryAccuracy
import time

from gen_examples import generate_custom_sentence, sequence_max_length

torch.manual_seed(42)

vocab_size = 14
char_embedding_dim = 10
lstm_hidden_state_size = 50
batch_size = 64
amount_of_batches = 100
mlp_hidden_layer_size = 50
lr = 0.01
epochs = 5
train_percentage = 0.8
accuracy_threshold = 0.5
pad_token = "<SHORT>"

labels2idx = {
    "neg": 0,
    "pos": 1,
}

# vocab for only non numerical values
vocab2encoding = {
    pad_token : 0,
    "a": 10,
    "b": 11,
    "c": 12,
    "d": 13
}

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.char_embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=char_embedding_dim,
            padding_idx=vocab2encoding[pad_token]
        )
        self.lstmCell = torch.nn.LSTMCell(char_embedding_dim, lstm_hidden_state_size)
        self.MLP_hidden_layer = torch.nn.Linear(lstm_hidden_state_size, mlp_hidden_layer_size)
        self.MLP_output_layer = torch.nn.Linear(mlp_hidden_layer_size, 1)

    def forward(self, input):
        embeddings = self.char_embedding(input)
        batch_size = embeddings.size(0)
        seq_len = embeddings.size(1)

        pad_idx = vocab2encoding[pad_token]
        mask = (input != pad_idx).float()

        h = torch.zeros(batch_size, lstm_hidden_state_size)
        c = torch.zeros(batch_size, lstm_hidden_state_size)

        for char in range(seq_len):
            current_mask = mask[:, char].unsqueeze(1)
            h_new, c_new = self.lstmCell(embeddings[:, char, :], (h, c))
            h = h_new * current_mask + h * (1 - current_mask)
            c = c_new * current_mask + c * (1 - current_mask)

        hidden_layer_output = self.MLP_hidden_layer(h)
        output = torch.nn.functional.sigmoid(self.MLP_output_layer(hidden_layer_output))

        return output

def create_training_data(amount_of_samples):
    pos_sentences = []
    neg_sentences = []

    for _ in range(amount_of_samples):
        pos_sentences.append(encode_to_one_hot(generate_custom_sentence("pos")))
        neg_sentences.append(encode_to_one_hot(generate_custom_sentence("neg")))

    pos_sentences = torch.stack(pos_sentences, dim=0)
    neg_sentences = torch.stack(neg_sentences, dim=0)

    pos_labels = torch.ones(amount_of_samples, dtype=torch.float)
    neg_labels = torch.zeros(amount_of_samples, dtype=torch.float)

    data = torch.cat((neg_sentences, pos_sentences))
    labels = torch.cat((neg_labels, pos_labels))

    dataset = TensorDataset(data, labels)

    train_size = int(train_percentage * len(dataset))
    dev_size = len(dataset) - train_size

    train_dataset, dev_dataset = random_split(dataset, [train_size, dev_size])

    train_sentences = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_sentences = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

    return train_sentences, dev_sentences


def encode_to_one_hot(sentence: str):
    encoding = [vocab2encoding[char] if char in vocab2encoding else int(char) for char in sentence]
    if len(encoding) < sequence_max_length:
        padding = [vocab2encoding[pad_token]] * (sequence_max_length - len(encoding))
        encoding.extend(padding)
    return torch.tensor(encoding)

def train(model: torch.nn.Module, train_dataloader: DataLoader, dev_dataloader: DataLoader):
    history = {
        'dev_loss': [],
        'dev_accuracy': [],
        'train_loss': [],
        'train_accuracy': []
    }

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    accuracy = BinaryAccuracy(threshold=accuracy_threshold)

    for epoch in range(epochs):
        total_train_loss = []
        total_train_accuracy = []
        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(inputs)
            output = torch.squeeze(output)
            total_train_accuracy.append(accuracy(output, labels))
            train_loss = criterion(output, labels)
            total_train_loss.append(train_loss.item())
            train_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total_dev_accuracy = []
            total_dev_loss = []
            for dev_inputs, dev_labels in dev_dataloader:
                dev_output = model(dev_inputs)
                dev_output = torch.squeeze(dev_output)
                dev_loss = criterion(dev_output, dev_labels)
                current_accuracy = accuracy(dev_output, dev_labels)
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

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, Dev Accuracy = {dev_accuracy:.4f}, Dev Loss = {dev_loss:.4f}")

    return history


def plot(data, title):
    plt.plot(data, marker='o')
    plt.title(f'{title}, Final: {data[-1]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.grid(True)
    plt.savefig("./images/" + title + ".png")
    plt.show()


if __name__ == "__main__":
    train_dataloader, dev_dataloader = create_training_data(amount_of_batches*batch_size)
    model = RNN()

    start_time = time.time()
    history = train(model, train_dataloader, dev_dataloader)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")

    plot(history['dev_accuracy'], 'Dev Accuracy')
    plot(history['train_accuracy'], 'Train Accuracy')
    plot(history['dev_loss'], 'Dev Loss')
    plot(history['train_loss'], 'Train Loss')