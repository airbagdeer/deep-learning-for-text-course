import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# --- MODEL ---
class BiLSTMWordEmbeddingTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, words_and_tags_dict = None, pad_idx=0):
        super(BiLSTMWordEmbeddingTagger, self).__init__()
        self.words_and_tags_dict = words_and_tags_dict
    
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.fwd_cell = nn.LSTMCell(embed_dim, hidden_dim)
        self.bwd_cell = nn.LSTMCell(embed_dim, hidden_dim)

        self.output_fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        emb = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        # Forward LSTMCell
        h_fwd = torch.zeros(batch_size, self.fwd_cell.hidden_size, device=x.device)
        c_fwd = torch.zeros(batch_size, self.fwd_cell.hidden_size, device=x.device)
        outputs_fwd = []
        for t in range(seq_len):
            h_fwd, c_fwd = self.fwd_cell(emb[:, t, :], (h_fwd, c_fwd))
            outputs_fwd.append(h_fwd)

        # Backward LSTMCell
        h_bwd = torch.zeros(batch_size, self.bwd_cell.hidden_size, device=x.device)
        c_bwd = torch.zeros(batch_size, self.bwd_cell.hidden_size, device=x.device)
        outputs_bwd = []
        for t in reversed(range(seq_len)):
            h_bwd, c_bwd = self.bwd_cell(emb[:, t, :], (h_bwd, c_bwd))
            outputs_bwd.insert(0, h_bwd)

        # Concatenate forward and backward hidden states
        outputs = [torch.cat([f, b], dim=1) for f, b in zip(outputs_fwd, outputs_bwd)]
        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, hidden*2]

        logits = self.output_fc(outputs)  # [batch, seq_len, num_classes]
        return logits

class BiLstmWordEmbeddingSequenceTrainer:
    def __init__(self, model, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda .is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad label ignored
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, y_train, X_dev=None, y_dev=None, tag2idx=None, task_type=None, batch_size=32, epochs=5, accuracy_logging_file_path=None):
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        seen_samples = 0

        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(xb)  # [batch, seq_len, num_classes]
                outputs = outputs.view(-1, outputs.shape[-1])  # [batch*seq_len, num_classes]
                yb = yb.view(-1)  # [batch*seq_len]

                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                seen_samples += xb.size(0)
                if X_dev is not None and seen_samples % 500 < batch_size:
                    acc = self.evaluate(X_dev, y_dev, tag2idx=tag2idx, task_type=task_type)
                    print(f"Epoch {epoch+1}, after {seen_samples} samples - Dev Accuracy: {acc:.2f}%")
                    if accuracy_logging_file_path:
                        with open(accuracy_logging_file_path, "a") as f:
                            f.write(f"{seen_samples},{acc:.2f}\n")
            print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    def evaluate(self, X_dev, y_dev, tag2idx=None, task_type=None):
        self.model.eval()
        dev_dataset = TensorDataset(X_dev, y_dev)
        dev_loader = DataLoader(dev_dataset, batch_size=32)
        total_correct = 0
        total_tokens = 0

        ignore_tag = tag2idx.get('O', 0) if tag2idx and task_type == "ner" else None

        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                preds = torch.argmax(logits, dim=-1)  # [batch, seq_len]

                # always ignore padding (0); only ignore 'O' tag for NER
                if ignore_tag is not None:
                    mask = (yb != 0) & (yb != ignore_tag)
                else:
                    mask = (yb != 0)

                correct = (preds == yb) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
                

        self.model.train()
        return 100.0 * total_correct / total_tokens if total_tokens > 0 else 0.0

    def predict(self, X):
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            preds = torch.argmax(logits, dim=-1)
        return preds.cpu()
