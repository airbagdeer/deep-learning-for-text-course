import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CharLSTMCellWordBiLSTMCellTagger(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim, char_hidden_dim, word_hidden_dim, num_classes, words_and_tags_dict=None, pad_idx=0):
        super().__init__()
        self.words_and_tags_dict = words_and_tags_dict
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=pad_idx)

        # Character-level forward LSTMCell only
        self.char_fwd_cell = nn.LSTMCell(char_embed_dim, char_hidden_dim)

        # Word-level BiLSTMCell
        self.word_fwd_cell = nn.LSTMCell(char_hidden_dim, word_hidden_dim)
        self.word_bwd_cell = nn.LSTMCell(char_hidden_dim, word_hidden_dim)

        self.output_fc = nn.Linear(word_hidden_dim * 2, num_classes)
        torch.nn.init.xavier_uniform_(self.char_embed.weight)

    def forward(self, char_ids):
        """
        char_ids: [batch, seq_len, max_word_len]
        """
        batch_size, seq_len, max_word_len = char_ids.size()
        device = char_ids.device

        # Flatten to process all words
        char_ids_flat = char_ids.view(-1, max_word_len)  # [batch*seq_len, max_word_len]
        emb = self.char_embed(char_ids_flat)  # [batch*seq_len, max_word_len, char_embed_dim]

        # Forward LSTMCell over characters
        h_fwd = torch.zeros(char_ids_flat.size(0), self.char_fwd_cell.hidden_size, device=device)
        c_fwd = torch.zeros_like(h_fwd)
        for t in range(max_word_len):
            h_fwd, c_fwd = self.char_fwd_cell(emb[:, t, :], (h_fwd, c_fwd))

        char_repr = h_fwd  # [batch*seq_len, char_hidden]
        word_repr = char_repr.view(batch_size, seq_len, -1)  # [batch, seq_len, char_hidden]

        # Word-level BiLSTMCell
        h_fwd = torch.zeros(batch_size, self.word_fwd_cell.hidden_size, device=device)
        c_fwd = torch.zeros_like(h_fwd)
        fwd_outputs = []
        for t in range(seq_len):
            h_fwd, c_fwd = self.word_fwd_cell(word_repr[:, t, :], (h_fwd, c_fwd))
            fwd_outputs.append(h_fwd)

        h_bwd = torch.zeros(batch_size, self.word_bwd_cell.hidden_size, device=device)
        c_bwd = torch.zeros_like(h_bwd)
        bwd_outputs = []
        for t in reversed(range(seq_len)):
            h_bwd, c_bwd = self.word_bwd_cell(word_repr[:, t, :], (h_bwd, c_bwd))
            bwd_outputs.insert(0, h_bwd)

        outputs = [torch.cat([f, b], dim=1) for f, b in zip(fwd_outputs, bwd_outputs)]
        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, word_hidden*2]

        return self.output_fc(outputs)  # [batch, seq_len, num_classes]

class BiLstmCharLstmTrainer:
    def __init__(self, model, lr=1e-3):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index in labels
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, y_train, X_dev=None, y_dev=None, tag2idx=None, task_type="ner",
              batch_size=32, epochs=5, accuracy_logging_file_path=None):
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        seen_samples = 0

        for epoch in range(epochs):
            total_loss = 0
            last_acc = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(xb)  # [batch, seq_len, num_classes]
                outputs = outputs.view(-1, outputs.shape[-1])  # [batch*seq_len, num_classes]
                yb = yb.view(-1)  # [batch*seq_len]

                loss = self.criterion(outputs, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optimizer.step()
                total_loss += loss.item()

                seen_samples += xb.size(0)
                if X_dev is not None and seen_samples % 500 < batch_size:
                    acc = self.evaluate(X_dev, y_dev, tag2idx=tag2idx, task_type=task_type)
                    last_acc = acc
                    print(f"  [After {seen_samples} samples] Dev Accuracy: {acc:.2f}%")

            print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
            if accuracy_logging_file_path:
                with open(accuracy_logging_file_path, "a") as f:
                    f.write(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}, Accuracy: {last_acc}\n")

    def evaluate(self, X_dev, y_dev, tag2idx=None, task_type="ner"):
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