import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 1. Model definition

class CharWordBiLSTMTagger(nn.Module):
    def __init__(self, 
                 word_vocab_size, word_embed_dim,
                 char_vocab_size, char_embed_dim, char_hidden_dim,
                 word_hidden_dim, num_classes, words_and_tags_dict=None, pad_idx=0):
        super().__init__()

        self.words_and_tags_dict  = words_and_tags_dict
            
        self.word_embed = nn.Embedding(word_vocab_size, word_embed_dim, padding_idx=pad_idx)
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=pad_idx)

        # Char-level forward LSTMCell
        self.char_fwd_cell = nn.LSTMCell(char_embed_dim, char_hidden_dim)

        # Word-level BiLSTMCell
        self.word_fwd_cell = nn.LSTMCell(char_hidden_dim + word_embed_dim, word_hidden_dim)
        self.word_bwd_cell = nn.LSTMCell(char_hidden_dim + word_embed_dim, word_hidden_dim)

        self.output_fc = nn.Linear(word_hidden_dim * 2, num_classes)

        torch.nn.init.xavier_uniform_(self.word_embed.weight)
        torch.nn.init.xavier_uniform_(self.char_embed.weight)

    def forward(self, word_ids, char_ids):
        """
        word_ids: [batch, seq_len]
        char_ids: [batch, seq_len, max_word_len]
        """
        batch_size, seq_len, max_word_len = char_ids.size()
        device = char_ids.device

        # Char-level embedding per word
        char_ids_flat = char_ids.view(-1, max_word_len)  # [batch*seq_len, max_word_len]
        emb = self.char_embed(char_ids_flat)  # [batch*seq_len, max_word_len, char_embed_dim]

        word_lengths = (char_ids_flat != 0).sum(dim=-1)  # [batch*seq_len]

        h_fwd = torch.zeros(char_ids_flat.size(0), self.char_fwd_cell.hidden_size, device=device)
        c_fwd = torch.zeros_like(h_fwd)
        for t in range(max_word_len):
            mask_t = (t < word_lengths).float().unsqueeze(-1)
            h_t, c_t = self.char_fwd_cell(emb[:, t, :], (h_fwd, c_fwd))
            h_fwd = h_t * mask_t + h_fwd * (1 - mask_t)
            c_fwd = c_t * mask_t + c_fwd * (1 - mask_t)
        char_repr = h_fwd.view(batch_size, seq_len, -1)  # [batch, seq_len, char_hidden_dim]

        # Word embedding
        word_emb = self.word_embed(word_ids)  # [batch, seq_len, word_embed_dim]

        # Concatenate char & word embeddings
        combined = torch.cat([word_emb, char_repr], dim=-1)  # [batch, seq_len, char_hidden_dim + word_embed_dim]

        # Word-level BiLSTMCell forward
        h_fwd = torch.zeros(batch_size, self.word_fwd_cell.hidden_size, device=device)
        c_fwd = torch.zeros_like(h_fwd)
        fwd_outputs = []
        for t in range(seq_len):
            h_fwd, c_fwd = self.word_fwd_cell(combined[:, t, :], (h_fwd, c_fwd))
            fwd_outputs.append(h_fwd)

        # Word-level BiLSTMCell backward
        h_bwd = torch.zeros(batch_size, self.word_bwd_cell.hidden_size, device=device)
        c_bwd = torch.zeros_like(h_bwd)
        bwd_outputs = []
        for t in reversed(range(seq_len)):
            h_bwd, c_bwd = self.word_bwd_cell(combined[:, t, :], (h_bwd, c_bwd))
            bwd_outputs.insert(0, h_bwd)

        outputs = [torch.cat([f, b], dim=1) for f, b in zip(fwd_outputs, bwd_outputs)]
        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, word_hidden_dim*2]

        logits = self.output_fc(outputs)  # [batch, seq_len, num_classes]
        return logits


# 2. Dataset class that returns word_ids, char_ids, and labels

class WordCharTagDataset(Dataset):
    def __init__(self, word_ids, char_ids, labels):
        self.word_ids = word_ids
        self.char_ids = char_ids
        self.labels = labels

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, idx):
        return self.word_ids[idx], self.char_ids[idx], self.labels[idx]


# 3. Trainer class

class CharWordBilstmTrainer:
    def __init__(self, model, lr=1e-3):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train_words, X_train_chars, y_train,
              X_dev_words=None, X_dev_chars=None, y_dev=None,
              tag2idx=None, task_type=None,
              batch_size=32, epochs=5, accuracy_logging_file_path=None):

        train_dataset = WordCharTagDataset(X_train_words, X_train_chars, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            seen_samples = 0

            for word_batch, char_batch, y_batch in train_loader:
                word_batch = word_batch.to(self.device)
                char_batch = char_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(word_batch, char_batch)  # [batch, seq_len, num_classes]
                outputs = outputs.view(-1, outputs.shape[-1])
                y_batch = y_batch.view(-1)

                loss = self.criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optimizer.step()

                total_loss += loss.item()
                seen_samples += word_batch.size(0)

                # Optional evaluation during training
                if X_dev_words is not None and seen_samples % 500 < batch_size:
                    acc = self.evaluate(X_dev_words, X_dev_chars, y_dev, tag2idx=tag2idx, task_type=task_type)
                    print(f"Epoch {epoch+1}, after {seen_samples} samples - Dev Accuracy: {acc:.2f}%")
                    if accuracy_logging_file_path:
                        with open(accuracy_logging_file_path, "a") as f:
                            f.write(f"{seen_samples},{acc:.2f}\n")
            print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss / len(train_loader):.4f}")

    def evaluate(self, X_words, X_chars, y_true, tag2idx=None, task_type=None):
        self.model.eval()
        dev_dataset = WordCharTagDataset(X_words, X_chars, y_true)
        dev_loader = DataLoader(dev_dataset, batch_size=32)

        total_correct = 0
        total_tokens = 0
        ignore_tag = tag2idx.get('O', 0) if tag2idx and task_type == "ner" else None

        with torch.no_grad():
            for word_batch, char_batch, y_batch in dev_loader:
                word_batch = word_batch.to(self.device)
                char_batch = char_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(word_batch, char_batch)
                preds = torch.argmax(logits, dim=-1)

                if ignore_tag is not None:
                    mask = (y_batch != 0) & (y_batch != ignore_tag)
                else:
                    mask = (y_batch != 0)

                correct = (preds == y_batch) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

        self.model.train()
        return 100.0 * total_correct / total_tokens if total_tokens > 0 else 0.0

    def predict(self, X_words, X_chars):
        self.model.eval()
        X_words = X_words.to(self.device)
        X_chars = X_chars.to(self.device)
        with torch.no_grad():
            logits = self.model(X_words, X_chars)
            preds = torch.argmax(logits, dim=-1)
        return preds.cpu()