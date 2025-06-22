import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class PrefixSuffixBiLSTMTagger(nn.Module):
    def __init__(self,
                 embed_dim,
                 word_vocab_size,
                 prefix_vocab_size,
                 suffix_vocab_size,
                 word_hidden_dim, num_classes, words_and_tags_dict=None, pad_idx=0):
        super().__init__()
        self.words_and_tags_dict = words_and_tags_dict

        self.word_embed = nn.Embedding(word_vocab_size, embed_dim, padding_idx=pad_idx)
        self.prefix_embed = nn.Embedding(prefix_vocab_size, embed_dim, padding_idx=pad_idx)
        self.suffix_embed = nn.Embedding(suffix_vocab_size, embed_dim, padding_idx=pad_idx)


        self.fwd_cell = nn.LSTMCell(embed_dim, word_hidden_dim)
        self.bwd_cell = nn.LSTMCell(embed_dim, word_hidden_dim)

        self.output_fc = nn.Linear(2 * word_hidden_dim, num_classes)

        torch.nn.init.xavier_uniform_(self.word_embed.weight)
        torch.nn.init.xavier_uniform_(self.prefix_embed.weight)
        torch.nn.init.xavier_uniform_(self.suffix_embed.weight)

    def forward(self, word_ids, prefix_ids, suffix_ids):
        batch_size, seq_len = word_ids.size()
        device = word_ids.device

        word_embs = self.word_embed(word_ids)
        prefix_embs = self.prefix_embed(prefix_ids)
        suffix_embs = self.suffix_embed(suffix_ids)

        combined = word_embs + prefix_embs + suffix_embs

        h_fwd = torch.zeros(batch_size, self.fwd_cell.hidden_size, device=device)
        c_fwd = torch.zeros_like(h_fwd)
        fwd_outputs = []
        for t in range(seq_len):
            h_fwd, c_fwd = self.fwd_cell(combined[:, t, :], (h_fwd, c_fwd))
            fwd_outputs.append(h_fwd)

        h_bwd = torch.zeros(batch_size, self.bwd_cell.hidden_size, device=device)
        c_bwd = torch.zeros_like(h_bwd)
        bwd_outputs = []
        for t in reversed(range(seq_len)):
            h_bwd, c_bwd = self.bwd_cell(combined[:, t, :], (h_bwd, c_bwd))
            bwd_outputs.insert(0, h_bwd)

        outputs = [torch.cat([f, b], dim=1) for f, b in zip(fwd_outputs, bwd_outputs)]
        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, 2 * hidden_dim]

        logits = self.output_fc(outputs)  # [batch, seq_len, num_classes]
        return logits


class PrefixSuffixDataset(Dataset):
    def __init__(self, word_ids, prefix_ids, suffix_ids, labels=None):
        self.word_ids = word_ids
        self.prefix_ids = prefix_ids
        self.suffix_ids = suffix_ids
        self.labels = labels

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, idx):
        if (self.labels != None):
            return (self.word_ids[idx], self.prefix_ids[idx], self.suffix_ids[idx], self.labels[idx])
        else:
            return (self.word_ids[idx], self.prefix_ids[idx], self.suffix_ids[idx])


class PrefixSuffixTrainer:
    def __init__(self, model, lr=1e-3):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_words, X_prefixes, X_suffixes, y,
              X_dev_words=None, X_dev_prefixes=None, X_dev_suffixes=None, y_dev=None,
              tag2idx=None, task_type=None,
              batch_size=32, epochs=5, accuracy_logging_file_path=None):

        dataset = PrefixSuffixDataset(X_words, X_prefixes, X_suffixes, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            seen_samples = 0

            for word_batch, prefix_batch, suffix_batch, y_batch in loader:
                word_batch = word_batch.to(self.device)
                prefix_batch = prefix_batch.to(self.device)
                suffix_batch = suffix_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(word_batch, prefix_batch, suffix_batch)

                outputs = outputs.view(-1, outputs.shape[-1])
                y_batch = y_batch.view(-1)

                loss = self.criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optimizer.step()

                total_loss += loss.item()
                seen_samples += word_batch.size(0)

                if X_dev_words is not None and seen_samples % 500 < batch_size:
                    acc = self.evaluate(X_dev_words, X_dev_prefixes, X_dev_suffixes, y_dev, tag2idx=tag2idx, task_type=task_type)
                    print(f"Epoch {epoch+1}, after {seen_samples} samples - Dev Accuracy: {acc:.2f}%")
                    if accuracy_logging_file_path:
                        with open(accuracy_logging_file_path, "a") as f:
                            f.write(f"{seen_samples},{acc:.2f}\n")

            print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss / len(loader):.4f}")

    def evaluate(self, X_words, X_prefixes, X_suffixes, y_true, tag2idx=None, task_type=None):
        self.model.eval()
        dataset = PrefixSuffixDataset(X_words, X_prefixes, X_suffixes, y_true)
        loader = DataLoader(dataset, batch_size=32)

        total_correct = 0
        total_tokens = 0
        ignore_tag = tag2idx.get('O', 0) if tag2idx and task_type == "ner" else None

        with torch.no_grad():
            for word_batch, prefix_batch, suffix_batch, y_batch in loader:
                word_batch = word_batch.to(self.device)
                prefix_batch = prefix_batch.to(self.device)
                suffix_batch = suffix_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(word_batch, prefix_batch, suffix_batch)
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

    def predict(self, X_words, X_prefixes, X_suffixes):
        self.model.eval()
        X_words = X_words.to(self.device)
        X_prefixes = X_prefixes.to(self.device)
        X_suffixes = X_suffixes.to(self.device)
        with torch.no_grad():
            logits = self.model(X_words, X_prefixes, X_suffixes)
            preds = torch.argmax(logits, dim=-1)
        return preds.cpu()