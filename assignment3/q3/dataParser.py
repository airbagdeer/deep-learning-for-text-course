import torch
from collections import defaultdict


class DataParser:
    @staticmethod
    def parse_test(test_path, word2idx):
        test_data = DataParser._load_file(test_path)
        train_X = DataParser._encode(test_data, build_vocab=True, no_labels=True)
        train_X = DataParser._pad_batch(train_X, pad_value=word2idx["<PAD>"])
        return train_X


    @staticmethod
    def parse(train_path, dev_path):
        train_data = DataParser._load_file(train_path)
        dev_data = DataParser._load_file(dev_path)

        # --- Vocab and Tag mappings ---
        word2idx = defaultdict(lambda: len(word2idx))
        word2idx["<PAD>"]
        word2idx["<UNK>"]

        tag2idx = defaultdict(lambda: len(tag2idx))
        tag2idx["<PAD>"]  # 0

        train_X, train_y = DataParser._encode(train_data, word2idx, tag2idx, build_vocab=True)
        dev_X, dev_y = DataParser._encode(dev_data, word2idx, tag2idx, build_vocab=False)
        

        train_X = DataParser._pad_batch(train_X, pad_value=word2idx["<PAD>"])
        train_y = DataParser._pad_batch(train_y, pad_value=tag2idx["<PAD>"])
        dev_X = DataParser._pad_batch(dev_X, pad_value=word2idx["<PAD>"])
        dev_y = DataParser._pad_batch(dev_y, pad_value=tag2idx["<PAD>"])

        # --- Convert to Tensors ---
        train_X = torch.tensor(train_X, dtype=torch.long)
        train_y = torch.tensor(train_y, dtype=torch.long)
        dev_X = torch.tensor(dev_X, dtype=torch.long)
        dev_y = torch.tensor(dev_y, dtype=torch.long)

        # --- Reverse mappings ---
        idx2tag = {v: k for k, v in tag2idx.items()}
        idx2word = {v: k for k, v in word2idx.items()}

        return train_X, train_y, dev_X, dev_y, dict(tag2idx), idx2tag, dict(word2idx), idx2word, len(word2idx)

    @staticmethod
    def _load_file(path, no_labels=False):
            if(no_labels):
                data = []
                sent = []
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            sent.append(line.lower())
                        else:
                            if sent:
                                data.append(sent)
                                sent = []
                    if sent:
                        data.append(sent)
                return data
            else:
                data = []
                sent, tags = [], []
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            word, tag = line.split()
                            sent.append(word.lower())
                            tags.append(tag)
                        else:
                            if sent:
                                data.append((sent, tags))
                                sent, tags = [], []
                    if sent:
                        data.append((sent, tags))
                return data
    
    @staticmethod
    def _encode(data, word2idx, tag2idx=None, build_vocab=True, no_labels=False):
            if(no_labels):
                encoded_X = []
                for words in data:
                    x = [word2idx.get(word, word2idx["<UNK>"]) for word in words]
                    encoded_X.append(x)
                return encoded_X
            else:
                encoded_X, encoded_y = [], []
                for words, tags in data:
                    if build_vocab:
                        x = [word2idx[word] for word in words]
                        y = [tag2idx[tag] for tag in tags]
                    else:
                        x = [word2idx.get(word, word2idx["<UNK>"]) for word in words]
                        y = [tag2idx.get(tag, 0) for tag in tags]  # Default PAD for unknown tags
                    encoded_X.append(x)
                    encoded_y.append(y)
                return encoded_X, encoded_y

    @staticmethod
    def _pad_batch(seqs, pad_value=0):
            max_len = max(len(s) for s in seqs)
            return [s + [pad_value] * (max_len - len(s)) for s in seqs]


class CharDataParser:
    @staticmethod
    def parse(train_path, dev_path):
        train_data = CharDataParser._load_file(train_path)
        dev_data = CharDataParser._load_file(dev_path)

        # --- Build Vocab ---
        char2idx = defaultdict(lambda: len(char2idx))
        word2idx = defaultdict(lambda: len(word2idx))
        tag2idx = defaultdict(lambda: len(tag2idx))

        # Special tokens
        PAD = "<PAD>"
        UNK = "<UNK>"
        char2idx[PAD]
        char2idx[UNK]
        word2idx[PAD]
        word2idx[UNK]
        tag2idx[PAD]

        # Encode
        train_X, train_y = CharDataParser._encode(train_data, char2idx, word2idx, tag2idx, build_vocab=True)
        dev_X, dev_y = CharDataParser._encode(dev_data, char2idx, word2idx, tag2idx, build_vocab=False)

        # Pad
        train_X_char, train_y = CharDataParser._pad_batch(train_X, train_y, char2idx[PAD], tag2idx[PAD])
        dev_X_char, dev_y = CharDataParser._pad_batch(dev_X, dev_y, char2idx[PAD], tag2idx[PAD])

        # Convert to tensors
        train_X_char = torch.tensor(train_X_char, dtype=torch.long)
        train_y = torch.tensor(train_y, dtype=torch.long)
        dev_X_char = torch.tensor(dev_X_char, dtype=torch.long)
        dev_y = torch.tensor(dev_y, dtype=torch.long)

        # Calculate vocab sizes
        char_vocab_size = len(char2idx)
        word_vocab_size = len(word2idx)
        tag_vocab_size = len(tag2idx)

        return (
            train_X_char, train_y,
            dev_X_char, dev_y,
            dict(tag2idx), {v: k for k, v in tag2idx.items()},
            dict(char2idx), {v: k for k, v in char2idx.items()},
            char_vocab_size, word_vocab_size, tag_vocab_size
        )
    @staticmethod
    def _load_file(path):
        data = []
        sent, tags = [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    word, tag = line.split()
                    sent.append(word.lower())
                    tags.append(tag)
                else:
                    if sent:
                        data.append((sent, tags))
                        sent, tags = [], []
            if sent:
                data.append((sent, tags))
        return data

    @staticmethod
    def _encode(data, char2idx, word2idx, tag2idx, build_vocab=True):
        encoded_X = []
        encoded_y = []

        for words, tags in data:
            sent_chars = []
            sent_tags = []

            for word, tag in zip(words, tags):
                # Encode word as list of char indices
                if build_vocab:
                    char_ids = [char2idx[c] for c in word]
                    _ = word2idx[word]
                    tag_id = tag2idx[tag]
                else:
                    char_ids = [char2idx.get(c, char2idx["<UNK>"]) for c in word]
                    tag_id = tag2idx.get(tag, tag2idx["<PAD>"])
                sent_chars.append(char_ids)
                sent_tags.append(tag_id)

            encoded_X.append(sent_chars)
            encoded_y.append(sent_tags)

        return encoded_X, encoded_y

    @staticmethod
    def _pad_batch(batch_char_seqs, batch_tag_seqs, char_pad_value, tag_pad_value):
        max_seq_len = max(len(seq) for seq in batch_char_seqs)
        max_word_len = max(len(word) for seq in batch_char_seqs for word in seq)

        padded_chars = []
        padded_tags = []

        for char_seq, tag_seq in zip(batch_char_seqs, batch_tag_seqs):
            padded_seq = []
            for word_chars in char_seq:
                padded_word = word_chars + [char_pad_value] * (max_word_len - len(word_chars))
                padded_seq.append(padded_word)
            # Pad sentence to max_seq_len
            pad_word = [char_pad_value] * max_word_len
            padded_seq += [pad_word] * (max_seq_len - len(padded_seq))
            padded_chars.append(padded_seq)

            padded_tags.append(tag_seq + [tag_pad_value] * (max_seq_len - len(tag_seq)))

        return padded_chars, padded_tags



if __name__ == '__main__':
    train_X_char, train_y, dev_X_char, dev_y, tag2idx, idx2tag, char2idx, idx2char = CharDataParser.parse(
        "/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/data/pos/train",
        "/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/data/pos/dev"
    )
    print('a')
    # train_X = DataParser.parse_test_pos(
    #     "/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/data/pos/train",
    #     "/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/data/pos/dev"