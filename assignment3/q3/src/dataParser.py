import torch
from collections import defaultdict


import torch
from collections import defaultdict

class DataParser:
    @staticmethod
    def parse_test(test_path, word2idx):
        test_data = DataParser._load_file(test_path, no_labels=True)
        test_X = DataParser._encode(test_data, word2idx, build_vocab=True, no_labels=True)
        test_X = DataParser._pad_batch(test_X, pad_value=word2idx["<PAD>"])
        test_X = torch.tensor(test_X, dtype=torch.long)
        return test_X

    @staticmethod
    def parse(train_path, dev_path=None):
        train_data = DataParser._load_file(train_path)

        dev_data = DataParser._load_file(dev_path) if dev_path else None

        word2idx = defaultdict(lambda: len(word2idx))
        word2idx["<PAD>"]
        word2idx["<UNK>"]

        tag2idx = defaultdict(lambda: len(tag2idx))
        tag2idx["<PAD>"]  # 0

        train_X, train_y = DataParser._encode(train_data, word2idx, tag2idx, build_vocab=True)

        if dev_data:
            dev_X, dev_y = DataParser._encode(dev_data, word2idx, tag2idx, build_vocab=False)
        else:
            dev_X, dev_y = None, None

        train_X = DataParser._pad_batch(train_X, pad_value=word2idx["<PAD>"])
        train_y = DataParser._pad_batch(train_y, pad_value=tag2idx["<PAD>"])

        train_X = torch.tensor(train_X, dtype=torch.long)
        train_y = torch.tensor(train_y, dtype=torch.long)

        if dev_X is not None and dev_y is not None:
            dev_X = DataParser._pad_batch(dev_X, pad_value=word2idx["<PAD>"])
            dev_y = DataParser._pad_batch(dev_y, pad_value=tag2idx["<PAD>"])
            dev_X = torch.tensor(dev_X, dtype=torch.long)
            dev_y = torch.tensor(dev_y, dtype=torch.long)

        idx2tag = {v: k for k, v in tag2idx.items()}
        idx2word = {v: k for k, v in word2idx.items()}

        return train_X, train_y, dev_X, dev_y, dict(tag2idx), idx2tag, dict(word2idx), idx2word, len(word2idx)

    @staticmethod
    def _load_file(path, no_labels=False):
        if no_labels:
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
        if no_labels:
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
                    y = [tag2idx.get(tag, 0) for tag in tags]
                encoded_X.append(x)
                encoded_y.append(y)
            return encoded_X, encoded_y

    @staticmethod
    def _pad_batch(seqs, pad_value=0):
        max_len = max(len(s) for s in seqs)
        return [s + [pad_value] * (max_len - len(s)) for s in seqs]


class CharDataParser:
    @staticmethod
    def parse_test(test_path, char2idx):
        test_data = CharDataParser._load_file(test_path, no_labels=True)
        train_X = CharDataParser._encode(test_data, char2idx=char2idx, build_vocab=True, no_labels=True)
        train_X = CharDataParser._pad_batch_char_only(train_X, char_pad_value=char2idx["<PAD>"])
        train_X = torch.tensor(train_X, dtype=torch.long)
        return train_X

    @staticmethod
    def parse(train_path, dev_path=None):
        train_data = CharDataParser._load_file(train_path)
        dev_data = CharDataParser._load_file(dev_path) if dev_path else None

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

        train_X, train_y = CharDataParser._encode(train_data, char2idx, word2idx, tag2idx, build_vocab=True)

        if dev_data is not None:
            dev_X, dev_y = CharDataParser._encode(dev_data, char2idx, word2idx, tag2idx, build_vocab=False)
        else:
            dev_X = dev_y = None

        train_X_char, train_y = CharDataParser._pad_batch(train_X, train_y, char2idx[PAD], tag2idx[PAD])
        train_X_char = torch.tensor(train_X_char, dtype=torch.long)
        train_y = torch.tensor(train_y, dtype=torch.long)

        if dev_X is not None and dev_y is not None:
            dev_X_char, dev_y = CharDataParser._pad_batch(dev_X, dev_y, char2idx[PAD], tag2idx[PAD])
            dev_X_char = torch.tensor(dev_X_char, dtype=torch.long)
            dev_y = torch.tensor(dev_y, dtype=torch.long)
        else:
            dev_X_char = dev_y = None

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
    def _load_file(path, no_labels=False):
        if no_labels:
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
    def _encode(data, char2idx=None, word2idx=None, tag2idx=None, build_vocab=True, no_labels=False):
        if no_labels:
            encoded_X = []
            for words in data:
                sent_chars = []
                for word in words:
                    char_ids = [char2idx.get(c, char2idx["<UNK>"]) for c in word]
                    sent_chars.append(char_ids)
                encoded_X.append(sent_chars)
            return encoded_X
        else:
            encoded_X = []
            encoded_y = []
            for words, tags in data:
                sent_chars = []
                sent_tags = []
                for word, tag in zip(words, tags):
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
            pad_word = [char_pad_value] * max_word_len
            padded_seq += [pad_word] * (max_seq_len - len(padded_seq))
            padded_chars.append(padded_seq)

            padded_tags.append(tag_seq + [tag_pad_value] * (max_seq_len - len(tag_seq)))

        return padded_chars, padded_tags

    @staticmethod
    def _pad_batch_char_only(batch_char_seqs, char_pad_value):
        max_seq_len = max(len(seq) for seq in batch_char_seqs)
        max_word_len = max(len(word) for seq in batch_char_seqs for word in seq)

        padded_chars = []
        for char_seq in batch_char_seqs:
            padded_seq = []
            for word_chars in char_seq:
                padded_word = word_chars + [char_pad_value] * (max_word_len - len(word_chars))
                padded_seq.append(padded_word)
            pad_word = [char_pad_value] * max_word_len
            padded_seq += [pad_word] * (max_seq_len - len(padded_seq))
            padded_chars.append(padded_seq)

        return padded_chars


class CombinedParser:
    @staticmethod
    def parse_test(test_path, char2idx, word2idx):
        train_x = DataParser.parse_test(test_path, word2idx)
        char_train_x = CharDataParser.parse_test(test_path, char2idx)
        return train_x, char_train_x


    @staticmethod
    def parse(train_path, dev_path):
            train_X, train_y, dev_X, dev_y, tag2idx, idx2tag, word2idx, idx2word, vocab_size = DataParser.parse(train_path, dev_path)
            char_train_X, _, char_dev_X, _, _, _, char2idx, idx2char, char_vocab_size, _, _ = CharDataParser.parse(train_path, dev_path)
            return (
                train_X,
                char_train_X,
                train_y,
                dev_X,
                char_dev_X,
                dev_y,
                word2idx,
                idx2word,
                char2idx,
                idx2char,
                tag2idx,
                idx2tag,
                vocab_size,
                char_vocab_size
            )

class PrefixSuffixParser:
    @staticmethod
    def parse_test(test_path, word2idx, prefix2idx, suffix2idx):
        test_data = DataParser._load_file(test_path, no_labels=True)

        test_X = []
        prefix_X = []
        suffix_X = []

        for sentence in test_data:
            x = [word2idx.get(word.lower(), word2idx["<UNK>"]) for word in sentence]
            p = [prefix2idx.get(word.lower()[:3], prefix2idx["<PAD>"]) for word in sentence]
            s = [suffix2idx.get(word.lower()[-3:], suffix2idx["<PAD>"]) for word in sentence]

            test_X.append(x)
            prefix_X.append(p)
            suffix_X.append(s)

        test_X = DataParser._pad_batch(test_X, pad_value=word2idx["<PAD>"])
        prefix_X = DataParser._pad_batch(prefix_X, pad_value=prefix2idx["<PAD>"])
        suffix_X = DataParser._pad_batch(suffix_X, pad_value=suffix2idx["<PAD>"])

        return (
            torch.tensor(test_X, dtype=torch.long),
            torch.tensor(prefix_X, dtype=torch.long),
            torch.tensor(suffix_X, dtype=torch.long),
        )

    @staticmethod
    def parse(train_path, dev_path=None):
        train_data = DataParser._load_file(train_path)
        dev_data = None
        if dev_path is not None:
            dev_data = DataParser._load_file(dev_path)

        # --- Vocab and Tag mappings ---
        word2idx = defaultdict(lambda: len(word2idx))
        word2idx["<PAD>"]
        word2idx["<UNK>"]

        tag2idx = defaultdict(lambda: len(tag2idx))
        tag2idx["<PAD>"]

        prefix2idx = defaultdict(lambda: len(prefix2idx))
        prefix2idx["<PAD>"]

        suffix2idx = defaultdict(lambda: len(suffix2idx))
        suffix2idx["<PAD>"]

        def encode_with_affixes(data, build_vocab):
            X, y, prefix_X, suffix_X = [], [], [], []

            for words, tags in data:
                if build_vocab:
                    x = [word2idx[word] for word in words]
                    y_ = [tag2idx[tag] for tag in tags]
                    prefixes = [prefix2idx[word[:3]] for word in words]
                    suffixes = [suffix2idx[word[-3:]] for word in words]
                else:
                    x = [word2idx.get(word, word2idx["<UNK>"]) for word in words]
                    y_ = [tag2idx.get(tag, 0) for tag in tags]
                    prefixes = [prefix2idx.get(word[:3], prefix2idx["<PAD>"]) for word in words]
                    suffixes = [suffix2idx.get(word[-3:], suffix2idx["<PAD>"]) for word in words]

                X.append(x)
                y.append(y_)
                prefix_X.append(prefixes)
                suffix_X.append(suffixes)

            return X, y, prefix_X, suffix_X

        train_X, train_y, prefix_train_X, suffix_train_X = encode_with_affixes(train_data, build_vocab=True)

        if dev_data is not None:
            dev_X, dev_y, prefix_dev_X, suffix_dev_X = encode_with_affixes(dev_data, build_vocab=False)
        else:
            dev_X = dev_y = prefix_dev_X = suffix_dev_X = None

        pad_word = word2idx["<PAD>"]
        pad_tag = tag2idx["<PAD>"]
        pad_prefix = prefix2idx["<PAD>"]
        pad_suffix = suffix2idx["<PAD>"]

        def pad(*args, pad_values):
            return [DataParser._pad_batch(lst, pad_value=v) for lst, v in zip(args, pad_values)]

        train_X, train_y, prefix_train_X, suffix_train_X = pad(
            train_X, train_y, prefix_train_X, suffix_train_X,
            pad_values=[pad_word, pad_tag, pad_prefix, pad_suffix]
        )

        if dev_data is not None:
            dev_X, dev_y, prefix_dev_X, suffix_dev_X = pad(
                dev_X, dev_y, prefix_dev_X, suffix_dev_X,
                pad_values=[pad_word, pad_tag, pad_prefix, pad_suffix]
            )

        train_X = torch.tensor(train_X, dtype=torch.long)
        train_y = torch.tensor(train_y, dtype=torch.long)
        prefix_train_X = torch.tensor(prefix_train_X, dtype=torch.long)
        suffix_train_X = torch.tensor(suffix_train_X, dtype=torch.long)

        if dev_data is not None:
            dev_X = torch.tensor(dev_X, dtype=torch.long)
            dev_y = torch.tensor(dev_y, dtype=torch.long)
            prefix_dev_X = torch.tensor(prefix_dev_X, dtype=torch.long)
            suffix_dev_X = torch.tensor(suffix_dev_X, dtype=torch.long)
        else:
            dev_X = dev_y = prefix_dev_X = suffix_dev_X = None

        idx2tag = {v: k for k, v in tag2idx.items()}
        idx2word = {v: k for k, v in word2idx.items()}
        idx2prefix = {v: k for k, v in prefix2idx.items()}
        idx2suffix = {v: k for k, v in suffix2idx.items()}

        return (
            train_X, prefix_train_X, suffix_train_X, train_y,
            dev_X, prefix_dev_X, suffix_dev_X, dev_y,
            dict(tag2idx), idx2tag,
            dict(word2idx), idx2word,
            dict(prefix2idx), idx2prefix,
            dict(suffix2idx), idx2suffix,
            len(word2idx), len(prefix2idx), len(suffix2idx)
        )