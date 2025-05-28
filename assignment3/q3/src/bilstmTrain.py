import torch
import sys
from bilstmWordEmbeddingTagger import BiLSTMWordEmbeddingTagger, BiLstmWordEmbeddingSequenceTrainer
from bilstmRnnEmbeddingTagger import BiLstmCharLstmTrainer, CharLSTMCellWordBiLSTMCellTagger
from bilstmCharWordEmbeddingTagger import CharWordBiLSTMTagger, CharWordBilstmTrainer
from bilestmSubWordEmbeddingTagger import PrefixSuffixBiLSTMTagger, PrefixSuffixTrainer
from dataParser import DataParser, CharDataParser, CombinedParser, PrefixSuffixParser
import argparse

def main():

    parser = get_parser()
    args = parser.parse_args()


    option, train_path, model_path, dev_path, task, accuracy_logging_file_path =  args.option, args.train_path, args.model_path, args.dev_path, args.task, args.accuracy_logging_file_path

    # Initialize and train
    model, trainer  = None, None
    if(option == 'a'):
        train_X, train_y, dev_X, dev_y, tag2idx, idx2tag, word2idx, idx2word, vocab_size = DataParser.parse(train_path, dev_path)
        words_and_tags_dict = {
        'word2idx':word2idx,
        'idx2word': idx2word,
        'tag2idx': tag2idx,
        'idx2tag': idx2tag
        }

        embed_dim = 32
        hidden_dim = 64
        num_classes = len(idx2tag)
        pad_idx = 0
        epochs = 5
        batch_size = 32
        lr = 0.001

        model = BiLSTMWordEmbeddingTagger(vocab_size, embed_dim, hidden_dim, num_classes, words_and_tags_dict=words_and_tags_dict, pad_idx=pad_idx)
        trainer = BiLstmWordEmbeddingSequenceTrainer(model, lr=lr)
        trainer.train(train_X, train_y, dev_X, dev_y, tag2idx=tag2idx, task_type=task, epochs=epochs, batch_size=batch_size, accuracy_logging_file_path=accuracy_logging_file_path)

    elif(option == 'b'):
        train_X, train_y, dev_X, dev_y, tag2idx, idx2tag, char2idx, idx2char, char_vocab_size, word_vocab_size, tag_vocab_size = CharDataParser.parse(train_path, dev_path)
        
        words_and_tags_dict = {
        'char2idx':char2idx,
        'idx2char': idx2char,
        'tag2idx': tag2idx,
        'idx2tag': idx2tag
        }

        char_embed_dim = 16
        char_hidden_dim = 64
        word_hidden_dim = 64
        num_classes = len(idx2tag) 
        pad_idx = 0
        epochs = 5
        batch_size = 16
        lr = 0.001

        model = CharLSTMCellWordBiLSTMCellTagger(char_vocab_size, char_embed_dim, char_hidden_dim, word_hidden_dim, num_classes, words_and_tags_dict=words_and_tags_dict, pad_idx=pad_idx)
        trainer = BiLstmCharLstmTrainer(model, lr=lr)
        trainer.train(train_X, train_y, dev_X, dev_y, tag2idx=tag2idx, task_type=task, epochs=epochs, batch_size=batch_size, accuracy_logging_file_path=accuracy_logging_file_path)

    
    elif(option == 'd'):
        train_X, char_train_X, train_y, dev_X, char_dev_X, dev_y, word2idx, idx2word, char2idx, idx2char, tag2idx, idx2tag, vocab_size, char_vocab_size = CombinedParser.parse(train_path, dev_path)
        words_and_tags_dict = {
        'char2idx':char2idx,
        'idx2char': idx2char,
        'word2idx':word2idx,
        'idx2word': idx2word,
        'tag2idx': tag2idx,
        'idx2tag': idx2tag
        }

        word_embed_dim = 32
        char_embed_dim = 16
        char_hidden_dim = 32
        word_hidden_dim = 64
        num_classes = len(idx2tag) 
        pad_idx = 0
        epochs = 5
        batch_size = 16
        lr = 0.003

        model = CharWordBiLSTMTagger(vocab_size, word_embed_dim,
                 char_vocab_size, char_embed_dim, char_hidden_dim,
                 word_hidden_dim, num_classes, words_and_tags_dict=words_and_tags_dict, pad_idx=0)
        trainer = CharWordBilstmTrainer(model, lr=lr)
        trainer.train(train_X, char_train_X, train_y, dev_X, char_dev_X, dev_y, tag2idx=tag2idx, task_type=task, epochs=epochs, batch_size=batch_size, accuracy_logging_file_path=accuracy_logging_file_path)
    
    elif(option == 'c'):
        (
        train_X, prefix_train_X, suffix_train_X, train_y,
        dev_X, prefix_dev_X, suffix_dev_X, dev_y,
        tag2idx, idx2tag,
        word2idx, idx2word,
        prefix2idx, idx2prefix,
        suffix2idx, idx2suffix,
        vocab_size, prefix_vocab_size, suffix_vocab_size
        ) = PrefixSuffixParser.parse(train_path, dev_path)

        words_and_tags_dict = {
        'idx2word': idx2word,
        'word2ix': word2idx,     
        'tag2idx': tag2idx,
        'idx2tag': idx2tag,
        'prefix2idx': prefix2idx,
        'idx2prefix': idx2prefix,
        'suffix2idx': suffix2idx,
        'idx2suffix': idx2suffix,
        }

        embed_dim = 32
        word_hidden_dim = 64
        num_classes = len(idx2tag) 
        pad_idx = 0
        epochs = 5
        batch_size = 16
        lr = 0.003

        model = PrefixSuffixBiLSTMTagger(
                embed_dim,
                vocab_size,
                prefix_vocab_size,
                suffix_vocab_size,
                word_hidden_dim, num_classes, words_and_tags_dict=words_and_tags_dict, pad_idx=0)
        trainer = PrefixSuffixTrainer(model, lr=lr)
        trainer.train(train_X, prefix_train_X, suffix_train_X, train_y,
              X_dev_words=dev_X, X_dev_prefixes=prefix_dev_X, X_dev_suffixes=suffix_dev_X, y_dev=dev_y,
              tag2idx=tag2idx, task_type=task,
              batch_size=batch_size, epochs=epochs, accuracy_logging_file_path=accuracy_logging_file_path)
    else:
        print(f'Not a valid option: {option}')
        return

    torch.save(model, model_path)


def get_parser():
    arg_parser = argparse.ArgumentParser(description="Run training and evaluation.")
    # Positional arguments (must be given in order)
    arg_parser.add_argument("option", help="Word representation option")
    arg_parser.add_argument("train_path", help="Path to the test file")
    arg_parser.add_argument("model_path", help="Path to the test file")

    arg_parser.add_argument("--dev_path", help="Path to the dev file (optional)", default=None)
    arg_parser.add_argument("--task", help="Ner/Pos", default=None)
    arg_parser.add_argument("--accuracy_logging_file_path", help="Path to logging file path (optional)", default=None)

    return arg_parser


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    main()

