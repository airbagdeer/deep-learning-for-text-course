import torch
import sys
from bilstmWordEmbeddingTagger import BiLSTMWordEmbeddingTagger, BiLstmWordEmbeddingSequenceTrainer
from bilstmRnnEmbeddingTagger import BiLstmCharLstmTrainer, CharLSTMCellWordBiLSTMCellTagger
from dataParser import DataParser, CharDataParser
# --- EXAMPLE USAGE ---
if __name__ == "__main__":

    _, option, train_path, model_path, dev_path, task, accuracy_logging_file_path = sys.argv

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

    elif(option == 'b'):
        train_X_char, train_y, dev_X_char, dev_y, tag2idx, idx2tag, char2idx, idx2char, char_vocab_size, word_vocab_size, tag_vocab_size = CharDataParser.parse(train_path, dev_path)
        
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

    trainer.train(train_X_char, train_y, dev_X_char, dev_y, tag2idx=tag2idx, task_type=task, epochs=epochs, batch_size=batch_size, accuracy_logging_file_path=accuracy_logging_file_path)
    torch.save(model, model_path)

