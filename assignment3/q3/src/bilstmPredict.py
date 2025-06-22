import sys
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader, TensorDataset
from dataParser import DataParser, CharDataParser, PrefixSuffixParser, CombinedParser
from bilestmSubWordEmbeddingTagger import PrefixSuffixDataset, PrefixSuffixBiLSTMTagger
from bilstmCharWordEmbeddingTagger import WordCharTagDataset, CharWordBiLSTMTagger
from bilstmWordEmbeddingTagger import BiLSTMWordEmbeddingTagger
from bilstmRnnEmbeddingTagger import CharLSTMCellWordBiLSTMCellTagger
import argparse

def predict(model_path, input_path, output_path, option):
    model = torch.load(model_path)
    words_and_tags_dict = model.words_and_tags_dict
    saved_data = words_and_tags_dict
    predictions = []
    test_data = []
    if(option == 'a'):
        test_data = DataParser.parse_test(input_path, saved_data['word2idx'])
        predictions = batch_predict_word_representation(model=model,idx2tag=saved_data['idx2tag'], words=test_data)
    elif(option == 'b'):
        test_data = CharDataParser.parse_test(input_path, saved_data['char2idx'])
        predictions = batch_predict_char_representation(model=model,idx2tag=saved_data['idx2tag'], chars=test_data)
    elif(option == 'c'):
        words_data, prefix_data, suffix_data = PrefixSuffixParser.parse_test(input_path, saved_data['word2idx'], saved_data['prefix2idx'], saved_data['suffix2idx'])
        predictions = batch_prefiction_subword_representation(model=model, words_data=words_data, prefix_data=prefix_data, suffix_data=suffix_data ,idx2tag=saved_data['idx2tag'])
    elif(option == 'd'):
        words_data, chars_data = CombinedParser.parse_test(input_path, saved_data['char2idx'], saved_data['word2idx'])
        predictions = batch_prefiction_combined_representation(model=model, words_data=words_data, chars_data=chars_data, idx2tag=saved_data['idx2tag'])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(format_tags(predictions))


def batch_prefiction_subword_representation(model, words_data, prefix_data, suffix_data, idx2tag, batch_size=32):
    model.eval()
    device = next(model.parameters()).device

    dataset = PrefixSuffixDataset(word_ids=words_data, prefix_ids=prefix_data, suffix_ids=suffix_data)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []

    with torch.no_grad():
        for word_batch, prefix_batch, suffix_batch in loader:
            word_batch = word_batch.to(device)
            prefix_batch = prefix_batch.to(device)
            suffix_batch = suffix_batch.to(device)
            logits = model(word_batch, prefix_batch, suffix_batch)
            preds = torch.argmax(logits, dim=-1)
            
            mask = word_batch != 0
            for sent_preds, sent_mask in zip(preds, mask):
                tags = [
                    idx2tag.get(p.item(), "<UNK>")
                    for p, m in zip(sent_preds, sent_mask)
                    if m.item()
                ]
                all_preds.append(tags)

    return all_preds


def batch_prefiction_combined_representation(model, words_data, chars_data, idx2tag, batch_size=32):
    model.eval()
    device = next(model.parameters()).device

    dataset = WordCharTagDataset(word_ids=words_data, char_ids=chars_data)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []

    with torch.no_grad():
        for word_batch, char_batch in loader:
            word_batch = word_batch.to(device)
            char_batch = char_batch.to(device)

            logits = model(word_batch, char_batch)
            preds = torch.argmax(logits, dim=-1)
            mask = word_batch != 0

            for sent_preds, sent_mask in zip(preds, mask):
                tags = [
                    idx2tag.get(p.item(), "<UNK>")
                    for p, m in zip(sent_preds, sent_mask)
                    if m.item()
                ]
                all_preds.append(tags)

    return all_preds


def batch_predict_word_representation(model, idx2tag, words, batch_size=32):
    model.eval()
    device = next(model.parameters()).device

    dataset = TensorDataset(words)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)  
            preds = torch.argmax(logits, dim=-1)
            for sent in preds:
                all_preds.append([idx2tag.get(p.item(), "<UNK>") for p in sent])

    return all_preds

def batch_predict_char_representation(model, idx2tag, chars, batch_size=32):
    model.eval()
    device = next(model.parameters()).device

    dataset = TensorDataset(chars)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=-1)

            mask = xb.any(dim=-1)

            for sent_preds, sent_mask in zip(preds, mask):
                tags = [
                    idx2tag.get(p.item(), "<UNK>")
                    for p, m in zip(sent_preds, sent_mask)
                    if m.item()
                ]
                all_preds.append(tags)

    return all_preds

def format_tags(tags):
    lines = []
    for inner_tags in tags:
        lines.extend(inner_tags)
        lines.append("")
    return "\n".join(lines)


def get_args_parser():
    arg_parser = argparse.ArgumentParser(description="Run training and evaluation.")
    arg_parser.add_argument("option", help="Word representation option")
    arg_parser.add_argument("model_path", help="Path to the model file")
    arg_parser.add_argument("input_path", help="Path to the test file")

    arg_parser.add_argument("--output_path", help="Path to the output file (optional)", default='./output.txt')

    return arg_parser

if __name__ == '__main__':


    args = get_args_parser().parse_args()

    option, model_path, input_path, output_path = args.option, args.model_path, args.input_path, args.output_path
    predict(model_path, input_path, output_path, option)