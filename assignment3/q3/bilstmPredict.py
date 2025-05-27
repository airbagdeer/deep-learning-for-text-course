import sys
import torch
from dataParser import DataParser

def predict(model_path, input_path, output_path, task):
    model = torch.load(model_path)
    words_and_tags_dict = model.words_and_tags_dict
    word2idx, _, tag2idx, idx2tag = words_and_tags_dict.values()
    train_data = DataParser.parse_test(input_path, word2idx)
    tags = [' '.join(_predict(model, idx2tag, X)) for X in train_data]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tags) + "\n")


def _predict(model, idx2tag, X):
        model.eval()
        X = X.to(model.device)
        with torch.no_grad():
            logits = model(X)
            preds = torch.argmax(logits, dim=-1)
        return [idx2tag.get(p, "<UNK>") for p in preds.cpu()]


if __name__ == '__main__':
    _, option, model_path, input_path, output_path, task = sys.argv
    predict(model_path, input_path, output_path, task)