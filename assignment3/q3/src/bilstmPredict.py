import sys
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader, TensorDataset
from dataParser import DataParser, CharDataParser

def predict(model_path, input_path, output_path, option):
    model = torch.load(model_path)
    words_and_tags_dict = model.words_and_tags_dict
    saved_data = words_and_tags_dict
    test_data = []
    if(option == 'a'):
        test_data = DataParser.parse_test(input_path, saved_data['word2idx'])
    elif(option == 'b'):
        test_data = CharDataParser.parse_test(input_path, saved_data['char2idx'])
    all_predicts = batch_predict(model=model,idx2tag=saved_data['idx2tag'], X_tensor=test_data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(format_tags(all_predicts))

def batch_predict(model, idx2tag, X_tensor, batch_size=32):
    model.eval()
    device = next(model.parameters()).device

    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)  # [batch, seq_len, num_classes]
            preds = torch.argmax(logits, dim=-1)  # [batch, seq_len]
            for sent in preds:
                all_preds.append([idx2tag.get(p.item(), "<UNK>") for p in sent])

    return all_preds

def format_tags(tags):
    lines = []
    for inner_tags in tags:
        lines.extend(inner_tags)
        lines.append("")  # Empty line between inner lists
    return "\n".join(lines)


if __name__ == '__main__':
    _, option, model_path, input_path, output_path, task = sys.argv
    predict(model_path, input_path, output_path, option)