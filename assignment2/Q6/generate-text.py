from langauge_model import CharLanguageModel, emphsample, build_vocab
import json
import os
import torch


os.chdir(os.path.dirname(os.path.abspath(__file__)))


with open("config.json", "r") as f:
    config = json.load(f)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHT_PATH = "./model_weights-50.pth"

SEQ_LEN = config["seq_len"]
BATCH_SIZE = config["batch_size"]
EMBEDDING_DIM = config["embedding_dim"]
HIDDEN_DIM_1 = config["hidden_dim_1"]
HIDDEN_DIM_2 = config["hidden_dim_2"]
EPOCHS = config["epochs"] 
PAD_TOKEN = config["pad_token"]
LEARNING_RATE = config["learning_rate"]
CORPUS_PATH = config["corpus_path"]
SAMPLE_EVERY_X_MINTUES = config.get("sample_every_x_mintues", None)
SAMPLE_PREFIX = config.get("sample_prefix", "")
SAMPLE_LENGTH = config.get("sample_length", 100)
CONV_OUT_CHANNELS = config["conv_out_channels"]
KERNEL_SIZE = config["kernel_size"]


with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    text = f.read()

vocab, stoi, itos = build_vocab(text)


def predict_text(model, prefix, n_chars, stoi, itos, device):
    model.eval()
    generated = list(prefix)
    input_seq = torch.tensor([stoi[c] for c in prefix], dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(n_chars - len(prefix)):
        with torch.no_grad():
            output = model(input_seq)  # shape: (1, seq_len, vocab_size)
            last_logits = output[0, -1]  # get the logits for the last time step

            # Apply softmax to get probabilities
            probs = torch.softmax(last_logits, dim=0)

            # Use torch.multinomial for sampling (ensure it's 1D)
            next_char_idx = torch.multinomial(probs.unsqueeze(0), num_samples=1).item()  # unsqueeze to make it 2D
            next_char = itos[next_char_idx]

            generated.append(next_char)

            # Update the input sequence with the new character
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]], device=device)], dim=1)

    return ''.join(generated)


def generate_samples(path, prefixes, n_chars):
    model = CharLanguageModel(len(vocab), EMBEDDING_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, SEQ_LEN, CONV_OUT_CHANNELS, KERNEL_SIZE)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=torch.device('cpu')))


    samples = [predict_text(model, prefix, n_chars, stoi, itos, DEVICE) for prefix in prefixes]
    save_to_file(path, samples)


def save_to_file(path, samples):
    content = "     \n".join(samples)
    with open(path, "w") as f:
        f.write(content)

if __name__ == '__main__':
    generate_samples('./samples.txt', ['Hello, The World'], 100)