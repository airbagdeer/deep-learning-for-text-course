from __future__ import annotations
import torch
import os

if __name__ == '__main__':
    import torch
    from torch import nn
    from torch import optim
    from transformer import TransformerLM
    import data
    import lm

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    seq_len = 128
    batch_size = 64
    data_path = "data/"
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0

    num_batches_to_train = 50000
    if data_path == "heb-data/":
        checkpoint_dir = "heb-checkpoints"
    else:
        checkpoint_dir = "eng-checkpoints/"
    checkpoint_interval = 1000 # Save checkpoint every 1000 batches

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer, tokenized_data = data.load_data(data_path)
    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model: torch.nn.Module = TransformerLM(
            n_layers,
            n_heads,
            embed_size,
            seq_len,
            tokenizer.vocab_size(),
            mlp_hidden_size,
            with_residuals = True,
            device=device
        ).to(device) # Move model to device

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])

    model.train()
    
    num_batches = 0
    training_complete = False
    while not training_complete:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train:
                training_complete = True
                break
            num_batches = num_batches + 1

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x = batch_x.to(device) # Move input to device
            batch_y = batch_y.to(device) # Move labels to device

            logits = model(batch_x)

            loss = lm.compute_loss(logits, batch_y)

            # parameters update
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                if num_batches % 100 == 0:
                    for _ in range(1):
                        model.eval()
                        # Move input to device for sampling
                        if data_path == "heb-data/":
                            sampled = tokenizer.detokenize(model.sample_continuation(tokenizer.tokenize("שלום"), 500))
                        elif data_path == "data/":
                            sampled = tokenizer.detokenize(model.sample_continuation(tokenizer.tokenize("Hello"), 500))
                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                    print("")
            
            # Checkpointing
            if num_batches % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint_{num_batches}.pt")
                torch.save({
                    'num_batches': num_batches,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
