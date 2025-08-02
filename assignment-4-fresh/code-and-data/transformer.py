from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False, return_attention=False):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len, return_attention)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals
        self.return_attention = return_attention

    def forward(self, inputs):
        x = inputs
        if self.with_residuals:
            if self.return_attention:
                attn_out, attentions = self.causal_attention(self.layer_norm_1(x))
                x = x + attn_out
                x = x + self.mlp(self.layer_norm_2(x))
                return x, attentions
            else:
                x = x + self.causal_attention(self.layer_norm_1(x))
                x = x + self.mlp(self.layer_norm_2(x))
        else:
            x = self.layer_norm_1(x)
            if self.return_attention:
                x, attentions = self.causal_attention(x)
            else:
                x = self.causal_attention(x)
            x = self.layer_norm_2(x)
            x = self.mlp(x)
        if self.return_attention:
            return x, attentions
        return x

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x):
        b, n = x.size()
        positions = torch.arange(0, n, dtype=torch.long, device=x.device)
        tok_embeddings = self.token_embeddings(x)
        pos_embeddings = self.position_embeddings(positions)
        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            return_attention: bool = False,
            device=None
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals, return_attention) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len
        self.return_attention = return_attention
        self.device = device if device is not None else torch.device("cpu")

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)
        if self.return_attention:
            all_attentions = []
            for layer in self.layers:
                x, attentions = layer(x)
                all_attentions.append(attentions)
            x = self.layer_norm(x)
            logits = self.word_prediction(x)
            return logits, all_attentions
        else:
            for layer in self.layers:
                x = layer(x)
            x = self.layer_norm(x)
            logits = self.word_prediction(x)
            return logits

    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32).to(self.device))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token.item())
                feed_to_lm.append(sampled_token.item())
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32).to(self.device))
                logits_for_last_token = logits[0][-1]

                # Apply temperature
                if temperature == 0.0:
                    # If temperature is 0, do greedy sampling
                    sampled_token = torch.argmax(logits_for_last_token).unsqueeze(0)
                else:
                    logits_for_last_token = logits_for_last_token / temperature

                    # Apply top-k filtering
                    if topK > 0:
                        top_k_values, top_k_indices = torch.topk(logits_for_last_token, k=topK, dim=-1)
                        # Create a mask for non-top-k elements
                        mask = torch.full_like(logits_for_last_token, float('-inf'))
                        mask[top_k_indices] = logits_for_last_token[top_k_indices]
                        logits_for_last_token = mask

                    distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                    sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)

                generated.append(sampled_token.item())
                feed_to_lm.append(sampled_token.item())
        return generated

    def init_weights(self):
        # initialize weights
        # TODO implement initialization logic for embeddings and linear layers.
        # The code break down the parameters by type (layer-norm, linear, embedding),
        # but can also condition on individual names, for example by checking pn.endswith(...).
        for pn, p in self.named_parameters():
            if isinstance(p, nn.LayerNorm):
                torch.nn.init.zeros_(p.bias)
                torch.nn.init.ones_(p.weight)
            elif isinstance(p, nn.Linear):
                torch.nn.init.normal_(p.weight, mean=0.0, std=0.02)
                if p.bias is not None:
                    torch.nn.init.zeros_(p.bias)
            elif isinstance(p, nn.Embedding):
                torch.nn.init.normal_(p.weight, mean=0.0, std=0.02)


    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                # print(f"feed_to_lm: {feed_to_lm}")
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32).to(self.device))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token.item())
                feed_to_lm.append(sampled_token.item())
        return generated
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32))
                logits_for_last_token = logits[0][-1]

                # Apply temperature
                if temperature == 0.0:
                    # If temperature is 0, do greedy sampling
                    sampled_token = torch.argmax(logits_for_last_token).unsqueeze(0)
                else:
                    logits_for_last_token = logits_for_last_token / temperature

                    # Apply top-k filtering
                    if topK > 0:
                        top_k_values, top_k_indices = torch.topk(logits_for_last_token, k=topK, dim=-1)
                        # Create a mask for non-top-k elements
                        mask = torch.full_like(logits_for_last_token, float('-inf'))
                        mask[top_k_indices] = logits_for_last_token[top_k_indices]
                        logits_for_last_token = mask

                    distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                    sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)

                generated.append(sampled_token.item())
                feed_to_lm.append(sampled_token.item())
        return generated

