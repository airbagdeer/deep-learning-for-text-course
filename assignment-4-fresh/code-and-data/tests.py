import torch
import attention
import torch.nn.functional as F
from transformer import TransformerLM

def test_attention_scores():
    # fill in values for the a, b and expected_output tensor.
    a = torch.tensor([[[1., 0.], [0., 1.]]])
    b = torch.tensor([[[1., 0.], [0., 1.]]])
    expected_output = torch.tensor([[[0.7071, 0.0000], [0.0000, 0.7071]]])

    A = attention.attention_scores(a, b)
    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(A, expected_output)

def test_kqv():
    x = torch.randn(1, 10, 6)
    linear = attention.create_kqv_matrix(6, n_heads=3)
    k, q, v = attention.kqv(x, linear)
    assert k.size() == (1, 10, 2)
    assert q.size() == (1, 10, 2)
    assert v.size() == (1, 10, 2)

def test_create_causal_mask():
    mask = attention.create_causal_mask(embed_dim=6, n_heads=3, max_context_len=5)
    expected_mask = torch.tensor([[[1., 0., 0., 0., 0.],
                                   [1., 1., 0., 0., 0.],
                                   [1., 1., 1., 0., 0.],
                                   [1., 1., 1., 1., 0.],
                                   [1., 1., 1., 1., 1.]]])
    assert torch.allclose(mask, expected_mask)
    assert mask.size() == (1, 5, 5)

def test_self_attention():
    v = torch.randn(1, 3, 4)
    A = torch.randn(1, 3, 3)
    mask = torch.tensor([[[1., 0., 0.],
                          [1., 1., 0.],
                          [1., 1., 1.]]])
    sa = attention.self_attention(v, A, mask)
    assert sa.size() == v.size()

    # Test without mask
    sa_no_mask = attention.self_attention(v, A)
    assert sa_no_mask.size() == v.size()

def test_multi_head_attention_layer():
    x = torch.randn(1, 10, 6)
    kqv_matrices = [attention.create_kqv_matrix(6, n_heads=3) for _ in range(3)]
    mask = attention.create_causal_mask(embed_dim=6, n_heads=3, max_context_len=10)
    sa = attention.multi_head_attention_layer(x, kqv_matrices, mask)
    assert sa.size() == x.size()

def test_transformer_lm():
    lm = TransformerLM(n_layers=2, n_heads=2, embed_size=4, max_context_len=10, vocab_size=100, mlp_hidden_size=8, with_residuals=True)
    inputs = torch.randint(0, 100, (1, 5))
    outputs = lm(inputs)
    assert outputs.size() == (1, 5, 100)

def test_better_sample_continuation():
    # Create a dummy TransformerLM instance for testing
    vocab_size = 10
    max_context_len = 5
    lm = TransformerLM(n_layers=1, n_heads=1, embed_size=4, max_context_len=max_context_len, vocab_size=vocab_size, mlp_hidden_size=8, with_residuals=True)

    # Test with temperature = 0 (greedy sampling)
    prefix = [1, 2]
    max_tokens = 3
    temperature = 0.0
    topK = 0
    generated_greedy = lm.better_sample_continuation(prefix, max_tokens, temperature, topK)
    assert len(generated_greedy) == max_tokens

    # Test with temperature > 0 and topK > 0
    prefix = [1, 2]
    max_tokens = 3
    temperature = 1.0
    topK = 3
    generated_sampled = lm.better_sample_continuation(prefix, max_tokens, temperature, topK)
    assert len(generated_sampled) == max_tokens

if __name__ == '__main__':
    test_attention_scores()
    test_kqv()
    test_create_causal_mask()
    test_self_attention()
    test_multi_head_attention_layer()
    test_transformer_lm()
    test_better_sample_continuation()