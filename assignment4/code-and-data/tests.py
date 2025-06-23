import torch
import attention
from attention import create_kqv_matrix, perform_kqv


def test_attention_scores():
    # fill in values for the a, b and expected_output tensor.
    a = torch.tensor([]) # a three-dim tensor
    b = torch.tensor([]) # a three-dim tensor
    expected_output = torch.tensor([]) # a three-dim tensor

    A = attention.attention_scores(a, b)

    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(A, expected_output)


def test_kqv():
    input_vector_dim = 10
    n_squence = 20
    batch_size = 30
    kqv = create_kqv_matrix(input_vector_dim)
    X = torch.randn(batch_size, n_squence, input_vector_dim)
    k,q,v = perform_kqv(X, kqv)
    assert k.shape == (batch_size, n_squence, input_vector_dim)
    assert q.shape == (batch_size, n_squence, input_vector_dim)
    assert v.shape == (batch_size, n_squence, input_vector_dim)
    print(f'Test kqv passed :)')

if __name__ == "__main__":
    test_kqv()