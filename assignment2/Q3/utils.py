import numpy as np
import torch

NER_TRAIN = r"../ner/train"
NER_DEV = r"../ner/dev"
NER_TEST = r"../ner/test"

POS_TRAIN = r"../pos/train"
POS_DEV = r"../pos/dev"
POS_TEST = r"../pos/test"

DTYPE = torch.float32

torch_to_numpy_dtype = {
    torch.float32: np.float32,
    torch.float64: np.float64
}