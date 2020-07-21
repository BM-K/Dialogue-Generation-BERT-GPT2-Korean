import torch
import torch.nn as nn
from KoGPT2.kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from KoGPT2.kogpt2.utils import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer

tok_path = get_tokenizer()
_, gpt_vocab = get_pytorch_kogpt2_model()
gpt_tokenizer = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)

class GPT2(nn.Module):
    def __init__(self):
        super(GPT2, self).__init__()
        self.tok_path = get_tokenizer()
        self.gpt_model, self.vocab = get_pytorch_kogpt2_model()

    def forward(self, inputs):
        pred = self.gpt_model(inputs)
        return pred.to('cuda')
