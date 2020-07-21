import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SPECIAL_TOKENS = {'pad_token': "<pad>", 'bos_token': "<bos>", 'eos_token': "<eos>"}
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = GPT2Model.from_pretrained('gpt2')
model.resize_token_embeddings(len(gpt_tokenizer))

gpt_vocab = gpt_tokenizer.special_tokens_map  # 일단 여기서 터질거임.

class GPT2(nn.Module):
    def __init__(self):
        super(GPT2, self).__init__()
        self.gpt_model = model

    def forward(self, inputs):
    
        pred = model(inputs)
        
        return pred.to(device)

