import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel, BertConfig
gpt_vocab_size = 50000
d_ff = 2048  # FeedForward dimension

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(3).unsqueeze(1)  # PAD = 3 이므로 eq(3)
    # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)  # 상삼각행렬 반환
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = int(args.d_model / args.n_heads)

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)
        # padding 부분을 -1000000 처럼 큰 음수값을 할당하여 softmax 후 해당 값을 0으로 나오게 만들어야함.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.args = args
        self.d_k = int(args.d_model / args.n_heads)
        self.d_v = int(args.d_model / args.n_heads)
        self.n_heads = args.n_heads
        self.W_Q = nn.Linear(args.d_model, self.d_k * args.n_heads)  # init (512 x 64 * 8)
        self.W_K = nn.Linear(args.d_model, self.d_k * args.n_heads)
        self.W_V = nn.Linear(args.d_model, self.d_v * args.n_heads)
        self.li1 = nn.Linear(args.n_heads * self.d_v, args.d_model)
        self.layer_norm = nn.LayerNorm(args.d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # q_s:[batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # k_s:[batch_size x n_heads x len_q x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s:[batch_size x n_heads x len_q x d_v]

        context, attn = ScaledDotProductAttention(self.args)(q_s, k_s, v_s, attn_mask=None)
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.li1(context)

        return self.layer_norm(output + residual), attn
        # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=args.d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=args.d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = self.relu(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.dec_enc_attn = MultiheadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_inputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        with torch.no_grad():
            dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs  #, dec_self_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])

    def forward(self, dec_inputs, enc_outputs):  # dec_inputs : [batch_size x target_len]
        for layer in self.layers:
            dec_outputs = \
                layer(dec_inputs, enc_outputs, dec_self_attn_mask=None, dec_enc_attn_mask=None)

        return dec_outputs

class ETRI_KOBERT(nn.Module):
    def __init__(self, temp_dir, args):
        super(ETRI_KOBERT, self).__init__()
        self.li1 = nn.Linear(768, args.d_model)
        self.model = BertModel.from_pretrained("./ETRI_KoBERT/003_bert_eojeol_pytorch", cache_dir=temp_dir)

    def forward(self, x, segs, mask):
        top_vec, _ = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        top_vec = self.li1(top_vec)
        return top_vec

class Transformer(nn.Module):
    def __init__(self, cache_dir, args):
        super(Transformer, self).__init__()
        self.bert = ETRI_KOBERT(cache_dir, args)
        self.decoder = Decoder(args)
        self.projection = nn.Linear(args.d_model, gpt_vocab_size, bias=False)
        bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=768,
                                 num_hidden_layers=12, num_attention_heads=12)
        self.bert.model = BertModel(bert_config)

    def forward(self, enc_inputs, dec_inputs, segment_ids, attn_mask):
        bert_encoding_vec = self.bert(enc_inputs, segment_ids, attn_mask)
        dec_outputs = self.decoder(dec_inputs, bert_encoding_vec)
        dec_logits = self.projection(dec_outputs)  # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1))

