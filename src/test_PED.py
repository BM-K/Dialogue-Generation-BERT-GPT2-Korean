import torch
import torch.nn as nn

from others.tokenization_etri_eojeol import BertTokenizer

class M2M(nn.Module):
    def __init__(self):
        super().__init__()




if __name__ == "__main__":

    #enc_model = BertModel.from_pretrained("/workspace/kobert_sum/PreSumm/ETRI_KoBERT/003_bert_eojeol_pytorch")
    enc_tokenizer = BertTokenizer.from_pretrained('/workspace/BERTSUM/kobert_sum/PreSumm/ETRI_koBERT/003_bert_eojeol_pytorch/vocab.txt', do_lower_case=False)

    # dec_model, dec_vocab = get_pytorch_kogpt2_model()


    #tok_path = get_tokenizer()
    #dec_tokenize = SentencepieceTokenizer(tok_path)
    #print(dec_tokenize('안녕하세요 반갑습니다.'))
    #print(dec_vocab[dec_tokenize('안녕하세요 반갑습니다.')])
    #exit()


