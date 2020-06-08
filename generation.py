import torch
from chatspace import ChatSpace
from data_loader import bert_tokenizer, prepro
from utils import get_segment_ids_vaild_len, gen_attention_mask
spacer = ChatSpace()

def inference(model, gpt_model, gpt_vocab, args):
    gpt_init_token = gpt_vocab[gpt_vocab.bos_token]
    gpt_pad_token = gpt_vocab[gpt_vocab.padding_token]
    gpt_eos_token = gpt_vocab[gpt_vocab.eos_token]
    gpt_unk_token = gpt_vocab[gpt_vocab.unknown_token]

    init_token = bert_tokenizer.cls_token
    pad_token = bert_tokenizer.pad_token
    init_token_idx = bert_tokenizer.convert_tokens_to_ids(init_token)
    pad_token_idx = bert_tokenizer.convert_tokens_to_ids(pad_token)

    sentence = input("문장을 입력하세요 : ")
    sentence = prepro(sentence)
    init_token = torch.tensor([init_token_idx])
    pad_token = torch.tensor([pad_token_idx])

    tokens = bert_tokenizer.tokenize(sentence)
    enc_inputs = torch.tensor([bert_tokenizer.convert_tokens_to_ids(tokens)])
    enc_inputs = torch.cat([init_token.unsqueeze(0), enc_inputs], dim=-1)

    cur_enc_len = len(enc_inputs[0])
    for i in range(args.max_len - cur_enc_len):
        enc_inputs = torch.cat([enc_inputs, pad_token.unsqueeze(0)], dim=-1)
    enc_inputs = enc_inputs.cuda()

    segment_ids, valid_len = get_segment_ids_vaild_len(enc_inputs, pad_token_idx)
    attention_mask = gen_attention_mask(enc_inputs, valid_len)

    dec_inputs = torch.tensor([[gpt_init_token]])
    pred = []

    model.eval()
    gpt_model.eval()

    for i in range(args.max_len):
        with torch.no_grad():
            dec_in = gpt_model(dec_inputs.cpu())

        y_pred = model(enc_inputs, dec_in, segment_ids, attention_mask)
        y_pred_ids = y_pred.max(dim=-1)[1]
        print("yp : ", y_pred_ids)
        if (y_pred_ids[-1] == gpt_eos_token):
            y_pred_ids = y_pred_ids.squeeze(0).cpu()
            for idx in range(len(y_pred_ids)):
                if y_pred_ids[idx] == gpt_eos_token:
                    pred = [pred[x].numpy().tolist() for x in range(len(pred))]
                    pred = list(pred)
                    pred = gpt_vocab.to_tokens(pred)
                    pred_sentence = "".join(pred)
                    pred_sentence = pred_sentence.replace('▁', ' ')
                    pred_str = spacer.space(pred_sentence)
                    print(">> ", pred_str)
                    break
                else:
                    pred.append(y_pred_ids[idx])
            return 0

        else:
            dec_inputs = torch.cat([
                dec_inputs.cpu(), y_pred_ids[-1].unsqueeze(0).unsqueeze(0).cpu()], dim=-1)
