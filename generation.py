import math
import copy
import torch
from konlpy.tag import Mecab
from chatspace import ChatSpace
import nltk.translate.bleu_score as bleu
from keyword_matrix import keyword_loader
from data_loader import bert_tokenizer, prepro
from gpt_model import GPT2, gpt_vocab, gpt_tokenizer
from utils import get_segment_ids_vaild_len, gen_attention_mask, concat_pad

mecab = Mecab()
spacer = ChatSpace()

init_token = bert_tokenizer.cls_token
pad_token = bert_tokenizer.pad_token

init_token_idx = bert_tokenizer.convert_tokens_to_ids(init_token)
pad_token_idx = bert_tokenizer.convert_tokens_to_ids(pad_token)

init_token = torch.tensor([init_token_idx])
pad_token = torch.tensor([pad_token_idx])

gpt_init_token = gpt_vocab[gpt_vocab.bos_token]
gpt_eos_token = gpt_vocab[gpt_vocab.eos_token]


def inference(model, gpt_vocab, args, data_file_front, first_check, bert_tok):

    if first_check==0:
        print("\nPlease wait... calculating distinct, bleu score")

        # distinct score 를 구하기 위한 부분
        Q_list, A_list, P_list = infer_test_set(model, gpt_vocab, args, data_file_front, bert_tok)

        # bleu score 를 구하기 위한 부분
        Bleu_score(A_list, P_list)
    
    exit()
    # 아래부터 inference
    sentence = input("문장을 입력하세요 : ")
    sentence = prepro(sentence)

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
       
        keyword = make_random_keyword(enc_inputs)
        
        y_pred = model(enc_inputs, dec_in, segment_ids, attention_mask, None)
        y_pred_ids = y_pred.max(dim=-1)[1]

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

# test set Q에 대한 Inference
# distinct score를 구하기 위한 함수
def infer_test_set(model, gpt_vocab, args, data_file_front, bert_tok):
    valid_data_ = f'{data_file_front}_test.txt.tsv'
    valid_data_list = []
    valid_Q_list = []
    valid_A_list = []
    valid_P_list = []
    with open(f'{args.data_dir}/{valid_data_}', "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        all_val_len = len(lines)

        for line in lines:
            data_split = line.split('\t')
            q, a = data_split[0], data_split[1].strip()
            valid_data_list.append([q, a])
    
    data_len = 0
    
    if args.useKey == 'True':
        keyword, refine_idx = keyword_loader(args, 'test', bert_tok)
      
    all_len = all_val_len
    all_val_len = math.ceil(all_val_len / args.batch_size)
    
    for step in range(all_val_len):
        batch_step=0
        
        if step+1 == all_val_len:
            max_batch_len = all_len - args.batch_size * step    
        else:
            max_batch_len = args.batch_size
        
        for qaidx in range(max_batch_len):
            
            input_sentence = valid_data_list[data_len][0]
            gold_sentence = valid_data_list[data_len][1]

            sentence = prepro(input_sentence)
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
        
            for i in range(args.max_len):
                
                if i == 0:
                    dec_inputs = concat_pad(args, dec_inputs)

                if args.useKey == 'True' or args.useKeyLayer == 'True':
                    y_pred = model(enc_inputs, dec_inputs, segment_ids, attention_mask, [keyword[step][batch_step]], [refine_idx[step][batch_step]])
                else:
                    y_pred = model(enc_inputs, dec_inputs, segment_ids, attention_mask, None, False)
                
                y_pred_ids = y_pred.max(dim=-1, keepdim=False)[1]
                next_word = y_pred_ids.data[i]
                next_symbol = next_word.item()
                 
                if (next_symbol == gpt_eos_token):
                    
                    y_pred_ids = y_pred_ids.squeeze(0).cpu()
                    for idx in range(len(y_pred_ids)):
                        if y_pred_ids[idx] == gpt_eos_token:
                            pred = [pred[x].numpy().tolist() for x in range(len(pred))]
                            pred = list(pred)
                            pred = gpt_vocab.to_tokens(pred)
                            pred_sentence = "".join(pred)
                            pred_sentence = pred_sentence.replace('▁', ' ')
                            pred_str = spacer.space(pred_sentence)
                            valid_Q_list.append(input_sentence)
                            valid_A_list.append(gold_sentence)
                            valid_P_list.append(pred_str)

                            break
                        else:
                            pred.append(y_pred_ids[idx])
                    break

                else:
                    try:
                        dec_inputs[0][i+1] = next_symbol
                    except IndexError:
                        break
                    
            data_len += 1
            batch_step += 1
            if batch_step == max_batch_len:
                break

    valid_P_sen = ""
    for idx in range(len(valid_P_list)):
        if idx == 0:
            valid_P_sen = valid_P_list[idx]
        else:
            valid_P_sen = valid_P_sen + " " + valid_P_list[idx]

    with open(f'./for_distinct/{data_file_front}.txt', "w", encoding="utf-8-sig") as out_file:
        hyung = mecab.pos(u'{0}'.format(valid_P_sen))
        temp_sen = ""
        for i in range(len(hyung)):
            if i == 0:
                temp_sen = hyung[i][0]
            else:
                temp_sen = temp_sen + " " + hyung[i][0]
        pred = f'{temp_sen}'
        out_file.write(pred)

    with open(f'./output_QAP/{data_file_front}_QAP.txt', "w", encoding="utf-8-sig") as out_file:
        for i in range(len(valid_Q_list)):
            msg_Q = f'Q> {valid_Q_list[i]}\n'
            msg_A = f'A> {valid_A_list[i]}\n'
            msg_P = f'P> {valid_P_list[i]}\n'
            out_file.write(msg_Q)
            out_file.write(msg_A)
            out_file.write(msg_P)
            out_file.write("----------------\n")
    
    with open(f'./output_QAP/{data_file_front}_Q.txt', "w", encoding="utf-8-sig") as out_file:
        for i in range(len(valid_Q_list)):
            msg_Q = f'{valid_Q_list[i]}\n'
            out_file.write(msg_Q)
    with open(f'./output_QAP/{data_file_front}_P.txt', "w", encoding="utf-8-sig") as out_file:
        for i in range(len(valid_P_list)):
            msg_P = f'{valid_P_list[i]}\n'
            out_file.write(msg_P)

    return valid_Q_list, valid_A_list, valid_P_list

def Bleu_score(A_list, P_list):
    hyung_tae_so_A_list = []
    hyung_tae_so_P_list = []
    bleu_score = 0

    for token in A_list:
        hyung_tae_so_A = bert_tokenizer.tokenize(token)
        temp_sen = ' '.join(hyung_tae_so_A)
        hyung_tae_so_A_list.append(temp_sen)

    for token in P_list:
        hyung_tae_so_P = bert_tokenizer.tokenize(token)
        temp_sen = ' '.join(hyung_tae_so_P)
        hyung_tae_so_P_list.append(temp_sen)

    for idx in range(len(A_list)):
        candidate = hyung_tae_so_P_list[idx].split(' ')
        references = hyung_tae_so_A_list[idx].split(' ')
        
        #print(candidate, "||", references)
        #bleu_score += bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)), candidate.split())
        bleu_score += bleu.sentence_bleu([references], candidate, weights=(1, 0, 0, 0))
        
    print("avg bleu score >", bleu_score/len(A_list))
