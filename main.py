import time
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from chatspace import ChatSpace
from data_loader import load_data, prepro
from transformer_based_decoder import Transformer
from matrix import acc, epoch_time, test_time_visual
from gpt_model import GPT2, gpt_vocab, gpt_tokenizer
spacer = ChatSpace()

def define_args(parser):
    parser.add_argument('--max_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.005)  # 256 0.001
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=768)  # for tran
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--n_heads', type=int, default=12)
    parser.add_argument('--train_', type=str, default='True')
    parser.add_argument('--test_', type=str, default='True')
    parser.add_argument('--data_dir', type=str, default='./aihub_category')
    args = parser.parse_args()
    return args

def get_target(temp_target):
    pad_token = torch.tensor([gpt_pad_token])
    for idx in range(len(temp_target)):
        temp = temp_target[idx][1:]
        temp = torch.cat([temp, pad_token.cuda()], dim=-1)
        temp_target[idx] = temp

    return temp_target

def get_dec_inputs(temp_dec):
    pad_token = gpt_pad_token
    eos_token = gpt_eos_token
    for idx in range(len(temp_dec)):
        temp = temp_dec[idx][:].cpu().tolist()
        eos_idx = temp.index(eos_token)
        temp[eos_idx] = pad_token
        temp = torch.tensor(temp)
        temp_dec[idx] = temp

    return temp_dec.cpu()

def get_segment_ids_vaild_len(inputs):
    bert_pad_idx = pad_token_idx
    v_len_list = [0] * len(inputs)

    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            if inputs[i][j] == bert_pad_idx:
                break
            else:
                v_len_list[i] += 1

    segment_ids = torch.zeros_like(inputs).long().to(device)
    valid_length = torch.tensor(v_len_list, dtype=torch.int32)

    return segment_ids, valid_length

def gen_attention_mask(token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
        attention_mask[i][:v] = 1
    return attention_mask.float()

def train(model, gpt_model, iterator, optimizer, criterion):
    total_loss = 0
    iter_num = 0
    train_acc = 0
    model.train()
    gpt_model.eval()

    for step, batch in enumerate(iterator):
        optimizer.zero_grad()

        enc_inputs = batch.que

        copy_dec_inputs = copy.deepcopy(batch.ans)
        copy_dec_target = copy.deepcopy(batch.ans)

        dec_inputs = get_dec_inputs(copy_dec_inputs)
        target_ = get_target(copy_dec_target)
        target_ = target_.view(-1)

        with torch.no_grad():
            dec_inputs = gpt_model(dec_inputs)

        segment_ids, valid_len = get_segment_ids_vaild_len(enc_inputs)
        attention_mask = gen_attention_mask(enc_inputs, valid_len)

        outputs = model(enc_inputs, dec_inputs, segment_ids, attention_mask)
        loss = criterion(outputs, target_)

        loss.backward()
        optimizer.step()

        total_loss += loss
        iter_num += 1
        with torch.no_grad():
            tr_acc = acc(outputs, target_, gpt_pad_token)
        train_acc += tr_acc

        #test_time_visual(args, enc_inputs, outputs, target_, bert_tokenizer, gpt_vocab)
    return total_loss.data.cpu().numpy() / iter_num, train_acc.data.cpu().numpy() / iter_num

def valid(model, gpt_model, iterator, optimizer, criterion):
    total_loss = 0
    iter_num = 0
    test_acc = 0
    model.eval()
    gpt_model.eval()

    with torch.no_grad():
        for step, batch in enumerate(iterator):

            enc_inputs = batch.que

            copy_dec_inputs = copy.deepcopy(batch.ans)
            copy_dec_target = copy.deepcopy(batch.ans)

            dec_inputs = get_dec_inputs(copy_dec_inputs)
            target_ = get_target(copy_dec_target)
            target_ = target_.view(-1)

            with torch.no_grad():
                dec_inputs = gpt_model(dec_inputs)

            segment_ids, valid_len = get_segment_ids_vaild_len(enc_inputs)
            attention_mask = gen_attention_mask(enc_inputs, valid_len)

            outputs = model(enc_inputs, dec_inputs, segment_ids, attention_mask)
            loss = criterion(outputs, target_)

            total_loss += loss
            iter_num += 1
            te_acc = acc(outputs, target_, gpt_pad_token)
            test_time_visual(args, enc_inputs, outputs, target_, bert_tokenizer, gpt_vocab)
            test_acc += te_acc

        return total_loss.data.cpu().numpy() / iter_num, test_acc.data.cpu().numpy() / iter_num

def test(model, gpt_model, iterator, optimizer, criterion):
    total_loss = 0
    iter_num = 0
    test_acc = 0
    model.eval()
    gpt_model.eval()

    with torch.no_grad():
        for step, batch in enumerate(iterator):
            
            enc_inputs = batch.que

            copy_dec_inputs = copy.deepcopy(batch.ans)
            copy_dec_target = copy.deepcopy(batch.ans)

            dec_inputs = get_dec_inputs(copy_dec_inputs)
            target_ = get_target(copy_dec_target)
            target_ = target_.view(-1)

            with torch.no_grad():
                dec_inputs = gpt_model(dec_inputs)

            segment_ids, valid_len = get_segment_ids_vaild_len(enc_inputs)
            attention_mask = gen_attention_mask(enc_inputs, valid_len)

            outputs = model(enc_inputs, dec_inputs, segment_ids, attention_mask)
            loss = criterion(outputs, target_)

            total_loss += loss
            iter_num += 1
            te_acc = acc(outputs, target_, gpt_pad_token)
            test_time_visual(args, enc_inputs, outputs, target_, bert_tokenizer, gpt_vocab)
            test_acc += te_acc

        return total_loss.data.cpu().numpy() / iter_num, test_acc.data.cpu().numpy() / iter_num

def inference(model, gpt_model):
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

    segment_ids, valid_len = get_segment_ids_vaild_len(enc_inputs)
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

def main(Que, Ans, train_loader, test_loader, valid_loader):
    early_stop_check = 0

    for idx, (key, value) in enumerate(args.__dict__.items()):
        if idx == 0:
            print("\nargparse{\n", "\t", key, ":", value)
        elif idx == len(args.__dict__) - 1:
            print("\t", key, ":", value, "\n}")
        else:
            print("\t", key, ":", value)

    transformer_model = Transformer(cache_dir, args).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=gpt_pad_token)
    optimizer = optim.Adam(transformer_model.parameters(), lr=args.lr)

    best_valid_loss = float('inf')
    sorted_path = f'./output_dir/{data_file_front}.pt'

    gpt_model = GPT2()

    if args.train_ == 'True':
        for epoch in range(args.num_epochs):
            start_time = time.time()

            # train, validation
            train_loss, train_acc = train(
                transformer_model, gpt_model, train_loader, optimizer, criterion)

            valid_loss, valid_acc = valid(
                transformer_model, gpt_model, valid_loader, optimizer, criterion)
            
            # time cal
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if early_stop_check == args.patience:
                print("Early stopping")
                break

            # 전에 학습된 loss 보다 현재 loss 가 더 낮을시 모델 저장.
            if valid_loss < best_valid_loss:
                early_stop_check = 0
                best_valid_loss = valid_loss
                torch.save(transformer_model.state_dict(), sorted_path)
                print(f'\n\t## SAVE valid_loss: {valid_loss:.3f} | valid_acc: {valid_acc:.3f} ##')
            else:
                early_stop_check += 1

            # print loss and acc
            print(f'\n\t==Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s==')
            print(f'\t==Train Loss: {train_loss:.3f} | Train acc: {train_acc:.3f}==')
            print(f'\t==Valid Loss: {valid_loss:.3f} | Valid acc: {valid_acc:.3f}==\n')

    if args.test_ == 'True':
        transformer_model = Transformer(cache_dir, args).cuda()
        transformer_model.load_state_dict(torch.load(sorted_path))

        test_loss, test_acc = test(
            transformer_model, gpt_model, test_loader, optimizer, criterion)

        # print loss and acc
        print(f'\n\t==Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}==\n')

        while (True):
            inference(transformer_model, gpt_model)
            print("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = define_args(parser)
    cache_dir = './cache_dir'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    Que, Ans, train_loader, test_loader, valid_loader, pad_token_idx, gpt_pad_token, bert_tokenizer, data_file_front, gpt_init_token, gpt_eos_token= \
        load_data(args, gpt_tokenizer, gpt_vocab)
    main(Que, Ans, train_loader, test_loader, valid_loader)
