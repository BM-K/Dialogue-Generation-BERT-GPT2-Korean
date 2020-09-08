import random
import time
import copy
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from transformers import AdamW
from generation import inference
from data_loader import load_data, prepro
from keyword_matrix import keyword_loader
from transformer_based_decoder import Transformer
from transformer_based_decoder_layer import Transformer_layer
from gpt_model import GPT2, gpt_vocab, gpt_tokenizer
from matrix import acc, epoch_time, test_time_visual
from utils import get_target, get_dec_inputs, get_segment_ids_vaild_len, gen_attention_mask

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def define_args(parser):
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.00001) # 0.00001
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--n_heads', type=int, default=12)
    parser.add_argument('--train_', type=str, default='True')
    parser.add_argument('--test_', type=str, default='True')
    parser.add_argument('--useKey', type=str, default='False')
    parser.add_argument('--useKeyLayer', type=str, default='False')
    parser.add_argument('--data_dir', type=str, default='./domain_data')
    args = parser.parse_args()
    return args


iteration = 0
def train(model, iterator, optimizer, criterion, args, bert_tok):
    total_loss = 0
    iter_num = 0
    train_acc = 0
    global iteration

    model.train()

    if args.useKey == 'True':
        keyword, refine_idx = keyword_loader(args, 'train', bert_tok)
    
    for step, batch in enumerate(iterator):
        
        optimizer.zero_grad()

        enc_inputs = batch.que
    
        copy_dec_inputs = copy.deepcopy(batch.ans)
        copy_dec_target = copy.deepcopy(batch.ans)

        dec_inputs = get_dec_inputs(copy_dec_inputs, gpt_pad_token, gpt_eos_token)
        target_ = get_target(copy_dec_target, gpt_pad_token)
        target_ = target_.view(-1)
 
        segment_ids, valid_len = get_segment_ids_vaild_len(enc_inputs, pad_token_idx)
        attention_mask = gen_attention_mask(enc_inputs, valid_len)
        
        if args.useKey == 'True':
            outputs = model(enc_inputs, dec_inputs, segment_ids, attention_mask, keyword[step], refine_idx[step])
        else:
            outputs = model(enc_inputs, dec_inputs, segment_ids, attention_mask, None, refine_idx[step])
        
        loss = criterion(outputs, target_)
    
        loss.backward()
        optimizer.step()

        total_loss += loss
        iter_num += 1
        with torch.no_grad():
            tr_acc = acc(outputs, target_, gpt_pad_token)
        train_acc += tr_acc

        if step % 2 == 0:
            total_train_loss.append(total_loss.data.cpu().numpy()/iter_num)
            iteration_list.append(iteration)
            iteration += 1
    
    return total_loss.data.cpu().numpy() / iter_num, train_acc.data.cpu().numpy() / iter_num


def valid(model, iterator, optimizer, criterion, args, bert_tok):
    total_loss = 0
    iter_num = 0
    test_acc = 0
    model.eval()

    if args.useKey == 'True':
        keyword, refine_idx = keyword_loader(args, 'valid', bert_tok)

    with torch.no_grad():
        for step, batch in enumerate(iterator):
            enc_inputs = batch.que

            copy_dec_inputs = copy.deepcopy(batch.ans)
            copy_dec_target = copy.deepcopy(batch.ans)

            dec_inputs = get_dec_inputs(copy_dec_inputs, gpt_pad_token, gpt_eos_token)
            target_ = get_target(copy_dec_target, gpt_pad_token)
            target_ = target_.view(-1)

            segment_ids, valid_len = get_segment_ids_vaild_len(enc_inputs, pad_token_idx)
            attention_mask = gen_attention_mask(enc_inputs, valid_len)
            
            if args.useKey == 'True':
                outputs = model(enc_inputs, dec_inputs, segment_ids, attention_mask, keyword[step], refine_idx[step])
            else:
                outputs = model(enc_inputs, dec_inputs, segment_ids, attention_mask, None, refine_idx[step])
            
            loss = criterion(outputs, target_)

            total_loss += loss
            iter_num += 1
            te_acc = acc(outputs, target_, gpt_pad_token)
        
            test_time_visual(args, enc_inputs, outputs, target_, bert_tokenizer, gpt_vocab)
            test_acc += te_acc

        return total_loss.data.cpu().numpy() / iter_num, test_acc.data.cpu().numpy() / iter_num


def test(model, iterator, optimizer, criterion, args, bert_tok):
    total_loss = 0
    iter_num = 0
    test_acc = 0
    model.eval()
    
    if args.useKey == 'True':
        keyword, refine_idx = keyword_loader(args, 'test', bert_tok)

    with torch.no_grad():
        for step, batch in enumerate(iterator):
            enc_inputs = batch.que

            copy_dec_inputs = copy.deepcopy(batch.ans)
            copy_dec_target = copy.deepcopy(batch.ans)

            dec_inputs = get_dec_inputs(copy_dec_inputs, gpt_pad_token, gpt_eos_token)
            target_ = get_target(copy_dec_target, gpt_pad_token)
            target_ = target_.view(-1)

            segment_ids, valid_len = get_segment_ids_vaild_len(enc_inputs, pad_token_idx)
            attention_mask = gen_attention_mask(enc_inputs, valid_len)
        
            if args.useKey == 'True':
                outputs = model(enc_inputs, dec_inputs, segment_ids, attention_mask, keyword[step], refine_idx[step])
            else:
                outputs = model(enc_inputs, dec_inputs, segment_ids, attention_mask, None, refine_idx[step])

            loss = criterion(outputs, target_)

            total_loss += loss
            iter_num += 1
            te_acc = acc(outputs, target_, gpt_pad_token)

            test_acc += te_acc

        return total_loss.data.cpu().numpy() / iter_num, test_acc.data.cpu().numpy() / iter_num


def main(train_loader_, test_loader_, valid_loader_, bert_tok):
    early_stop_check = 0

    for idx, (key, value) in enumerate(args.__dict__.items()):
        if idx == 0:
            print("\nargparse{\n", "\t", key, ":", value)
        elif idx == len(args.__dict__) - 1:
            print("\t", key, ":", value, "\n}")
        else:
            print("\t", key, ":", value)

    if args.useKeyLayer == 'True' and args.useKey == 'True':
        transformer_model = Transformer_layer(cache_dir, args).to(device)
    else:
        transformer_model = Transformer(cache_dir, args).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=gpt_pad_token)
    optimizer = optim.AdamW(transformer_model.parameters(), lr=args.lr)

    best_valid_loss = float('inf')
    sorted_path = f'./output_dir/{data_file_front}_keyword_resi_layer_refine_soft.pt'

    if args.train_ == 'True':
        for epoch in range(args.num_epochs):
            start_time = time.time()

            # train, validation
            train_loss, train_acc = train(
                transformer_model, train_loader_, optimizer, criterion, args, bert_tok)

            valid_loss, valid_acc = valid(
                transformer_model, test_loader_, optimizer, criterion, args, bert_tok)

            # time cal
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if early_stop_check == args.patience:
                print("\nEarly stopping")
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

    #for i in range(len(total_train_loss)):
    #    summary.add_scalar('loss/loss_tr', total_train_loss[i], iteration_list[i])

    if args.test_ == 'True':
        if args.useKeyLayer == 'True' and args.useKey == 'True':
            transformer_model = Transformer_layer(cache_dir, args).to(device)
            transformer_model.load_state_dict(torch.load(sorted_path))
        else:
            transformer_model = Transformer(cache_dir, args).to(device)
            transformer_model.load_state_dict(torch.load(sorted_path))

        test_loss, test_acc = test(
            transformer_model, valid_loader_, optimizer, criterion, args, bert_tok)
       
        #print loss and acc
        print(f'\n\t==Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}==\n')
        first_check = 0

        while (True):
            inference(transformer_model, gpt_vocab, args, data_file_front, first_check, bert_tok)
            first_check = 1
            print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = define_args(parser)
    
    cache_dir = './cache_dir'
    iteration_list = []
    total_train_loss = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    train_loader, test_loader, valid_loader, pad_token_idx, gpt_pad_token, bert_tokenizer, data_file_front, gpt_init_token, gpt_eos_token = load_data(args, gpt_tokenizer, gpt_vocab)

    main(train_loader, test_loader, valid_loader, bert_tokenizer)
