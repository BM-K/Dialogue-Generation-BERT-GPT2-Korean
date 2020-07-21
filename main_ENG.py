import time
import copy
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from generation_ENG import inference
from tensorboardX import SummaryWriter
from data_loader_ENG_bert import load_data, prepro
from keyword_matrix_ENG import keyword_loader
from transformer_based_decoder_ENG import Transformer
from gpt_model_ENG import GPT2, gpt_tokenizer
from matrix_ENG import acc, epoch_time, test_time_visual
from transformer_based_decoder_PALs import Transformer_PALs
from utils import get_target, get_dec_inputs, get_segment_ids_vaild_len, gen_attention_mask

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
#to = torch.tensor([9776, 326, 257])
#print(gpt_tokenizer.convert_ids_to_tokens(to))
#exit()

def define_args(parser):
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--hidden_size_aug', type=int, default=204)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--n_heads', type=int, default=12)
    parser.add_argument('--train_', type=str, default='True')
    parser.add_argument('--test_', type=str, default='True')
    parser.add_argument('--PALs_', type=str, default='False')
    parser.add_argument('--data_dir', type=str, default='./new_reddit')
    args = parser.parse_args()
    return args


iteration = 0


def train(model, gpt_model, iterator, optimizer, criterion):
    total_loss = 0
    iter_num = 0
    train_acc = 0
    global iteration

    model.train()
    gpt_model.eval()
    #keyword = keyword_loader(args, 'train')
    
    for step, batch in enumerate(iterator):

        optimizer.zero_grad()
        enc_inputs = batch.que
    
        copy_dec_inputs = copy.deepcopy(batch.ans)
        copy_dec_target = copy.deepcopy(batch.ans)
        
        dec_inputs = get_dec_inputs(copy_dec_inputs, gpt_pad_token, gpt_eos_token)
        
        target_ = get_target(copy_dec_target, gpt_pad_token)
        target_ = target_.view(-1)
        
        
        with torch.no_grad():
            dec_inputs = gpt_model(dec_inputs)
    
        #segment_ids, valid_len = get_segment_ids_vaild_len(enc_inputs, pad_token_idx)
        #attention_mask = gen_attention_mask(enc_inputs, valid_len)
        
        outputs = model(enc_inputs, dec_inputs, None)
        
        loss = criterion(outputs, target_)

        loss.backward()
        optimizer.step()

        total_loss += loss
        iter_num += 1
        with torch.no_grad():
            tr_acc = acc(outputs, target_, gpt_pad_token)
        train_acc += tr_acc

        if step % 2 == 0:
            total_train_loss.append(total_loss.data.cpu().numpy() / iter_num)
            iteration_list.append(iteration)
            iteration += 1

        # test_time_visual(args, enc_inputs, outputs, target_, bert_tokenizer, gpt_vocab)

    return total_loss.data.cpu().numpy() / iter_num, train_acc.data.cpu().numpy() / iter_num


def valid(model, gpt_model, iterator, optimizer, criterion):
    total_loss = 0
    iter_num = 0
    test_acc = 0
    model.eval()
    gpt_model.eval()
    #keyword = keyword_loader(args, 'valid')

    with torch.no_grad():
        for step, batch in enumerate(iterator):
            enc_inputs = batch.que

            copy_dec_inputs = copy.deepcopy(batch.ans)
            copy_dec_target = copy.deepcopy(batch.ans)

            dec_inputs = get_dec_inputs(copy_dec_inputs, gpt_pad_token, gpt_eos_token)
            target_ = get_target(copy_dec_target, gpt_pad_token)
            target_ = target_.view(-1)

            with torch.no_grad():
                dec_inputs = gpt_model(dec_inputs)

            #segment_ids, valid_len = get_segment_ids_vaild_len(enc_inputs, pad_token_idx)
            #attention_mask = gen_attention_mask(enc_inputs, valid_len)

            outputs = model(enc_inputs, dec_inputs, None)
            loss = criterion(outputs, target_)

            total_loss += loss
            iter_num += 1
            te_acc = acc(outputs, target_, gpt_pad_token)
            print("$$ valid")
            test_time_visual(args, enc_inputs, outputs, target_, bert_tokenizer, gpt_tokenizer)
            test_acc += te_acc

        return total_loss.data.cpu().numpy() / iter_num, test_acc.data.cpu().numpy() / iter_num


def test(model, gpt_model, iterator, optimizer, criterion):
    total_loss = 0
    iter_num = 0
    test_acc = 0
    model.eval()
    gpt_model.eval()
    #keyword = keyword_loader(args, 'test')

    with torch.no_grad():
        for step, batch in enumerate(iterator):
            enc_inputs = batch.que

            copy_dec_inputs = copy.deepcopy(batch.ans)
            copy_dec_target = copy.deepcopy(batch.ans)

            dec_inputs = get_dec_inputs(copy_dec_inputs, gpt_pad_token, gpt_eos_token)
            target_ = get_target(copy_dec_target, gpt_pad_token)
            target_ = target_.view(-1)

            with torch.no_grad():
                dec_inputs = gpt_model(dec_inputs)

            #segment_ids, valid_len = get_segment_ids_vaild_len(enc_inputs, pad_token_idx)
            #attention_mask = gen_attention_mask(enc_inputs, valid_len)

            outputs = model(enc_inputs, dec_inputs, None)
            loss = criterion(outputs, target_)

            total_loss += loss
            iter_num += 1
            te_acc = acc(outputs, target_, gpt_pad_token)

            test_acc += te_acc

        return total_loss.data.cpu().numpy() / iter_num, test_acc.data.cpu().numpy() / iter_num


def main(train_loader_, test_loader_, valid_loader_):
    early_stop_check = 0

    for idx, (key, value) in enumerate(args.__dict__.items()):
        if idx == 0:
            print("\nargparse{\n", "\t", key, ":", value)
        elif idx == len(args.__dict__) - 1:
            print("\t", key, ":", value, "\n}")
        else:
            print("\t", key, ":", value)

    if args.PALs_ == 'True':
        transformer_model = Transformer_PALs(cache_dir, args).to(device)
    else:
        transformer_model = Transformer(cache_dir, args).to(device)
    
    #transformer_model = torch.nn.DataParallel(transformer_model)

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
                transformer_model, gpt_model, train_loader_, optimizer, criterion)

            valid_loss, valid_acc = valid(
                transformer_model, gpt_model, test_loader_, optimizer, criterion)

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
        if args.PALs_ == 'True':
            transformer_model = Transformer_PALs(cache_dir, args).cuda()
            transformer_model.load_state_dict(torch.load(sorted_path))
        else:
            transformer_model = Transformer(cache_dir, args).cuda()
            transformer_model.load_state_dict(torch.load(sorted_path))

        test_loss, test_acc = test(
            transformer_model, gpt_model, valid_loader_, optimizer, criterion)

        # print loss and acc
        print(f'\n\t==Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}==\n')
        first_check = 0
        
        while (True):
            inference(transformer_model, gpt_model, args, data_file_front, first_check)
            first_check = 1
            print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = define_args(parser)
    #summary = SummaryWriter('runs/gpt_bert')
    cache_dir = './cache_dir'
    iteration_list = []
    total_train_loss = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader, test_loader, valid_loader, pad_token_idx, gpt_pad_token, bert_tokenizer, data_file_front, gpt_init_token, gpt_eos_token = \
        load_data(args, gpt_tokenizer)
    print("gpt_pad :", gpt_pad_token)

    main(train_loader, test_loader, valid_loader)
