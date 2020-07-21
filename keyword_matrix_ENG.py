import csv
import copy
import torch
import torch.nn as nn
from data_loader_ENG_bert import bert_tokenizer

softmax = nn.Softmax(dim=-1)
relu = nn.ReLU()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def keyword(args, bert_vec, keyword_):
    # keyword attention tuning
    keyword_tensor = keyword_filled_pad_3(args, keyword_)

    # bert vector에 keyword attention 적용
    for idx in range(len(bert_vec)):
        bert_vec[idx] = bert_vec[idx] * keyword_tensor[idx].view(args.max_len, -1).to(device)

    return bert_vec


def keyword_filled_pad_3(args, keyword_):
    one_tensor = torch.FloatTensor([1]).to(device)

    for idx in range(len(keyword_)):
        keyword_[idx] = keyword_[idx].to(device)
        
        # min max avg 계산
        min_tensor = torch.min(keyword_[idx])
        keyword_[idx] = keyword_[idx] - min_tensor

        max_tensor = torch.max(keyword_[idx])
        keyword_[idx] = keyword_[idx] / max_tensor

        # kw_avg = torch.sum(keyword_[idx])/len(keyword_[idx])
        keyword_[idx] = keyword_[idx] + 0.5

        # CLS token에 one tensor cat
        keyword_[idx] = torch.cat([one_tensor, keyword_[idx]], dim=-1)

        # PAD token에 one tensor cat
        if len(keyword_[idx]) < args.max_len:
            for j in range(args.max_len - len(keyword_[idx])):
                keyword_[idx] = torch.cat([keyword_[idx], one_tensor], dim=-1)
        else:
            keyword_[idx] = keyword_[idx][0:args.max_len]
        
    return keyword_


def for_addition_layer(args, keyword_):
    one_tensor = torch.FloatTensor([1]).to(device)
    all_tensor = torch.zeros(len(keyword_), args.max_len, args.d_model).to(device)
    inner_tensor = torch.zeros(args.max_len, args.d_model).to(device)

    for idx in range(len(keyword_)):
        keyword_[idx] = keyword_[idx].to(device)

        min_tensor = torch.min(keyword_[idx])
        keyword_[idx] = (keyword_[idx] - min_tensor)

        max_tensor = torch.max(keyword_[idx])
        keyword_[idx] = keyword_[idx] / max_tensor

        keyword_[idx] = keyword_[idx] + 0.5
        keyword_[idx] = torch.cat([one_tensor, keyword_[idx]], dim=-1)

        if len(keyword_[idx]) < args.max_len:
            for j in range(args.max_len - len(keyword_[idx])):
                keyword_[idx] = torch.cat([keyword_[idx], one_tensor], dim=-1)
        else:
            keyword_[idx] = keyword_[idx][0:args.max_len]

        for k in range(args.max_len):
            inner_tensor[k][:] = keyword_[idx][k]

        dummy_tensor = copy.deepcopy(inner_tensor).unsqueeze(0)
        all_tensor[idx] = dummy_tensor

    return all_tensor


# keyword 값을 읽어와서 batch size 만큼 나눠 return
def keyword_loader(args, task):
    key_list = []
    key_batch_list = []
    batch_num = 0
    lines_check = 0
    category = 'reddit'

    with open(f'reddit_mini_keyword/N_reddit_mini_keyword.{task}', "r", encoding="utf-8-sig") as f:
        #print(f'--read keyword-- \n{category}_keyword/{category}.{task}\n')
        lines = f.readlines()
        lines_len = len(lines)
        for line in lines:
            line_split = line.split(' ')
            
            line_split = [float(line_split[i]) for i in range(len(line_split))]
            line_split = torch.FloatTensor(line_split)

            key_list.append(line_split)
            batch_num += 1
            lines_check += 1
            
            if batch_num == args.batch_size:
                key_batch_list.append(key_list)
                key_list = []
                batch_num = 0
            elif lines_len == lines_check:
                key_batch_list.append(key_list)

    # 개수 체크
    assert len(key_batch_list) * args.batch_size - (args.batch_size - len(key_batch_list[-1])) == lines_len
    
    return key_batch_list


def check_keyword_bert(args, task):
    key_list = []

    with open(f'food_keyword/food_keyword.{task}', "r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:
            line_split = line.split(' ')
            key_list.append(len(line_split))

    data = []
    data_list = []
    with open(f'aihubdata/1_food_{task}.txt_{task}.tsv', "r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:
            line_split = line.split('\t')[0]
            tokens = bert_tokenizer.tokenize(line_split)
            data_list.append(len(tokens))
            data.append(tokens)

    assert len(key_list) == len(data_list)

    print(task)
    for idx in range(len(key_list)):
        if key_list[idx] != data_list[idx]:
            print("Score 길이 error")
            print(data[idx])
            print("index : ", idx, " key len : ", key_list[idx], " data len : ", data_list[idx])
            print("--------------------------")
            continue

    print("끝")


def file_():
    train = []
    with open(f'./aihubdata/1_food_valid.txt_valid.tsv', "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data_split = line.split('\t')
            q, a = data_split[0], data_split[1].strip()
            train.append([q, a])

    train_key = []
    with open(f'./food_keyword/food_keyword.valid', "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            # line = [(line[i]) for i in range(len(line))]
            train_key.append([line.strip()])

    for i in range(len(train)):
        train[i] = train[i] + train_key[i]

    with open(f'./aihubdata/1_food_valid_key.tsv', "w", encoding="utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in range(len(train)):
            tsv_writer.writerow([train[i][0], train[i][1], train[i][2]])


if __name__ == '__main__':
    print("__keyword matrix__")
    file_()
