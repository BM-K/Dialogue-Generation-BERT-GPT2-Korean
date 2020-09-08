import csv
import copy
import torch
import torch.nn as nn
from data_loader import bert_tokenizer
softmax = nn.Softmax(dim=-1)
relu = nn.ReLU()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def keyword(args, bert_vec, keyword_):
    # keyword attention tuning
    keyword_tensor = keyword_filled_pad_3(args, keyword_)
    
    #print_keyattn_exm(bert_vec, keyword_tensor, bert_tokenizer)
   
    #exit()
    # bert vector에 keyword attention 적용
    for idx in range(len(bert_vec)):
        
        #bert_vec[idx] = bert_vec[idx] * keyword_tensor[idx].view(args.max_len, -1).to(device)
        bert_vec[idx] = bert_vec[idx] + keyword_tensor[idx].view(args.max_len, -1).to(device)
    
    return bert_vec


def keyword_filled_pad_3(args, keyword_):
    one_tensor = torch.FloatTensor([0]).to(device)

    for idx in range(len(keyword_)):
        keyword_[idx] = keyword_[idx].to(device)

        # min max avg 계산
        min_tensor = torch.min(keyword_[idx])
        keyword_[idx] = keyword_[idx] - min_tensor
        
        max_tensor = torch.max(keyword_[idx])
        keyword_[idx] = keyword_[idx] / max_tensor
        
        keyword_[idx] = keyword_[idx] + 0.5
        keyword_[idx] = torch.cat([one_tensor, keyword_[idx]], dim=-1)

        # PAD token에 one tensor cat
        if len(keyword_[idx]) < args.max_len:
            for j in range(args.max_len - len(keyword_[idx])):
                keyword_[idx] = torch.cat([keyword_[idx], one_tensor], dim=-1)
        else:
            keyword_[idx] = keyword_[idx][0:args.max_len]
    
    return keyword_


def for_addition_layer(args, keyword_, refine_idx, want_tensor=None):
    
    one_tensor = torch.FloatTensor([want_tensor]).to(device)

    all_tensor = torch.rand(len(keyword_), args.max_len, args.d_model).to(device)
    inner_tensor = torch.rand(args.max_len, args.d_model).to(device)
    for_key_mask = torch.rand(len(keyword_), args.max_len).to(device)

    for idx in range(len(keyword_)):
        keyword_[idx] = keyword_[idx].to(device)

        if refine_idx is not None:
            keyword_[idx] = torch.cat([one_tensor, keyword_[idx]], dim=-1)
        else:
            keyword_len = len(keyword_[idx])

            min_tensor = torch.min(keyword_[idx])
            keyword_[idx] = (keyword_[idx] - min_tensor)

            max_tensor = torch.max(keyword_[idx])
            keyword_[idx] = keyword_[idx] / max_tensor

            keyword_[idx] = keyword_[idx] + 0.5
            keyword_[idx] = torch.cat([one_tensor, keyword_[idx]], dim=-1)

        if refine_idx is not None:
            for step, val in enumerate(refine_idx[idx]):
                if val == None:
                    continue
                keyword_[idx][val] = one_tensor
            
                
        if len(keyword_[idx]) < args.max_len:
            for j in range(args.max_len - len(keyword_[idx])):
                keyword_[idx] = torch.cat([keyword_[idx], one_tensor], dim=-1)
        else:
            keyword_[idx] = keyword_[idx][0:args.max_len]
        
        for_key_mask[idx] = keyword_[idx]
        
        for k in range(args.max_len):
            inner_tensor[k][:] = keyword_[idx][k]

        dummy_tensor = copy.deepcopy(inner_tensor).unsqueeze(0)
        
        all_tensor[idx] = dummy_tensor
    
    return all_tensor.to(device), for_key_mask.to(device)


def refine(key_list, Q_list, args):
    batch_num = 0
    lines_check = 0
    
    refine_list = []
    refine_batch_list = []
    
    for i in range(len(Q_list)):
        for j in range(len(Q_list[i])):
            
            sub_re = []
            for k in range(len(Q_list[i][j])):
                if Q_list[i][j][k].find('_') == 0:
                    sub_re.append(k+1)
            
            if len(sub_re) == 0:
                refine_list.append([None])
            else:
                refine_list.append(sub_re)
                
            batch_num += 1
            lines_check +=1

            if batch_num == args.batch_size:
                refine_batch_list.append(refine_list)
                refine_list = []
                batch_num = 0

    refine_batch_list.append(refine_list)
    
    return refine_batch_list

def refine_key(key, args):
    mean = 0
    step_num = 0
    for_update_idx = []

    # cal key score mean
    for step, score_list in enumerate(key):
        for score in score_list:
            mean += score.mean()
            step_num += 1
    
    mean = mean / step_num - args.score_ratio

    # update key score idx
    for step, score_list in enumerate(key):
        temp_list = []
        for score in score_list:
            result = (score < mean).nonzero()+1
            result = result.view(1,-1).squeeze(0).squeeze(0).squeeze(0).tolist()
            
            # 평균보다 낮은 score만 존재
            if result == []:
                temp_list.append([9999])
                continue
            
            if type(result) is not list:
                result = [result]
            temp_list.append(result)

        for_update_idx.append(temp_list)
        
    return for_update_idx


# keyword 값을 읽어와서 batch size 만큼 나눠 return
def keyword_loader(args, task, bert_tok):
    Q_list = []
    Q_batch_list = []
    key_list = []
    key_batch_list = []
    batch_num = 0
    lines_check = 0
    category = 'enter_my_keyword'

    with open(f'domain_data/3_entertain_{task}.txt.tsv', "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        lines_len = len(lines)
        for line in lines:
            Q = line.split('\t')[0]
            Q = bert_tok.tokenize(Q)
            
            Q_list.append(Q)
            batch_num += 1
            lines_check += 1
            
            if batch_num == args.batch_size:
                Q_batch_list.append(Q_list)
                Q_list = []
                batch_num = 0

        Q_batch_list.append(Q_list)
    
    batch_num = 0
    lines_check = 0

    with open(f'domain_keyword/{category}.{task}', "r", encoding="utf-8-sig") as f:
        print(f'--read keyword-- \n/{category}.{task}\n')
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
        
        key_batch_list.append(key_list)

    # 개수 체크
    assert len(key_batch_list)*args.batch_size - (args.batch_size - len(key_batch_list[-1])) == lines_len
    
    #update_idx = refine_key(key_batch_list, args)
    refine_idx = refine(key_batch_list, Q_batch_list, args)
    
    return key_batch_list, refine_idx

if __name__ == '__main__':
    print("__keyword matrix__")
