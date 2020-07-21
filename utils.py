import torch

def get_target(temp_target, gpt_pad_token):
    pad_token = torch.tensor([gpt_pad_token])
    for idx in range(len(temp_target)):
        temp = temp_target[idx][1:]
        temp = torch.cat([temp, pad_token.cuda()], dim=-1)
        temp_target[idx] = temp

    return temp_target


def get_dec_inputs(temp_dec, gpt_pad_token, gpt_eos_token):
    pad_token = gpt_pad_token
    eos_token = gpt_eos_token
    for idx in range(len(temp_dec)):
        temp = temp_dec[idx][:].cpu().tolist()
        eos_idx = temp.index(eos_token)
        temp[eos_idx] = pad_token
        temp = torch.tensor(temp)
        temp_dec[idx] = temp

    return temp_dec.cpu()


def get_segment_ids_vaild_len(inputs, pad_token_idx):
    bert_pad_idx = pad_token_idx
    v_len_list = [0] * len(inputs)

    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            if inputs[i][j] == bert_pad_idx:
                break
            else:
                v_len_list[i] += 1

    segment_ids = torch.zeros_like(inputs).long().cuda()
    valid_length = torch.tensor(v_len_list, dtype=torch.int32)

    return segment_ids, valid_length


def gen_attention_mask(token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
        attention_mask[i][:v] = 1

    return attention_mask.float()