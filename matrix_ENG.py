import torch

# acc 계산 함수
def acc(yhat, y, gpt_pad_token):
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1] # [0]: max value, [1]: index of max value
        acc = (yhat == y).float()[y != gpt_pad_token].mean()  # padding은 acc에서 제거
    return acc

# 시간 계산 함수
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

# 학습시 학습되는 q, a 출력
def test_time_visual(args, inputs, outputs, targets, bert_tk, gpt_tokenizer):
    print("\t---------------")
    input_sentence_list = bert_tk.convert_ids_to_tokens(inputs[0])
    input_sentence = ""
    for idx, token in enumerate(input_sentence_list):
        if idx == 0:
            continue  # [CLS] token continue
        elif token == '[PAD]':
            break  # bert pad token
        else:
            input_sentence += " "+token
    print("input> ", input_sentence.replace('##', ''))

    targets = targets[0:args.max_len].squeeze().tolist()
    target_sentence = ""
    
    for idx, token in enumerate(targets):
        if token == '<eos>':
            break
        else:
            target_sentence += gpt_tokenizer.convert_ids_to_tokens(token)
    print("target> ", target_sentence.replace('Ġ', ' '))

    outputs = outputs.max(dim=-1)[1]
    outputs = outputs[0:args.max_len].squeeze().tolist()
    outputs_sentence = ""
    for idx, token in enumerate(outputs):
        # if token == gpt_vocab[gpt_vocab.bos_token]:
        #     continue
        if token == '<eos>':
            break
        else:
            outputs_sentence += gpt_tokenizer.convert_ids_to_tokens(token)
    print("output> ", outputs_sentence.replace('Ġ', ' '))
