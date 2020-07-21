import re
import torch
from torchtext import data
from transformers import BertTokenizer
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

init_token = bert_tokenizer.cls_token
pad_token = bert_tokenizer.pad_token
unk_token = bert_tokenizer.unk_token

init_token_idx = bert_tokenizer.convert_tokens_to_ids(init_token)
pad_token_idx = bert_tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = bert_tokenizer.convert_tokens_to_ids(unk_token)

#pattern = '[^\w\s]'
# tokenizer
def prepro(text):
    #text = re.sub(pattern=pattern, repl='', string=text)
    return text

def load_data(args, gpt_tokenizer):
    data_file_front = 'new_reddit'
    train_data_ = f'new_tsv_reddit_train.tsv'
    test_data_ = f'new_tsv_reddit_test.tsv'
    valid_data_ = f'new_tsv_reddit_valid.tsv'

    print("\n\t____-----Get Data-----____")
    print(f'\ttrain: {train_data_}')
    print(f'\ttest: {test_data_}')
    print(f'\tvalid: {valid_data_}')
    
    # Add the <pad> token to the vocabulary
    SPECIAL_TOKENS = {'pad_token': "<pad>", 'bos_token': "<bos>", 'eos_token': "<eos>"}
    gpt_tokenizer.add_special_tokens(SPECIAL_TOKENS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt_init_token = gpt_tokenizer.convert_tokens_to_ids(['<bos>'])[0]
    gpt_pad_token = gpt_tokenizer.convert_tokens_to_ids(['<pad>'])[0]
    gpt_eos_token = gpt_tokenizer.convert_tokens_to_ids(['<eos>'])[0]
    gpt_unk_token = gpt_tokenizer.convert_tokens_to_ids(['<|endoftext|>'])[0]
    
    # tokenizer bert
    def tokenizer_bert(text):
        text = prepro(text)
        tokens = bert_tokenizer.tokenize(text)
        tokens = bert_tokenizer.convert_tokens_to_ids(tokens)
        return tokens

    # tokenizer gpt
    def tokenizer_gpt(text):
        text = prepro(text)
        tokens = gpt_tokenizer(text)
        tokens = tokens['input_ids']#[:args.max_len-1]
        return tokens

    Question = data.Field(use_vocab=False,
                          lower=False,
                          tokenize=tokenizer_bert,
                          init_token=init_token_idx,
                          pad_token=pad_token_idx,
                          unk_token=unk_token_idx,
                          fix_length=args.max_len,
                          batch_first=True)

    Answer = data.Field(use_vocab=False,
                        lower=False,
                        tokenize=tokenizer_gpt,
                        init_token=gpt_init_token,
                        eos_token=gpt_eos_token,
                        pad_token=gpt_pad_token,
                        unk_token=gpt_unk_token,
                        fix_length=args.max_len,
                        batch_first=True)


    train_data, test_data, valid_data = TabularDataset.splits(
            path=args.data_dir, train=train_data_, test=test_data_, validation=valid_data_, format='tsv',
            fields=[('que', Question), ('ans', Answer)], skip_header=False
        )

    train_loader = BucketIterator(dataset=train_data, batch_size=args.batch_size, device=device, shuffle=False)
    test_loader = BucketIterator(dataset=test_data, batch_size=args.batch_size, device=device, shuffle=False)
    valid_loader = BucketIterator(dataset=valid_data, batch_size=args.batch_size, device=device, shuffle=False)

    return train_loader, test_loader, valid_loader, pad_token_idx, gpt_pad_token, bert_tokenizer, data_file_front, gpt_init_token, gpt_eos_token

if __name__ == "__main__":
    print("__main__ data_loader")

