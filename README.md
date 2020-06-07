# QA_Model_bert-gpt

## Model Intro
Question을 KoBERT를 통해 representation 하고, Answer를 pre-trained gpt2 <<self attn까지>> 를 통과시켜 벡터 표현을 얻는다. 그 후 BERT 표현과 GPT표현을 transformer <<enc-dec attn -> ff>> 로 통과시켜 학습함. 단 << >> 구간에선 gradient를 fix 시킴.
## How to install
```ruby
git clone https://github.com/BM-K/QA_Model_bert-gpt.git
cd QA_Model_bert-gpt
pip install -r requirements_gpt.txt
pip install .
```
## How to use
```ruby
python main.py --train_ True --batch_size 256 --num_epochs 15
<<user only argparse>>
``` 
