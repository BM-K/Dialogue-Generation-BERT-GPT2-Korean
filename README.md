# QA_Model_bert-gpt

## Model Intro
Question을 KoBERT를 통해 representation 하고, Answer를 pre-trained gpt2 <<self attn까지>> 를 통과시켜 벡터 표현을 얻는다. 그 후 BERT 표현과 GPT표현을 transformer <<enc-dec attn -> ff>> 로 통과시켜 학습함. 단 << >> 구간에선 gradient를 fix 시킴. <br>
-본 repo에선 데이터와 KoBERT 모델을 제공하지 않음-
## How to install
```ruby
git clone https://github.com/BM-K/QA_Model_bert-gpt.git
cd QA_Model_bert-gpt
pip install -r requirements_gpt.txt
pip install .
```
## How to use
```ruby
python data_to_tsv ※ aihub_category 데이터가 있는경우 
python main.py --train_ True --batch_size 256 --num_epochs 15 --data_dir ./$your_data_dir
<<user only argparse>>
``` 
