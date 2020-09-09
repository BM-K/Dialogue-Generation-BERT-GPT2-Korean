# Dialogue-Generation-BERT-GPT2-Korean using keyword layer
연구용 레파지토리 <br>
'aihub data' was used

## How to use
본 repo에선 ETRI KoBERT를 제공하지 않음

install konlpy mecab (for inference) -> https://konlpy.org/ko/stable/install/
```ruby
git clone https://github.com/BM-K/Dialogue-Generation-BERT-GPT2-Korean.git
cd Dialogue-Generation-BERT-GPT2-Korean
git clone https://github.com/SKT-AI/KoGPT2.git

mv modeling_gpt2.py {/your transformers lib}
＊ your transformers lib example => /opt/conda/lib/python3.7/site-packages/transformers/

pip install -r requirements_gpt.txt
python main.py
```

## Screen after execution
<img src = "https://user-images.githubusercontent.com/55969260/88027255-f0ce0d00-cb71-11ea-9cfe-8f5849acb0c9.png">
<img src = "https://user-images.githubusercontent.com/55969260/88027416-2ecb3100-cb72-11ea-9918-921ca5a7dd0f.png">
