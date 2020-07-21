
import torch
# from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
# from gluonnlp.data import SentencepieceTokenizer
# from kogpt2.utils import get_tokenizer
#
# tok_path = get_tokenizer()
# model, vocab = get_pytorch_kogpt2_model()
# tok = SentencepieceTokenizer(tok_path)
# sent = '2019년 한해를 보내며,'
# toked = tok(sent)


# while 1:
#   input_ids = torch.tensor([vocab[vocab.bos_token],] + vocab[toked]).unsqueeze(0)
#
#   pred = model(input_ids)
#   print(torch.argmax(pred, axis=-1).squeeze().tolist())
#   exit()
#   gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
#   print(gen)
#   exit()
#   if gen == '</s>':
#       break
#   sent += gen.replace('▁', ' ')
#   toked = tok(sent)
#
# print(sent)

import csv

def read_test():
  data_file_front = 'A_food'
  train_data = f'{data_file_front}.txt'
  total = 0
  ne = 0
  qa_pair = []
  with open(f'./aihub_category/{train_data}', "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
      total += 1
      qa = line.split('|')
      qa[1] = qa[1].replace('\n', '')
      ans = line.split('|')[1]
      ans = ans.replace('\n','')
      if ans.find('네') >= 0:
        print(ans)
        ne += 1
        continue
      else:
        qa_pair.append(qa)
  print(ne/total*100)
  exit()
  total = len(qa_pair)
  train_len = round(total * 0.8)
  train_range = range(0, train_len)
  valid_len = round((total - train_len) / 2)
  valid_range = range(train_len, train_len + valid_len)
  test_range = range(train_len + valid_len, total)

  assert total == (len(train_range) + len(test_range) + len(valid_range))

  with open(f'./aihub_category/{data_file_front}_train123.tsv', "w", encoding="utf-8") as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in train_range:
      tsv_writer.writerow([qa_pair[i][0], qa_pair[i][1]])

  with open(f'./aihub_category/{data_file_front}_test123.tsv', "w", encoding="utf-8") as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in test_range:
      tsv_writer.writerow([qa_pair[i][0], qa_pair[i][1]])

  with open(f'./aihub_category/{data_file_front}_valid123.tsv', "w", encoding="utf-8") as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in valid_range:
      tsv_writer.writerow([qa_pair[i][0], qa_pair[i][1]])

import nltk.translate.bleu_score as bleu

# 아메리카노로 뜨겁게 해주세요
candidate1 = '샷이랑 같이 넣어드릴게요'
references1 = [
    '아메리카노 주문받았습니다'
]

# 도시락 반찬 많이 나와요?
candidate11 = '가능합니다'
references11 = [
    '도시락 반찬 이름 써가지고 나오면 넣어드려요'
]

# 중화 비빔밥은 뭐 들어가요?
candidate2 = '가능합니다'
references2 = [
    '해산물은 공통적으로 다 들어가요'
]

# 아직 수박은 판매를 안하나요?
candidate3 = '여기 있습니다'
references3 = [
    '하우스 수박은 판매를 하고 있습니다.'
]

# 콜드브루랑 아메리카노랑 뭐가 달라요?
candidate4 = '콜드 브루는 찬물로 해주세요'
references4 = [
    '콜드브루가 디카페인이에요'
]

# 이거 노르웨이산인가요 연어요?
candidate5 = '아니요'
references5 = [
    '아침에 직접 잡은거에요'
]

# 수저하고 그건 어딨어요?
candidate6 = '여기 있습니다'
references6 = [
    '수저 옆쪽에 여시면 있어요'
]

# 계정 등록하신 카드인가요?
candidate7 = '여기 있습니다'
references7 = [
    '확인해드릴게요'
]


#  BLEU 점수
print(bleu.sentence_bleu(list(map(lambda ref: ref.split(), references7)),candidate7.split()))

#if __name__ == '__main__':
#  read_test()