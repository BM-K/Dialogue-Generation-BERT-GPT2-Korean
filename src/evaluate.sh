python train.py \
  -task abs \
  -mode test \
  -batch_size 3000 \
  -test_batch_size 500 \
  -bert_data_path ../bert_data/kn \
  -log_file ../logs/test_abs_bert_kn_only \
  -model_path "../models_dec_only" \
  -sep_optim true \
  -use_interval true \
  -visible_gpus 1 \
  -max_pos 512 \
  -max_length 200 \
  -alpha 0.95 \
  -min_length 5 \
  -result_path ../logs/abs_bert_kn_result_only \
  -share_emb true \
  -use_bert_emb true \
  -test_from "../models_dec_only/model_step_14800.pt"
  # -test_all