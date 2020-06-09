# Step 1
python preprocess.py \
  -mode format_to_lines \
  -raw_path "../raw_data/kn" \
  -save_path "../json_data/kn" \
  -n_cpus 2 \
  -map_path ""

# Step 2
python preprocess.py \
  -mode format_to_bert \
  -raw_path "../json_data"  \
  -save_path "../bert_data" \
  -n_cpus 2 \
  -log_file ../logs/preprocess.log \
  -use_bert_basic_tokenizer true \
  -lower false \
  -add_tokens false \
