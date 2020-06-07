import csv
from pathlib import Path

def to_tsv(file_name):
    file_data = []
    with open(f'./aihub_category/{file_name}', "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data_split = line.split('|')
            q, a = data_split[0], data_split[1].strip()
            file_data.append([q, a])

    total = len(file_data)
    train_len = round(total * 0.8)
    train_range = range(0, train_len)
    valid_len = round((total - train_len)/2)
    valid_range = range(train_len, train_len+valid_len)
    test_range = range(train_len+valid_len, total)

    assert total == (len(train_range) + len(test_range) + len(valid_range))

    with open(f'./aihub_category/{file_name}_train.tsv', "w", encoding="utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in train_range:
            tsv_writer.writerow([file_data[i][0], file_data[i][1]])

    with open(f'./aihub_category/{file_name}_test.tsv', "w", encoding="utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in test_range:
            tsv_writer.writerow([file_data[i][0], file_data[i][1]])

    with open(f'./aihub_category/{file_name}_valid.tsv', "w", encoding="utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in valid_range:
            tsv_writer.writerow([file_data[i][0], file_data[i][1]])


data_dir = Path(f"aihub_category")
list_ann = list(data_dir.glob("*.txt"))
total = len(list_ann)

for i in range(total):
    current_path = list_ann[i]
    print(f"\n\t[{i + 1}/{total}] processing '{current_path.name}'")
    to_tsv(str(current_path.name))
