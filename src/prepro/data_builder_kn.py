import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin

import torch
from multiprocess import Pool

from others.logging import logger
from others.tokenization_etri_eojeol import BertTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams

import tqdm

def add_tokens(tokenizer):
    symbols = ["[EOQ]"]
    last_ids = tokenizer.vocab_size - 1
    for idx, symbol in enumerate(symbols):
        if not (symbol in tokenizer.vocab) :
            tokenizer.vocab[symbol] = last_ids + idx + 1
            tokenizer.ids_to_tokens.update({last_ids + idx + 1: symbol})

    return tokenizer

def load_json(p, lower=False):
    regex = r"([\w][.])+?[ ]([\w])"
    to = r"\1\n\2"

    examples = []
    flag = False

    for data in tqdm.tqdm(json.load(open(p))):
        if not (data['type'] == "video"):
            if len(data["content"]) == 0 or len(data['summary']) == 0:
                continue
            src_sentences = re.sub(regex, to, data["content"].strip()).split("\n")
            tgt_sentences = re.sub(regex, to, data['summary'].strip()).split("\n")


            src_tokens = [clean(sent).split(' ') for sent in src_sentences]
            tgt_tokens = [clean(sent).split(' ') for sent in tgt_sentences]


            examples.append({"src" : src_tokens, "tgt" : tgt_tokens})

    return examples


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)

class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('../ETRI_koBERT/003_bert_eojeol_pytorch/vocab.txt', do_lower_case=False)

        if self.args.add_tokens:
            self.tokenizer = add_tokens(self.tokenizer)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '<S>'
        self.tgt_eos = '<T>'
        self.tgt_sent_split = '[EOQ]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        # 문장 사이 [SEP] [CLS] 토큰 넣기
        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        # 문장 양끝에 넣기
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        # tgt_subtokens_str = '[BOS] ' + ' [EOQ] '.join(
        #     [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in tgt]) + ' [EOS]'
        tgt_subtokens_str = '<S> ' + ' '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in tgt]) + ' <T>'

        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = ' <q> '.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        if is_test:
            with open("../bert_data_test" + "/bert_data_test.txt", "w") as f:
                f.write("src_subtokens_ids({}) : {} \n".format(len(src_subtoken_idxs),src_subtoken_idxs))
                f.write("src_subtokens_ids_tokens : {} \n".format([self.tokenizer.ids_to_tokens[int(id)] for id in src_subtoken_idxs]))
                f.write("src_subtokens({}) : {} \n".format(len(src_subtokens), src_subtokens))
                f.write("sent_labels({}) : {} \n".format(len(src_subtoken_idxs),src_subtoken_idxs))
                f.write("tgt_subtokens({}) : {} \n".format(len(tgt_subtokens_str), tgt_subtokens_str))
                f.write("tgt_subtoken_idxs({}) : {} \n".format(len(tgt_subtoken_idxs),tgt_subtoken_idxs))
                f.write("segments_ids({}) : {} \n".format(len(segments_ids),segments_ids))
                f.write("cls_ids({}) : {} \n".format(len(cls_ids),cls_ids))
                f.write("src_txt({}) : {} \n".format(len(src_txt),src_txt))
                f.write("tgt_txt({}) : {} \n".format(len(tgt_txt),tgt_txt))


        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(args):

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']

        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):

    corpus_mapping = {"valid" : ["valid"],
                      "train" : ["train"],
                      "test" : ["test"]}
    corpus_type = ['valid', 'test', 'train']

    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = f.split('/')[-1].split('.')[0]
        if (real_name in corpus_mapping["valid"]):
            valid_files.append(f)
        elif (real_name in corpus_mapping["test"]):
            test_files.append(f)
        elif (real_name in corpus_mapping["train"]):
            train_files.append(f)
        # else:
        #     train_files.append(f)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    print(corpora)
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset += d

        pool.close()
        pool.join()

        div, mod = divmod(len(dataset), args.shard_size)

        n_iter = div if mod == 0 else div + 1

        for p_ct in tqdm.tqdm(range(n_iter), desc="Shard Iter: "):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)

            with open(pt_file, 'w') as save:
                shard_size_dataset = dataset[:args.shard_size]
                save.write(json.dumps(shard_size_dataset, ensure_ascii=False))
                dataset = dataset[args.shard_size:]


def _format_to_lines(params):
    f, args = params
    print(f)
    examples = load_json(f, args.lower)
    return examples


