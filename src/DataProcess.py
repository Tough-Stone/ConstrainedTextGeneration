import json
import random
import re
import sys
import time

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from transformers import BertTokenizer


def is_contains_english(str):
    my_re = re.compile(r'[A-Za-z]', re.S)
    res = re.findall(my_re, str)
    if len(res):
        return True
    else:
        return False


def tfidf(input_text):
    text = []
    for sentence in input_text:
        str = ""
        for i in sentence:
            str += i
            str += " "
        text.append(str)
    vectorizer = CountVectorizer()
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(text))
    text_weight = tf_idf.toarray()
    score = []
    for i in range(len(text)):
        print(f'''\r已经计算{i+1}/{len(text)}个tf-idf分数''', end='')
        aaa = [0]
        for j in input_text[i]:
            if is_contains_english(j):
                j = j.lower()
            if j in vectorizer.get_feature_names_out():
                aaa.append(text_weight[i][vectorizer.vocabulary_[j]])
            else:
                aaa.append(0)  # 这里不确定是不是0
        aaa.append(0)
        score.append(aaa)
    return score


def insert(start, end, max_insert_label, insert_mode, tf_idf):
    # [start, end)
    pos = []
    if insert_mode == 0:  # select the left part
        for z in range(max_insert_label):
            pos.append(z + start)
    elif insert_mode == 1:  # select the middle part
        start += (end - start - max_insert_label + 1) / 2
        start = int(start)
        for z in range(max_insert_label):
            pos.append(z + start)
    elif insert_mode == 2:  # select the right part
        for z in range(max_insert_label):
            pos.append(end - max_insert_label + z)
    elif insert_mode == 3:  # randomly select
        pos = np.random.choice(end - start, max_insert_label, replace=False) + start
        pos = pos.tolist()
        pos = sorted(pos)
    else:
        _, indices = torch.topk(torch.tensor(tf_idf), k=max_insert_label)
        indices += start
        pos = indices.tolist()
        pos = sorted(pos)
    return pos


def create_replaced_samples(input_ids_list, length_list, tf_idf_list):
    positions_list = []
    incorrect_input_ids_list = []
    label_ids_list = []
    target_ids_list = []
    original_positions_list = []
    original_input_ids_list = []
    for input_ids, length, tf_idf in zip(input_ids_list, length_list, tf_idf_list):
        for i in range(3):
            assert length >= 3
            # randomly draw a segment from input_ids
            sublen = random.randint(8, length)
            start_id = random.randint(0, length - sublen)
            end_id = start_id + sublen

            incorrect_input_ids = input_ids[start_id:end_id][:]
            incorrect_input_ids[0] = [tokenizer.bos_token_id]
            incorrect_input_ids[-1] = [tokenizer.eos_token_id]
            label_ids = [0]
            # insert
            pre_target_id = [[tokenizer.bos_token_id]]
            if start_id != 0:
                insert_label = min(1, start_id) + 1
                label_ids.append(insert_label)
                if start_id <= 1:
                    pre_target_id = input_ids[:start_id + 1]
                else:
                    pos = insert(1, start_id + 1, 1, 0, tf_idf[1:start_id + 1])
                    pre_target_id = [[tokenizer.bos_token_id]]
                    for p in pos:
                        pre_target_id.append(input_ids[p])
            if start_id == 0:
                label_ids += [0] * (sublen - 2)
            else:
                label_ids += [0] * (sublen - 3)
            pos_target_id = [[tokenizer.eos_token_id]]
            if end_id != length:
                insert_label = min(1, length - end_id) + 1
                label_ids.append(insert_label)

                if length - end_id <= 1:
                    pos_target_id = input_ids[end_id - 1:]
                else:
                    pos_target_id = []
                    pos = insert(end_id - 1, length, 1, 0, tf_idf[end_id - 1:length])
                    for p in pos:
                        pos_target_id.append(input_ids[p])
                    pos_target_id.append([tokenizer.eos_token_id])
            else:
                label_ids.append(0)

            target_ids = pre_target_id + input_ids[start_id + 1:end_id - 1] + pos_target_id
            incorrect_input_ids_list.append(incorrect_input_ids)
            label_ids_list.append(label_ids)
            target_ids_list.append(target_ids)
            # sample the number of replace tokens
            num_replace_tokens = max(1, int(0.15 * (sublen - 2)))
            if start_id == 0:
                replace_tokens_pos = np.random.choice(sublen - 2, num_replace_tokens, replace=False) + 1
            else:
                replace_tokens_pos = np.random.choice(sublen - 3, num_replace_tokens, replace=False) + 2

            replace_tokens_pos = replace_tokens_pos.tolist()
            replace_tokens_pos = sorted(replace_tokens_pos)
            # print(replace_tokens_pos)
            positions_list.append(replace_tokens_pos)

            original_positions_list.append([p + start_id for p in replace_tokens_pos])
            original_input_ids_list.append(input_ids)

        # construct 1 sentences only with replacement
        for j in range(2):
            # sample the number of replace tokens
            num_replace_tokens = max(1, int(0.15 * (length - 2)))
            replace_tokens_pos = np.random.choice(length - 2, num_replace_tokens, replace=False) + 1
            replace_tokens_pos = replace_tokens_pos.tolist()
            replace_tokens_pos = sorted(replace_tokens_pos)

            incorrect_input_ids = input_ids[:]
            label_ids = [0] * len(incorrect_input_ids)
            target_ids = input_ids[:]
            incorrect_input_ids_list.append(incorrect_input_ids)
            label_ids_list.append(label_ids)
            target_ids_list.append(target_ids)

            positions_list.append(replace_tokens_pos)

            original_positions_list.append(replace_tokens_pos[:])
            original_input_ids_list.append(input_ids)

    for incorrect_input_ids, label_ids, positions, target_ids in zip(incorrect_input_ids_list, label_ids_list,
                                                                     positions_list, target_ids_list):
        for p in positions:
            # random generate the replaced token id
            replaced_token_id = random.randint(0, tokenizer.vocab_size - 1)
            while replaced_token_id in [incorrect_input_ids[p], tokenizer.bos_token_id, tokenizer.eos_token_id,
                                        tokenizer.pad_token_id, tokenizer.mask_token_id]:
                replaced_token_id = random.randint(0, tokenizer.vocab_size - 1)
            incorrect_input_ids[p] = [replaced_token_id]
            label_ids[p] = 1
        assert len(incorrect_input_ids) == len(label_ids)
        assert sum([e if e > 1 else 1 for e in label_ids]) == len(target_ids)
    return incorrect_input_ids_list, label_ids_list, target_ids_list


def create_inserted_samples(input_ids_list=None, length_list=None, tf_idf_list=None):
    incorrect_input_ids_list = []
    label_ids_list = []
    target_ids_list = []
    for input_ids, length, tf_idf in zip(input_ids_list, length_list, tf_idf_list):
        for i in range(5):
        # for i in range(3):
            assert length > 3
            # sample the number of deleted tokens
            num_delete_tokens = random.randint(length // 1.5, length - 3)
            delete_tokens = np.random.choice(length - 2, num_delete_tokens, replace=False) + 1

            delete_tokens = delete_tokens.tolist()
            delete_tokens = sorted(delete_tokens)
            # add a token, so that while loop can end normally.
            # delete_tokens = [1,2,3,4,5,10,11,12,13,14,15,16]
            delete_tokens.append(100000)
            left_tokens = np.setdiff1d(np.arange(length), delete_tokens)
            left_tokens = left_tokens.tolist()

            label_ids = []
            target_ids = []
            j = 0
            for i in left_tokens:
                if i < delete_tokens[j]:  # copy
                    label_ids.append(0)
                else:
                    k = j
                    while i > delete_tokens[j]:
                        j += 1
                    # the blank is [k, j), so the number of deleted tokens is j-k
                    insert_label = min(1, j - k) + 1
                    label_ids.append(insert_label)

                    if j - k <= 1:
                        while k < j:
                            target_ids.append(input_ids[delete_tokens[k]])
                            k += 1
                    else:
                        # print(k,j, list(range(k,j)))
                        start = delete_tokens[k]
                        end = i
                        _tf_idf = tf_idf[start:end][:]
                        pos = insert(k, j, 1, 0, _tf_idf)

                        # print(pos)
                        for p in pos:
                            target_ids.append(input_ids[delete_tokens[p]])
                # add left tokens
                target_ids.append(input_ids[i])


            incorrect_input_ids_list.append([input_ids[p] for p in left_tokens])
            label_ids_list.append(label_ids)
            target_ids_list.append(target_ids)

            # print(incorrect_input_ids_list)
            # print(label_ids)
            # print(target_ids)
            # print(sum([e  if e > 1 else 1 for e in label_ids]),len(target_ids))
            assert len(incorrect_input_ids_list[-1]) == len(label_ids)
            assert sum([e if e > 1 else 1 for e in label_ids]) == len(target_ids)
    return incorrect_input_ids_list, label_ids_list, target_ids_list


# 对原始数据集进行预处理(original->preprocess)
def preprocess(input_file, output_file):
    # 定义需要获取的数据
    lengths = []  # 每句字数
    input_ids_list = []  # 每句中每个字的token id
    tf_idf_list = []  # 每句中每个字的词频-逆向文件频率
    # 读取segment.json
    with open(input_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
        segment = content['segment3']
        for line in segment:
            ids = [[tokenizer.bos_token_id]]
            for i in line:
                word_token = tokenizer.encode(i, add_special_tokens=True)
                ids.append(word_token[1:len(word_token)-1])
            ids.append([tokenizer.eos_token_id])
            lengths.append(len(ids))
            input_ids_list.append(ids)
        tf_idf_list = tfidf(segment)
    # 保存新数据为json文件
    data = {'lengths': lengths, 'input_ids_list': input_ids_list, 'tf_idf_list': tf_idf_list}
    with open(output_file, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))


# 对预处理数据进行合成(preprocess->synthetic)
def synthetic(input_file, output_file):
    start = time.time()
    j, index, total_len = 0, 0, 0
    # 定义需要获取的数据
    encoder_input_list = []
    encoder_label_list = []
    decoder_label_list = []
    # 读取预处理json文件中的数据
    with open(input_file, 'r', encoding='utf-8') as f:
        content = json.load(f)  # 获取预处理数据中的3个字段
        lengths = content['lengths']
        input_ids_list = content['input_ids_list']
        tf_idf_list = content['tf_idf_list']
    dataset_size = len(lengths)  # 数据集句子数
    batch_size = 1  # 一次处理的句子数,这里设为100
    # 获取合成数据
    funcs = [create_inserted_samples, create_replaced_samples]
    sub_input_ids = []
    sub_length = []
    sub_tf_idf = []
    for input_ids, length, tf_idf in zip(input_ids_list, lengths, tf_idf_list):
        index += 1
        j += 1
        sub_input_ids.append(input_ids)
        sub_length.append(length)
        sub_tf_idf.append(tf_idf)
        if j == batch_size or index == dataset_size:
            for f in funcs:
                encoder_input, encoder_label, decoder_label = f(input_ids_list=sub_input_ids,
                                                                length_list=sub_length,
                                                                tf_idf_list=sub_tf_idf)
                encoder_input_list += encoder_input
                encoder_label_list += encoder_label
                decoder_label_list += decoder_label
            total_len += j
            if total_len % 100 == 0:
                print(f'''\r已创建合成数据：{total_len}/{dataset_size}, 耗时：{time.time() - start:.2f}秒。''',
                      end='')
            if total_len >= dataset_size:
                print()
                break
            sub_input_ids = []
            sub_length = []
            sub_tf_idf = []
            j = 0
    # 保存数据到本地
    data = {'encoder_input_list': encoder_input_list,
            'encoder_label_list': encoder_label_list,
            'decoder_label_list': decoder_label_list}
    with open(output_file, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    # 加载BERT中文分词器
    tokenizer = BertTokenizer.from_pretrained('../checkpoints/bart-large-chinese')
    # 对验证集和训练集进行预处理
    modes = ['valid', 'train']
    for mode in modes:
        original_file = f'../dataset/original/{mode}_segment3.json'  # 原始数据路径
        preprocess_file = f'../dataset/preprocess/{mode}_preprocess.json'  # 预处理数据路径
        synthetic_file = f'../dataset/synthetic/{mode}_segment3_synthetic.json'  # 合成数据路径
        # original_file = f'../dataset/original/111.json'  # 原始数据路径
        # preprocess_file = f'../dataset/preprocess/ttt.json'  # 预处理数据路径
        # synthetic_file = f'../dataset/synthetic/ttt.json'  # 合成数据路径
        # print(f'开始对原始数据[{original_file}]进行预处理......')
        # preprocess(original_file, preprocess_file)  # 根据原始数据生成预处理数据
        # print(f'已根据原始数据[{original_file}]生成预处理数据[{preprocess_file}]。')
        print(f'开始利用预处理数据[{preprocess_file}]生成合成数据......')
        synthetic(preprocess_file, synthetic_file)  # 根据预处理数据生成合成数据
        print(f'已根据预处理数据[{preprocess_file}]生成合成数据[{synthetic_file}]。')
    print('=====数据处理完毕======')
