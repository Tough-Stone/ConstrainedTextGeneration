import json
from src.transformers import BertTokenizer

mode = ['train', 'valid']
tokenizer = BertTokenizer.from_pretrained('../checkpoints/bart-large-chinese')

for m in mode:
    encoder_input_list = []
    encoder_label_list = []
    decoder_label_list = []
    input_file = f'../dataset/original/{m}_set.json'
    output_file = f'../dataset/synthetic/{m}_synthetic.json'
    with open(input_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
        n = 0
        for data in content:  # valid_set.json:2600条，train_set.json:10400条
            text, key_words = data['text'], data['key_words']
            # 获取关键词token
            key_words_token = [tokenizer.bos_token_id]
            for i in key_words:
                t = tokenizer.encode(i)
                del t[0], t[len(t) - 1]
                key_words_token += t
            key_words_token.append(tokenizer.eos_token_id)
            # 获取句子token
            text_token = [tokenizer.bos_token_id]
            last_position = 0
            position_start = []  # 关键词在句子中出现的起始位置
            position_end = []  # 关键词在句子中出现的结束位置
            for key_word in key_words:
                last_position = text.index(key_word, last_position)
                position_start.append(last_position)
                position_end.append(last_position+len(key_word))
            for i in range(len(position_start)-1):
                sub_string_0 = text[position_start[i]:position_end[i]]
                sub_string_1 = text[position_end[i]:position_start[i+1]]
                t = tokenizer.encode(sub_string_0)
                del t[0], t[len(t) - 1]
                text_token += t
                t = tokenizer.encode(sub_string_1)
                del t[0], t[len(t) - 1]
                text_token += t
            sub_string_0 = text[position_start[len(position_start)-1]:position_end[len(position_start)-1]]
            sub_string_1 = text[position_end[len(position_start)-1]:]
            t = tokenizer.encode(sub_string_0)
            del t[0], t[len(t) - 1]
            text_token += t
            t = tokenizer.encode(sub_string_1)
            del t[0], t[len(t) - 1]
            text_token += t
            text_token.append(tokenizer.eos_token_id)
            # 获取插入label
            label = []
            j = 0  # 原句token的下标
            for i in range(len(key_words_token)):
                if j >= len(text_token):
                    label.append(0)
                    continue
                if key_words_token[i] == text_token[j]:
                    label.append(0)
                else:
                    cnt = 0
                    while key_words_token[i] != text_token[j] and j < len(text_token):
                        j += 1
                        if j >= len(text_token):
                            break
                        cnt += 1
                    label.append(2)
                j += 1
            encoder_input_list.append(key_words_token)
            for i in range(5):
                encoder_label_list.append(label)
                decoder_label_list.append(text_token)
            n += 1
            print(f'''\r已处理句子数：{n}/{len(content)}。''', end='\n')
    data = {'encoder_input_list': encoder_input_list,
            'encoder_label_list': encoder_label_list,
            'decoder_label_list': decoder_label_list}
    with open(output_file, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))
