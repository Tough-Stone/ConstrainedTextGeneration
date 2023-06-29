import json

input_file = '../output/sixth.txt'
output_file = '../output/sixth.json'

id = []
text = []
with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i % 4 == 0:
            id.append(i // 4 + 1)
        if i % 4 == 3:
            text.append(line[20:].replace(' ', ''))

dict = []
for i in range(len(id)):
    d = {'id': id[i], 'text': text[i]}
    dict.append(d)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(json.dumps(dict, indent=4, ensure_ascii=False))
