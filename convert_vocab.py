
import json
vocab = json.load(open(r'D:\TaoVm\vocab.json', 'r', encoding='utf-8'))
print('Vocab size:', len(vocab))
# Replace multi-char entries (pad, unk) with special codes
with open(r'D:\TaoVm\vocab.txt', 'w', encoding='utf-8') as f:
    f.write(str(len(vocab)) + '\n')
    for ch in vocab:
        if len(ch) == 1:
            f.write(str(ord(ch)) + '\n')
        elif ch == '<pad>':
            f.write('-1\n')  # special
        elif ch == '<unk>':
            f.write('-2\n')
        else:
            print('Unknown:', repr(ch))
            f.write('-3\n')
print('done')
