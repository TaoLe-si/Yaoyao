
import pyarrow.parquet as pq
import json

train_t = pq.read_table(r'D:\TaoVm\tinystories_train_0.parquet')
val_t = pq.read_table(r'D:\TaoVm\tinystories_val_0.parquet')

print('train rows:', train_t.num_rows)
print('val rows:', val_t.num_rows)

# Extract text - just write to a single .txt file
train_texts = train_t['text'].to_pylist()
val_texts = val_t['text'].to_pylist()

# Show samples
print('\n=== Train samples ===')
for i in [0, 100, 1000, 10000]:
    print(f'[{i}]', train_texts[i][:200])

print('\n=== Char stats ===')
all_text = '\n'.join(train_texts + val_texts)
chars = sorted(set(all_text))
print(f'Total chars: {len(all_text):,}')
print(f'Unique chars: {len(chars)}')
print(f'Chars: {"".join(chars[:60])}')
print(f'Tail: {"".join(chars[60:])}')

# Save as txt
with open(r'D:\TaoVm\tinystories_train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_texts))
with open(r'D:\TaoVm\tinystories_val.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(val_texts))

# Save vocab
vocab = ['<pad>', '<unk>'] + chars
with open(r'D:\TaoVm\vocab.json', 'w') as f:
    json.dump(vocab, f)

print(f'\nWrote train.txt, val.txt, vocab.json (V={len(vocab)})')
