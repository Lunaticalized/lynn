import numpy as np 
import pandas as pd 

import torch
from torch.utils import data
from transformers import BertTokenizer
import torch.optim as optim

from bert_extract import BertExtractModel

train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv', keep_default_na=False)
test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv', keep_default_na=False)
sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

train = np.array(train_df)
test = np.array(test_df)

# remove empty entries


def find_start_end(input_str, sub_str):
    start = input_str.find(sub_str)
    end = start + len(sub_str)
    return start, end

# data shape: structured array (text, start, end)
train = list(map(lambda x: (x[1],) + find_start_end(x[1],x[2]), train))
train = np.array(train, dtype = [('text', 'U144'), ('start', 'i'), ('end', 'i')])

# build the dataset
# ignore sentiment label for now
BATCH_SIZE = 2
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def collate_tweets(sample):
    'given a list of samples, pad the encodings to the same length, and return the samples'
    sample = np.array(sample, dtype = [('text', 'U144'), ('start', 'i'), ('end', 'i')])
    max_len = max(list(map(len,sample['text'])))
    batch_encoding = tokenizer.batch_encode_plus(sample['text'], return_tensors='pt', pad_to_max_length=True, max_length = max_len)
    input_ids = batch_encoding.get('input_ids')
    start_pos = torch.tensor(sample['start'], dtype=torch.long)
    end_pos = torch.tensor(sample['end'], dtype=torch.long)
    return input_ids, start_pos, end_pos

sampler = data.RandomSampler(train)
loader = data.DataLoader(train, sampler=sampler, batch_size=BATCH_SIZE, collate_fn=collate_tweets)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = BertExtractModel.from_pretrained('bert-base-uncased')
# model.half()                    # floating point half precision
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=5e-5)

model.train()

num_epochs = 2
for epoch in range(num_epochs):
    running_loss = 0.0
    for step, batch in enumerate(loader):
        optimizer.zero_grad()

        batch = tuple(t.to(device) for t in batch)
        input_ids, start_pos, end_pos = batch
        outputs = model(input_ids, start_pos, end_pos)
        loss = outputs[0]

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 5 == 0:    # print every 5 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, running_loss / 5))
            running_loss = 0.0

print('finished training')
