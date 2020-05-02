import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

class BertExtractModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertExtractModel, self).__init__(config)
        self.num_labels = 2
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids)
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

import numpy as np 
import pandas as pd 

import torch
from torch.utils import data
from transformers import BertTokenizer
import torch.optim as optim

# from bert_extract import BertExtractModel

train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv', keep_default_na=False)
test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv', keep_default_na=False)
sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

train = np.array(train_df)
train = train[0:1000]

# remove empty entries

# this is no good, need positions as token lists
def find_start_end(input_str, sub_str):
    start = input_str.find(sub_str)
    end = start + len(sub_str)
    return start, end

def find_start_end_token_pos(input_str, sub_str, tokenizer):
    input = tokenizer.encode(input_str)
    output = tokenizer.encode(sub_str)
    output = output[1:-1]       # get rid of special tokens

    matchpos = -1
    for pos in range(0, len(input) - len(output)):
        if input[pos:pos+len(output)] == output:
            matchpos = pos
    return matchpos, matchpos + len(output)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
# data shape: structured array (text, start, end)
train = list(map(lambda x: (x[1],) + find_start_end_token_pos(x[1],x[2],tokenizer), train))
train = list(filter(lambda x: x[1] != -1, train)) # discard the entries where there's no match
train = np.array(train, dtype = [('text', 'U144'), ('start', 'i'), ('end', 'i')])

# build the dataset
# ignore sentiment label for now
BATCH_SIZE = 32

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
# use_cuda = False
device = torch.device("cuda:0" if use_cuda else "cpu")

model = BertExtractModel.from_pretrained('bert-base-uncased')
# model.half()                    # floating point half precision
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=5e-5)

model.train()

num_epochs = 5
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

model.eval()
# try predict
# TODO the output is a lowercase string, convert back to input case?
# for now, ignore label
test = np.array(test_df['text'])
model.to(torch.device('cpu'))
ans = []
with torch.no_grad():
    for item in test[0:3]:
        input = tokenizer.encode(item)
        out = model(torch.tensor(input).unsqueeze(0))
        start_logits, end_logits, *_ = out
        start = np.argmax(start_logits.squeeze(0))
        end = np.argmax(end_logits.squeeze(0))
        ans.append(tokenizer.decode(input[start:end]))

ans = ["", "abc", "wtf!!", "", "yay"]
with open('ans.txt', 'w') as file:
    for item in ans:
        file.write('%s\n' % item)
