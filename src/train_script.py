import torch.nn as nn
import transformers
from transformers import DistilBertPreTrainedModel, DistilBertModel
import torch.nn.functional as F
from itertools import chain

class BertExtractModel(nn.Module):
    def __init__(self):
        super(BertExtractModel, self).__init__()
        self.num_labels = 4     # pos-start, pos-end, neg-start, neg-end
        self.bert = DistilBertModel.from_pretrained("../input/distilbertrelated/")
        self.logit = nn.Linear(self.bert.config.dim, self.num_labels, bias=False)
        
    def forward(self, sentiments=None,
                input_ids=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids)
        sequence_output = outputs[0]
        senti_index = torch.stack((2 * sentiments, 2 * sentiments + 1), dim=1)

        logits = torch.stack([ torch.index_select(b, 1, senti_index[i]) for i, b in enumerate(torch.unbind(self.logit(sequence_output)))], dim=0)
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
from transformers import DistilBertTokenizer
import torch.optim as optim

train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv', keep_default_na=False)
test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv', keep_default_na=False)
sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

train_df = train_df[train_df['sentiment'] != 'neutral'] # ignore neutral ones

sentiment_dict = {'positive': 0, 'neutral' : 2, 'negative': 1}
train_df['sentiment'] = [sentiment_dict[x] for x in train_df['sentiment']]
test_df['sentiment'] = [sentiment_dict[x] for x in test_df['sentiment']]

train = np.array(train_df)

def find_start_end_token_pos(input_str, sub_str, tokenizer):
    input = tokenizer.encode(input_str)
    output = tokenizer.encode(sub_str)
    output = output[1:-1]       # get rid of special tokens

    matchpos = -1
    for pos in range(0, len(input) - len(output)):
        if input[pos:pos+len(output)] == output:
            matchpos = pos
    return matchpos, matchpos + len(output)

tokenizer = DistilBertTokenizer.from_pretrained("../input/distilbertrelated/")
        
# data shape: structured array (text, start, end)
train = list(map(lambda x: (x[0],x[1],x[3]) +
                 find_start_end_token_pos(x[1],x[2],tokenizer), train))
train = list(filter(lambda x: x[3] != -1, train)) # discard the entries where there's no match
DATA_TYPE = [('id', 'U10'),
             ('text', 'U144'),
             ('sentiment', 'i'),
             ('start', 'i'),
             ('end', 'i')]
train = np.array(train, dtype = DATA_TYPE)

# build the dataset
# ignore sentiment label for now
BATCH_SIZE = 128

def collate_tweets(sample):
    'given a list of samples, pad the encodings to the same length, and return the samples'
    sample = np.array(sample, dtype = DATA_TYPE)
    max_len = max(list(map(len,sample['text'])))
    batch_encoding = tokenizer.batch_encode_plus(sample['text'],
                                                 return_tensors='pt',
                                                 pad_to_max_length=True,
                                                 max_length=max_len)
    sentiments = torch.tensor(sample['sentiment'], dtype=torch.long)
    input_ids = batch_encoding.get('input_ids')
    start_pos = torch.tensor(sample['start'], dtype=torch.long)
    end_pos = torch.tensor(sample['end'], dtype=torch.long)
    return sentiments, input_ids, start_pos, end_pos

use_cuda = torch.cuda.is_available()

NUM_EPOCHS = 4

# # for debug
# train = train[0:10]
# use_cuda = False
# BATCH_SIZE = 1
# NUM_EPOCHS = 1

device = torch.device("cuda:0" if use_cuda else "cpu")

model = BertExtractModel()
# model.half()                    # floating point half precision
model.to(device)

sampler = data.RandomSampler(train)
loader = data.DataLoader(train, sampler=sampler, batch_size=BATCH_SIZE, collate_fn=collate_tweets)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.3,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
lr = 5e-5
num_training_steps = 1000
num_warmup_steps = 100
warmup_proportion = float(num_warmup_steps) / float(num_training_steps)
optimizer_whole = transformers.AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer_whole, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

addon_params = chain(model.logit.parameters())

optimizer_logit = optim.SGD(addon_params, lr=1e-3, momentum = 0.9)
model.train()

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for step, batch in enumerate(loader):
        optimizer = optimizer_logit
        if step % 2 == 1 or epoch > 2:
            optimizer = optimizer_whole
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        senti, input_ids, start_pos, end_pos = batch
        outputs = model(senti, input_ids, start_pos, end_pos)
        loss = outputs[0]

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 10 == 9:    # print every n mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, running_loss / 10))
            running_loss = 0.0

print('finished training')

model.eval()
# try predict
# TODO the output is a lowercase string, convert back to input case?
# for now, ignore label
model.to(torch.device('cpu'))
ans = []

import re
def back_to_cased(selected, text):
    m = re.search(selected, text, re.IGNORECASE)
    if m is not None:
        return m[0]
    else:
        raise ValueError(selected, text)

def clean_input(text):
    # strip URLs
    urlpat = re.compile(r'https?:\/\/.*\b')
    text = urlpat.sub('', text)
    # strip underscore stuff
    pat = re.compile(r'@?_.*?\b')
    text = pat.sub('', text)
    return text.strip()

with torch.no_grad():
    for i, row in test_df.iterrows():
        input_text = clean_input(row['text'])
        input = tokenizer.encode(input_text)
        senti = row['sentiment']
        if senti == 2:          # neutral just predict the whole thing
            ans.append(input_text)
        else:
            out = model(torch.tensor(senti).unsqueeze(0), torch.tensor(input).unsqueeze(0))
            start_logits, end_logits, *_ = out
            start = np.argmax(start_logits.squeeze(0))
            end = np.argmax(end_logits.squeeze(0))
            ans.append(tokenizer.decode(input[start:end]))
        if i % 50 == 0:
            print('testing: ', i, '/', len(test_df))

sub_df['selected_text'] = ans
sub_df.to_csv("submission.csv", index=False)

print('done')
