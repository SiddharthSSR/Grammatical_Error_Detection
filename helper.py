import torch
import operator
import datetime

import numpy as np
from collections import Counter
from sklearn.metrics import recall_score, precision_score, f1_score


def get_text_length(text):
    split = text.split(" ")
    length = len(split)
    return length

def get_freq_words(sentences):
    sentence_list = list(sentences)
    word_count = Counter()

    for sent in sentence_list:
        split = sent.split(' ')
        word_count += Counter(split)
    return word_count

def get_uniq_words(word_count):
    sorted_dict = sorted(word_count.items(), key=operator.itemgetter(1))
    sorted_dict = dict(sorted_dict)
    uniq_words = list(sorted_dict.keys())
    freq_words = list(sorted_dict.values())
    return uniq_words

def get_most_freq(uniq_words):
    sort_reverse=[]
    for i in range(1,101):
        reverse = uniq_words[-i]
        sort_reverse.append(reverse)
    return sort_reverse



def encode(tokenizer, sentences, labels, test=True):
    """
    
    """
    # Tokenize all of the sentences and map the tokens to their word Ids
    input_ids = []
    attention_masks = []

    for sent in sentences:

        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.

        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 128,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensor
    if not(test):
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        return input_ids, attention_masks, labels
    else:
        return input_ids, attention_masks



def get_param_info(model, params):
    """
    
    """
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))



def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_precision(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return precision_score(labels_flat, pred_flat)

def flat_recall(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return recall_score(labels_flat, pred_flat)

def flat_f1(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))




