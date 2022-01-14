"""
nlp, assignment 4, 2021

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""
import math

import torch
import torch.nn as nn
import torchtext
# from torchtext import data
from torchtext.legacy import data
import torch.optim as optim
from math import log, isfinite
from collections import Counter
import numpy as np
import sys, os, time, platform, nltk, random

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed=2512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    # torch.backends.cudnn.deterministic = True


# utility functions to read the corpus
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    # TODO edit the dictionary to have your own details
    return {'name': 'Ofir Ben Shoham', 'id': '208642496', 'email': 'benshoho@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()
    return sentence


def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {}  # transitions probabilities
B = {}  # emissions probabilities


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
     and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and should be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and  emmisionCounts

    Args:
      tagged_sentences: a list of tagged sentences, each tagged sentence is a
       list of pairs (w,t), as retunred by load_annotated_corpus().

   Return:
      [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
  """
    # TODO complete the code
    for sentence in tagged_sentences:
        prev_tag = START
        current_tag = None

        for word, tag in sentence:
            # allTagCounts  [tag]
            allTagCounts[tag] += 1

            # perWordTagCounts [word][tag]
            if word not in perWordTagCounts.keys():
                perWordTagCounts[word] = Counter()
            perWordTagCounts[word][tag] += 1

            # transitionCounts  [prev tag][current tag]
            if prev_tag not in transitionCounts.keys():
                transitionCounts[prev_tag] = Counter()
            transitionCounts[prev_tag][tag] += 1
            prev_tag = tag

            # emissionCounts [tag][word]
            if tag not in emissionCounts.keys():
                emissionCounts[tag] = Counter()
            emissionCounts[tag][word] += 1
            current_tag = tag

        if current_tag not in transitionCounts.keys():
            transitionCounts[current_tag] = Counter()
        transitionCounts[current_tag][END] += 1

    # Build A - the transmission matrix C(t(i-1), t) / C(t(i-1))
    # It means, transitionCounts(t(i-1),t(i)) / allTagCounts(t(i-1))
    for prev_tag in transitionCounts.keys():
        A[prev_tag] = {}
        for tag in transitionCounts[prev_tag].keys():
            A[prev_tag][tag] = log(transitionCounts[prev_tag][tag] / sum(transitionCounts[prev_tag].values()))

    # Build B - the emission matrix C(t(i), w(i)) / C(t(i))
    # It means, emissionCounts(t(i), w(i)) / allTagCounts(t(i))
    for tag in emissionCounts.keys():
        B[tag] = {}
        for word in emissionCounts[tag].keys():
            B[tag][word] = log(emissionCounts[tag][word] / sum(emissionCounts[tag].values()))

    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params(), perWordTagCounts [word][tag]
        allTagCounts (Counter): tag counts, as specified in learn_params(), allTagCounts  [tag]

        Return:
        list: list of pairs
    """

    tagged_sentence = []
    for word in sentence:
        if word in perWordTagCounts.keys():
            tag = max(perWordTagCounts[word], key=perWordTagCounts[word].get)
        else:
            tag = sample_from_distribution(allTagCounts)
        tagged_sentence.append((word, tag))
    return tagged_sentence


def sample_from_distribution(all_tag_counts):
    """

    Args:
        all_tag_counts (Counter): tag counts, as specified in learn_params(), allTagCounts  [tag]

    Returns:
        str: the sampled word

    """
    total_sum = sum(all_tag_counts.values())
    distribution_dict = {tag: all_tag_counts[tag] / total_sum for tag in all_tag_counts.keys()}
    word = random.choices(list(distribution_dict.keys()), weights=list(distribution_dict.values()), k=1)[0]
    return word


# ===========================================
#       POS tagging with HMM
# ===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): the HMM  emission probabilities.

    Return:
        list: list of pairs
    """

    # TODO complete the code
    tagged_sentence = []
    viterbi_result = viterbi(sentence, A, B)
    tags = retrace(viterbi_result)
    for word, tag in zip(sentence, tags):
        tagged_sentence.append((word, tag))
    return tagged_sentence


def viterbi(sentence, A, B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tuple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probability of the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): the HMM emission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtracking.

        """
    # Hint 1: For efficiency reasons - for words seen in training there is no
    #      need to consider all tags in the tagset, but only tags seen with that
    #      word. For OOV you have to consider all tags.
    # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
    #         current list = [ the dummy item ]
    # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END

    # TODO complete the code
    pre_list = [(START, None, 0.0)]  # (t,r,p)
    for word in sentence:
        temp_list = []
        if word in perWordTagCounts.keys():
            tags = perWordTagCounts[word].keys()
        else:
            tags = allTagCounts.keys()
        for tag in tags:
            next_best = predict_next_best(word, tag, pre_list, A, B)
            temp_list.append(next_best)
        pre_list = temp_list
    v_last = predict_next_best("", END, pre_list, A, B)
    return v_last


# a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    tags = []
    while end_item is not None:
        tag, r, _ = end_item
        tags.append(tag)
        end_item = r
    tags = [t for t in tags if t not in [START, END]]
    tags.reverse()
    return tags


# a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list, A, B):
    """Returns a new item (tuple)
    """
    next_best_item = tag, None, -math.inf  # Each cell ("item") is a tuple (tag,reference to prev,prob)
    for prev in predecessor_list:
        prev_tag, _, prev_prob = prev
        next_prob = prev_prob
        # A[prev_tag][tag], B[tag][word]
        if tag in A[prev_tag]:
            a_value = A[prev_tag][tag]
        else:
            a_value = 0  # TODO: change it
        if len(word) > 0 and word in B[tag].keys():
            b_value = B[tag][word]
        else:
            b_value = 0  # TODO: change it
        next_prob += a_value + b_value
        _, _, current_best_prob = next_best_item
        if next_prob > current_best_prob:
            next_best_item = (tag, prev, next_prob)
    return next_best_item


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): the HMM emission probabilities.
     """
    p = 0  # joint log prob. of words and tags

    # TODO complete the code
    prev_tag = START
    for word, tag in sentence:
        # A[prev_tag][tag], B[tag][word]
        a_value = A[prev_tag][tag]
        b_value = B[tag][word]
        p += a_value + b_value
        prev_tag = tag
    assert isfinite(p) and 0  # Should be negative. Think why!
    return p


# ===========================================
#       POS tagging with BiLSTM
# ===========================================

""" You are required to support two types of bi-LSTM:
    1. a vanilla biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""


# Suggestions and tips, not part of the required API
#
#  1. You can use PyTorch torch.nn module to define your LSTM, see:
#     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#  2. You can have the BLSTM tagger model(s) implemented in a dedicated class
#     (this could be a subclass of torch.nn.Module)
#  3. Think about padding.
#  4. Consider using dropout layers
#  5. Think about the way you implement the input representation
#  6. Consider using different unit types (LSTM, GRU,LeRU)


class VanillaBiLSTM(nn.Module):
    def __init__(self, params_d):
        super().__init__()
        self.params_d = params_d
        vocab, tags = get_vocab_and_tags(self.params_d['data_fn'], self.params_d['min_frequency'],
                                         self.params_d['max_vocab_size'])
        self.embeddings, embedded_tokens = self.get_embeddings(vocab)
        self.tags_to_id = {tag: tag_id for tag_id, tag in enumerate(tags)}
        self.words_to_id = {word.lower(): word_id for word_id, word in enumerate(embedded_tokens)}
        self.lstm = nn.LSTM(input_size=self.params_d['embedding_dimension'],
                            hidden_size=56,
                            num_layers=self.params_d['num_of_layers'],
                            bidirectional=True,
                            dropout=0.5, dtype=torch.double)
        self.linear1 = nn.Linear(56 * 2, self.params_d['output_dimension'],
                                 dtype=torch.double)  # it's 56*2 due to bidirectional
        #self.linear2 = nn.Linear(56, self.params_d['output_dimension'],  dtype=torch.double)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(100, affine=False)

    def get_embeddings(self, vocab_tokens):
        # vocab = torchtext.vocab.build_vocab_from_iterator(vocab_tokens)
        embeddings_dict = load_pretrained_embeddings(self.params_d['pretrained_embeddings_fn'], vocab_tokens)
        embeddings_dict['padding'] = np.random.randn(100)
        weights = np.array(list(embeddings_dict.values()))
        weights = torch.from_numpy(weights)
        # Embedding from vocab_size to 100
        # embeddings size: [vocab_size, 100]
        embeddings = nn.Embedding.from_pretrained(weights, padding_idx=len(weights) - 1)
        # embeddings = nn.Embedding.from_pretrained(weights)
        return embeddings, embeddings_dict.keys()

    def forward(self, text):
        words_id = list(map(lambda word: self.words_to_id[
            word.lower()] if word.lower() in self.words_to_id else self.embeddings.padding_idx, text))
        words_id_tensor = torch.tensor(words_id, dtype=torch.int)
        x = self.embeddings(words_id_tensor)[None]
        #x = self.relu(x)
        outputs, _ = self.lstm(x)
       #outputs = self.relu(outputs)
        # outputs = self.dropout(outputs)
        outputs = self.linear1(outputs)
        #outputs = self.linear2(outputs)
        return outputs

    def tags_to_ids(self, tags, words):
        tags_id = list(
            map(lambda t: self.tags_to_id[t[1]] if t[0].lower() in self.words_to_id else -1, zip(words, tags)))
        tags_id_tensor = torch.tensor(tags_id, dtype=torch.int)
        return tags_id_tensor

    def ids_to_tags(self, ids):
        ids_to_tags = {v: k for k, v in self.tags_to_id.items()}
        return [ids_to_tags[tag_id] for tag_id in ids]


def get_vocab_and_tags(data_fn, min_frequency, max_vocab_size):
    """
        max_vocab_size sets a constraints on the vocab dimension.
        If the its value is smaller than the number of unique
        tokens in data_fn, the words to consider are the most
        frequent words. If max_vocab_size = -1, all words
        occurring more that min_frequency are considered.
        min_frequency provides a threshold under which words are
        not considered at all. (If min_frequency=1 all words
        up to max_vocab_size are considered;
        If min_frequency=3, we only consider words that appear
        at least three times.)

    Args:
        data_fn (str):
        min_frequency (int): min_frequency provides a threshold under which words are not considered at all.
        max_vocab_size (int): max_vocab_size sets a constraints on the vocab dimension.

    Returns:
        List[str] : list with the vocab of the data_fn

    """
    sentences = load_annotated_corpus(data_fn)
    vocab_tokens = []
    tags_tokens = set()
    for sentence in sentences:
        for word, tag in sentence:
            vocab_tokens.append(word)
            tags_tokens.add(tag)
    vocab_counter = Counter(vocab_tokens)
    final_vocab = vocab_counter
    # TODO: check case that both max_vocab_size == -1 & max_vocab_size < len(vocab_counter). what to do in that case?
    # TODO: or max_vocab_size == -1 and min_frequency > 1
    if max_vocab_size == -1:
        # all words occurring more that min_frequency are considered
        final_vocab = {word: count for word, count in vocab_counter.items() if count > min_frequency}
    elif max_vocab_size < len(vocab_counter):
        # the words to consider are the most frequent words
        final_vocab = final_vocab.most_common(max_vocab_size)
    if min_frequency > 1:
        final_vocab = {word: count for word, count in final_vocab.items() if count >= min_frequency}
    return final_vocab, list(tags_tokens)


def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       thr returned dict is may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurrence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimension.
                            If the its value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occurring more that min_frequency are considered.
                        min_frequency provides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should specify default values.
    Return:
        a dictionary with the at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict
    """

    input_rep = params_d['input_rep']
    if input_rep == 0:
        # vanilla lstm
        model = VanillaBiLSTM(params_d)
    else:
        # TODO: add the the second case for 1 (case-base)
        model = None
        raise NotImplementedError("Should be implemented!!")
    return {'lstm': model, 'input_rep': input_rep}

    # no need for this one as part of the API
    # def get_model_params(model):
    """
    Returns a dictionary specifying the parameters of the specified model.
    This dictionary should be used to create another instance of the model.

    Args:
        model (torch.nn.Module): the network architecture

    Return:
        a dictionary, containing at least the following keys:
        {'input_dimension': int,
        'embedding_dimension': int,
        'num_of_layers': int,
        'output_dimension': int}
    """

    # TODO complete the code

    # return params_d


def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """
    vectors = {}
    # the idea of this function was taken from stackoverflow.
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.split()
            word = tokens[0]
            if vocab is not None and word in vocab or vocab is None:
                vectors[word] = np.array(tokens[1:], dtype=np.double)
    return vectors


def train_rnn(model, train_data, val_data=None):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
                            {'lstm': torch.nn.Module object, input_rep: [0|1]}
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()S
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """
    # Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider using batching
    # 4. some of the above could be implemented in helper functions (not part of
    #    the required API)

    # TODO complete the code
    # TODO: add ignore_index  to CrossEntropyLoss(..)
    epochs_num = 28
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # you can set the parameters as you like
    # vectors = load_pretrained_embeddings(pretrained_embeddings_fn)
    # model = model.to(device)
    criterion = criterion.to(device)
    lstm_model = model['lstm'].to(device)
    optimizer = optim.Adam(lstm_model.parameters())
    validation_iterator = None
    train_data = data_preprocessing(train_data)
    sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))
    if val_data is None:
        train_iterator = data.BucketIterator.splits(train_data, batch_size=32, device=device, sort_key=sort_key)
    else:
        val_data = data_preprocessing(val_data)
        train_iterator, validation_iterator = data.BucketIterator.splits((train_data, val_data), batch_sizes=(32, 32),
                                                                         device=device, sort_key=lambda x: x)
    for epoch in range(epochs_num):
        epoch_loss = 0
        train_iterator.create_batches()
        for inx, batch in enumerate(train_iterator.batches):
            # print(f'batch inx= {inx}/{len(train_iterator) // batch.batch_size}')
            optimizer.zero_grad()
            # words = batch.dataset['words']
            # tags = batch.dataset['tags']
            words, tags = temp_from_dict_to_words_tags(batch)
            y = lstm_model.tags_to_ids(tags, words)  # TODO: consider to change it.
            optimizer.zero_grad()
            pred = lstm_model(words)
            pred = pred.view(-1, pred.shape[-1])
            y = y.view(-1).long()
            # print(f'pred.dtype {pred.dtype}, y.dtype={y.dtype}')
            # print(f'pred.shape= {pred.shape}, y.shape= {y.shape}')
            loss = criterion(pred, y)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        print(f'avg loss = {epoch_loss / len(train_iterator)}, epoch={epoch + 1}')
        if validation_iterator is not None:
            validation_iterator.create_batches()
            result = count_correct_validation(model, validation_iterator)
            print(f'epoch= {epoch + 1}, count_correct={result}')


def temp_from_dict_to_words_tags(batch_lists):
    words = []
    tags = []
    for current_list in batch_lists:
        words.extend(current_list['words'])
        tags.extend(current_list['tags'])
        # words.extend(current_list['words'])
        # tags.extend(current_list['tags'])
    return words, tags


def count_correct_validation(model, val_data):
    all_predicted_tags = []
    all_tags = []
    """
        for batch in val_data.batches:
        words, tags = temp_from_dict_to_words_tags(batch)
        all_tags.extend(list(zip(words, tags)))
        # words = batch.dataset['words']
        # all_tags.extend(list(zip(words, batch.dataset['tags'])))
        all_predicted_tags.extend(rnn_tag_sentence(words, model))
    print(f'all_tags.len = {len(all_tags)}')

    """
    words, tags = temp_from_dict_to_words_tags(list(val_data.dataset.values()))
    all_tags.extend(list(zip(words, tags)))
    print(f'all_tags.len = {len(all_tags)}')
    all_predicted_tags.extend(rnn_tag_sentence(words, model))
    return count_correct(all_tags, all_predicted_tags)


def data_preprocessing(sentences):
    """

    Args:
        sentences (list(str)): a list of annotated sentences in the format returned by load_annotated_corpus()

    Returns:
        dict(int): dictionary in a format {{0: {'words': [..], 'tags':[..]}, .. }

    """
    dict_to_return = {}
    for inx, sentence in enumerate(sentences):
        words, tags = zip(*sentence)
        dict_to_return[inx] = {'words': words, 'tags': tags}
    return dict_to_return


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.

    Return:
        list: list of pairs
    """

    # TODO complete the code
    with torch.no_grad():
        lstm_model = model['lstm'].to(device)
        pred = lstm_model(sentence)
        pred = pred.view(-1, pred.shape[-1])
        pred_ids = pred.argmax(-1).tolist()
        pred_tags = lstm_model.ids_to_tags(pred_ids)
        return [(word, tag) for word, tag in zip(sentence, pred_tags)]
    # return tagged_sentence


def get_best_performing_model_params():
    """Returns a dictionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    # TODO complete the code

    return model_params


# ===========================================================
#       Wrapper function (tagging with a specified model)
# ===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model isteld and the input_rep flag (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): the HMM emission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)


    Return:
        list: list of pairs
    """
    if list(model.keys())[0] == 'baseline':
        return baseline_tag_sentence(sentence, list(model.values())[0], list(model.values())[1])
    if list(model.keys())[0] == 'hmm':
        return hmm_tag_sentence(sentence, list(model.values())[0], list(model.values())[1])
    if list(model.keys())[0] == 'blstm':
        return rnn_tag_sentence(sentence, list(model.values())[0])
    if list(model.keys())[0] == 'cblstm':
        return rnn_tag_sentence(sentence, list(model.values())[0])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correctly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    correct_count, correct_oov_count, oov_count = 0, 0, 0

    assert len(gold_sentence) == len(pred_sentence)

    for gold, pred in zip(gold_sentence, pred_sentence):
        gold_word, gold_tag = gold
        pred_word, pred_tag = pred
        if pred_tag == gold_tag:
            correct_count += 1
            if pred_word not in perWordTagCounts.keys():
                oov_count += 1
                correct_oov_count += 1
        else:
            if pred_word not in perWordTagCounts.keys():
                oov_count += 1
    return correct_count, correct_oov_count, oov_count
