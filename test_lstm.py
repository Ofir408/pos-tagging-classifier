import torch

import tagger
from tagger import VanillaBiLSTM, train_rnn

params_dict = {'max_vocab_size': 100000,
               'min_frequency': 1,
               'input_rep': 0,
               'embedding_dimension': 100,
               'num_of_layers': 2,
               'output_dimension': 17,
               'pretrained_embeddings_fn': 'glove.6B.100d.txt',
               'data_fn': 'en-ud-train.upos.tsv'
               }
model = VanillaBiLSTM(params_dict)
# model_input = torch.tensor([["hello", 'world']])
# print(model(model_input))

embedding_dict_sample = {'hello': 1, 'world': 2, 'other': 2}
# t = torch.tensor([embedding_dict_sample[x.item()] for x in model_input])

# train
train_data = tagger.load_annotated_corpus(params_dict['data_fn'])
tagger.learn_params(train_data)
model_dict = tagger.initialize_rnn_model(params_dict)
dev_data = tagger.load_annotated_corpus('en-ud-dev.upos.tsv')
train_rnn(model_dict, train_data, dev_data)
result = tagger.rnn_tag_sentence(
    'Nervous people make mistakes , so I suppose there will be a wave of succesfull arab attacks .'.split(), model_dict)
print(result)

