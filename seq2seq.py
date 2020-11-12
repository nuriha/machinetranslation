#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way.
"""


from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
import math
import numpy as np
from io import open
import re
import unicodedata
import string

import matplotlib
#if you are running on the gradx/ugradx/ another cluster,
#you will need the following line
#if you run on a local machine, you can comment it out
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid,
# it can be very easy to confict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"
Padding_token = "<P>"

SOS_index = 0
EOS_index = 1
Padding_index = 2
MAX_LENGTH = 15


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token, Padding_index: Padding_token}
        self.n_words = 3  # Count SOS and EOS and padding

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"),
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################

class EncoderRNN(nn.Module):
    """the class for the enoder RNN
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.padding_idx = Padding_index
        self.wordEmbeddings = nn.Embedding(self.input_size, self.hidden_size, self.padding_idx)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        #implement padding of some sort


    def forward(self, input, hidden):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        encoder_padding_mask = input.eq(self.padding_idx).t() #?? idk what this means
        seq_len = torch.sum(encoder_padding_mask, dim=-1).squeeze(-1)
        sorted_len, sorted_idx = seq_len.sort(0, descending=False)
        seqlen, batchsize = input.size()
        realinput = input.detach().clone()
        if batchsize > 1:
            for i in range(batchsize):
                realinput[:,i] = input.t()[sorted_idx[i]]
        else:
            realinput = input
            sorted_len = sorted_len.view(-1)
            #print(input.size())
        sorted_len = seqlen - sorted_len
        embedding = self.wordEmbeddings(realinput)
        #if batchsize == 1:
            #print(sorted_len)
            #print(batchsize)
            #print(embedding.size())
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedding, sorted_len)
        size = 1, batchsize, self.hidden_size
        h0 = embedding.new_zeros(*size)
        c0 = embedding.new_zeros(*size)
        packed_output, (hidden_final, cell_final) = self.lstm(packed_embedding, (h0, c0))
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, padding_value=self.padding_idx,total_length=seqlen)
        return (output, hidden_final, cell_final, encoder_padding_mask.t())

    def get_initial_hidden_state(self, batchsize):
        return torch.zeros(1, batchsize, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    """the class for the decoder
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.dropout = nn.Dropout(self.dropout_p)
        self.padding_idx = Padding_index
        self.wordEmbeddings = nn.Embedding(self.output_size, self.hidden_size, self.padding_idx)
        #self.embed = nn.Embedding(self.hidden_size, self.output_size, self.padding_idx)
        #self.encoder_hidden = Linear(self.hidden_size, self.hidden_size)
        #self.encoder_cell = Linear(self.hidden_size, self.hidden_size)
        self.input_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.log_softmax = nn.LogSoftmax()

        #self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        #self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size*2, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size, self.dropout)

    def forward(self, input, encoder_outputs, train):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights

        Dropout (self.dropout) should be applied to the word embeddings.
        """
        encoder_outs = encoder_outputs[0]
        encoder_hiddens = encoder_outputs[1]
        encoder_cells = encoder_outputs[2]
        encoder_padding_mask = encoder_outputs[3]

        srclen = encoder_outs.size(0)
        seqlen, batchsize = input.size()
        embedding = self.wordEmbeddings(input)
        embedding = self.dropout(embedding)
        prev_hidden = [encoder_hiddens[0]]
        prev_cell = [encoder_cells[0]]
        len = seqlen
        if not train:
            len = MAX_LENGTH

        input_feed = embedding.new_zeros(batchsize, self.hidden_size)
        attn_scores = embedding.new_zeros(srclen, len, batchsize)
        outs = []

        for di in range(len):
            if input_feed is not None:
                if batchsize == 1:
                    print(embedding.size())
                    print(input_feed.size())
                lstm_input = torch.cat((embedding[di,:,:], input_feed), dim=1)
            else:
                lstm_input = embedding[di]
            hidden, cell = self.lstm(lstm_input, (prev_hidden[0], prev_cell[0]))
            attn_out = self.input_proj(hidden)
            attn_score = (encoder_outs * attn_out.unsqueeze(0)).sum(dim=2)
            attn_score = (attn_score.masked_fill(encoder_padding_mask, float("-inf")))
            attn_score = F.softmax(attn_score, dim = 0)
            attn_scores[:,di,:] = attn_score
            attn_out = (attn_score.unsqueeze(2) * encoder_outs).sum(dim = 0)
            attn_out = torch.tanh(self.output_proj(torch.cat((attn_out, hidden), dim = 1)))
            attn_out = self.dropout(attn_out)
            prev_hidden[0] = hidden
            prev_cell[0] = cell
            if input_feed is not None:
                input_feed = attn_out
            outs.append(attn_out)
        x = torch.cat(outs, dim =0).view(len, batchsize, self.hidden_size) #si from paper?
        x = x.transpose(1,0)
        x = self.dropout(x)
        log_softmax = self.log_softmax(x)
        attn_scores = attn_scores.transpose(0,2)
        return log_softmax, attn_out, attn_scores

    def get_initial_hidden_state(self, batchsize):
        return torch.zeros(1, batchsize, self.hidden_size, device=device)


######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, tgt_vocab_size, max_length=MAX_LENGTH):
    #hard code batch size here:
    batchsize = 4
    encoder_hidden = encoder.get_initial_hidden_state(batchsize)
    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()
    input_length = input_tensor.size()[0]
    target_length = target_tensor.size()[0]

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    #for ei in range(input_length):
        #input to encoder is different - give batch of sequence
        #need to pad tensors to make them all same size, use binary mask to figure out where padding ends
    encoder_outputs = encoder(input_tensor, encoder_hidden)
    #encoder_outputs[ei] = encoder_output[0, 0]

    #decoder_input = torch.tensor([[SOS_index]], device=device)
    #decoder_hidden = encoder_hidden
    decoder_attentions = torch.zeros(max_length, max_length)
    decoder_output, decoder_hidden, decoder_attention = decoder(target_tensor, encoder_outputs, True)
    #change decoder output to 2d

    #decodeoutput = decoder_output.view(batchsize * target_length, 1362)
    l = nn.Linear(decoder.hidden_size, tgt_vocab_size)
    decodeoutput = l(decoder_output).view(batchsize * target_length, -1)
    target_tensor = target_tensor.view(batchsize * target_length)
    #print(decodeoutput.size())
    #print(target_tensor.size())
    loss += criterion(decodeoutput, target_tensor)

    loss.backward()
    optimizer.step()
    #return 0
    return loss.item()

######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """
    batchsize = 1

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state(batchsize)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        #for ei in range(input_length):
        #print(input_tensor)
        encoder_outputs = encoder(input_tensor, encoder_hidden)
            #encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_index]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            print(decoder_input.size())
            #print(encoder_outputs.size())
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, encoder_outputs, False)
            print(decoder_output.size())
            print(decoder_hidden.size())
            print(decoder_attention.size())
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions):
    """visualize the attention mechanism. And save it to a file.
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """

    "*** YOUR CODE HERE ***"
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=5000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss(ignore_index=Padding_index)

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every
    mybatch = 4
    while iter_num < args.n_iters:
        iter_num += mybatch
        #max_input_len = 0
        #max_target_len = 0
        in_tens = [0 for i in range(mybatch)]
        tar_tens = [0 for i in range(mybatch)]
        for i in range(mybatch):
            training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))
            in_tens[i] = training_pair[0].squeeze(1)
            #print(training_pair[0].size())
            #max_input_len = max(max_input_len, in_tens[i].size())
            tar_tens[i] = training_pair[1].squeeze(1)
            #max_target_len = max(max_target_len, tar_tens[i].size())
        #input_tensor = torch.stack(in_tens)
        #target_tensor = torch.stack(tar_tens)
        #print(in_tens)
        input_tensor = nn.utils.rnn.pad_sequence(in_tens, padding_value=Padding_index, batch_first=False)
        #print(input_tensor.size())
        target_tensor = nn.utils.rnn.pad_sequence(tar_tens, padding_value=Padding_index, batch_first=False)
        '''for input, target in zip(input_tensor, target_tensor):
            if input.size() != max_input_len:
                input = nn.functional.pad(input, pad=(input.size(),max_input_len), value=Padding_token)
            if target.size() != max_target_len:
                target = nn.functional.pad(target, pad=(target.size(), max_target_len), value=Padding_token)
        '''

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion, tgt_vocab.n_words)
        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab)


if __name__ == '__main__':
    main()
