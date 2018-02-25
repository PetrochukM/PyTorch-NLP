import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from lib.nn.attention import Attention
from lib.configurable import configurable
from lib.text_encoders import EOS_INDEX
from lib.text_encoders import SOS_INDEX
from lib.text_encoders import PADDING_INDEX
from lib.nn.lock_dropout import LockedDropout

logger = logging.getLogger(__name__)

# TODO: https://arxiv.org/pdf/1707.06799v1.pdf
# Consider CRF instead of softmax
# http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#bi-lstm-conditional-random-field-discussion

# TODO: Add an option for using the same length as the input as the max length for target
# Helps with marking


class SeqDecoder(nn.Module):
    r"""
    Provides functionality for decoding in a SeqToSeq framework, with an option for attention.

    Args:
        vocab (torchtext.vocab.Vocab):
            an object of torchtext.vocab.Vocab class

        embedding_size (int):
            the size of the embedding for the input

        rnn_size (int):
            The number of recurrent units. Based on Nils et al., 2017, we choose the
            default value of 128. <https://arxiv.org/pdf/1707.06799v1.pdf>.

        n_layers (numpy.int, int, optional):
            Number of RNN layers used in SeqToSeq. Based on Nils et
            al., 2017, we choose the default value of 2 as a "robust rule of thumb".
            <https://arxiv.org/pdf/1707.06799v1.pdf>

        rnn_cell (str, optional):
            type of RNN cell (default: gru)

        embedding_dropout (float, optional):
            dropout probability for the input sequence (default: 0)

        use_attention(bool, optional):
            Flag adds attention to the decoder. Attention is commonly used in SeqToSeq to attend to
            the encoder states. Based on wide community adoption, we recommend attention for
            SeqToSeq. <http://ruder.io/deep-learning-nlp-best-practices/index.html#fnref:27>
    """

    @configurable
    def __init__(self,
                 vocab_size,
                 embedding_size=100,
                 rnn_size=100,
                 n_layers=2,
                 rnn_cell='gru',
                 embedding_dropout=0,
                 rnn_dropout=0,
                 rnn_variational_dropout=0,
                 decode_dropout=0,
                 use_attention=True,
                 scheduled_sampling=False,
                 freeze_embeddings=False,
                 fixed_length=None,
                 include_vocab=None):  # Only used during unrolling
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = int(n_layers)
        self.rnn_size = int(rnn_size)
        self.use_attention = use_attention
        self.fixed_length = fixed_length
        self.include_vocab = include_vocab
        self.scheduled_sampling = scheduled_sampling
        self.rnn_size = rnn_size

        self.embedding = nn.Embedding(
            self.vocab_size, int(embedding_size), padding_idx=PADDING_INDEX)
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.rnn_dropout = LockedDropout(p=rnn_dropout)

        rnn_cell = rnn_cell.lower()
        if rnn_cell == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.rnn = self.rnn_cell(
            input_size=embedding_size,
            hidden_size=self.rnn_size,
            num_layers=self.n_layers,
            dropout=rnn_variational_dropout)

        if use_attention:
            self.attention = Attention(self.rnn_size)

        self.out = nn.Sequential(
            nn.Linear(rnn_size, rnn_size),
            nn.BatchNorm1d(rnn_size),
            nn.ReLU(),
            nn.Dropout(p=decode_dropout),
            nn.Linear(rnn_size, vocab_size))

    def _init_start_input(self, batch_size):
        """
        "<s>" returns a start of sequence token tensor for every sequence in the batch.

        Args:
            batch_size
        Returns:
            SOS (torch.LongTensor [1, batch_size]): Start of sequence token
        """
        init_input = Variable(torch.LongTensor(1, batch_size).fill_(SOS_INDEX), requires_grad=False)
        if torch.cuda.is_available():  # TODO: Fix by having a is_cuda parameter
            init_input = init_input.cuda()
        return init_input

    def _get_batch_size(self, target_output, encoder_hidden):
        """
        Utility function for getting the batch_size from target_output or encoder_hidden.

        Returns:
            (int) batch size
        """
        batch_size = 1
        if target_output is not None:
            batch_size = target_output.size(1)
        else:
            if self.rnn_cell is nn.LSTM:
                batch_size = encoder_hidden[0].size(1)
            elif self.rnn_cell is nn.GRU:
                batch_size = encoder_hidden.size(1)
        return batch_size

    def step(self, last_decoder_output, decoder_hidden, encoder_outputs):
        """
        Using last decoder output, decoder hidden, and encoder outputs predict the next token.

        Args:
            last_decoder_output (torch.LongTensor [output_len, batch_size]): variable containing the
                last decoder output
            decoder_hidden (tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
            encoder_outputs (torch.FloatTensor [seq_len, batch_size, rnn_size]): variable containing
                the encoded features of the input sequence
        Returns:
            predicted_softmax (torch.FloatTensor [batch_size, output_len, vocab_size]): variable containing the
                confidence for one token per sequence in the batch.
            decoder_hidden (tuple or tensor): variable containing the features in the hidden
                state dependant on torch.nn.GRU or torch.nn.LSTM
            attention (torch.FloatTensor [batch_size, output_len, input_len]): Attention weights on per token.
        """
        output_len = last_decoder_output.size(0)
        batch_size = last_decoder_output.size(1)
        rnn_size = self.rnn_size

        embedded = self.embedding(last_decoder_output)
        embedded = self.embedding_dropout(embedded)

        output, hidden = self.rnn(embedded, decoder_hidden)
        output = self.rnn_dropout(output)
        # (output_len, batch_size, rnn_size) -> (batch_size, output_len, rnn_size)
        output = output.transpose(0, 1).contiguous()

        attention_weights = None
        if self.use_attention:
            # Batch first encoder_outputs
            encoder_outputs = encoder_outputs.transpose(0, 1).contiguous()
            output, attention_weights = self.attention(output, encoder_outputs)

        # (batch_size, output_len, rnn_size) -> (batch_size * output_len, rnn_size)
        output = output.view(-1, rnn_size)
        # (batch_size * output_len, rnn_size) -> (batch_size * output_len, vocab_size)
        output = self.out(output)
        predicted_softmax = F.log_softmax(output)
        # (batch_size * output_len, vocab_size) -> (batch_size, output_len, vocab_size)
        predicted_softmax = predicted_softmax.view(batch_size, output_len, self.vocab_size)
        return predicted_softmax, hidden, attention_weights

    def _get_eos_indexes(self, predictions):
        """
        Args:
            decoder_output (torch.FloatTensor [batch_size, vocab_size]): decoder output for a single
                rnn pass
        Returns:
            (list) indexes of EOS tokens
        """
        eos_batches = predictions.data.view(-1).eq(EOS_INDEX).nonzero()
        if eos_batches.dim() > 0:
            return eos_batches.squeeze(1).tolist()
        else:
            return []

    def beam_search(self, n_beams, encoder_hidden, encoder_outputs, batch_size, fixed_length=None):
        """
        Using encoder hidden and encoder outputs make a prediction for the decoded sequence.
        This uses the decoder output to guess the next sequence.

        Args:
            n_beams (int): number of beams to search and return
            decoder_hidden (tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
            encoder_outputs (torch.FloatTensor [seq_len, batch_size, rnn_size]): variable containing
                the encoded features of the input sequence
            batch_size (int) size of the batch
        Returns:
            decoder_outputs (torch.FloatTensor [fixed_length - 1, batch_size, vocab_size]): outputs
                of the decoder at each timestep
            decoder_hidden (tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
            attention_weights (torch.FloatTensor [fixed_length - 1, batch_size, input_len]) attention
                weights for every decoder_output 
        """
        # TODO: Add a vocabulary filter depending on the current output
        # For each step, instead of picking max_1, we pick max topk
        batch_size = encoder_outputs.size(1)

        fixed_length = self.fixed_length if fixed_length is None else fixed_length
        decoder_hidden = None if self.use_attention else encoder_hidden
        decoder_input = self._init_start_input(batch_size)
        eos_tokens = set()
        # Start n_beams per row at 100% proability
        beam_probability = torch.FloatTensor(batch_size, n_beams).fill_(1.0)
        beam_output = torch.LongTensor(batch_size, n_beams).fill_(SOS_INDEX)

        while True:
            decoder_output, decoder_hidden, step_attention_weights = self.step(
                decoder_input, decoder_hidden, encoder_outputs)

            decoder_output = decoder_output.squeeze(1)
            if batch_size == decoder_output.shape()[0]:
                # decoder_output -> (batch_size, vocab_size)
                topk_output, topk_indices = decoder_output.topk(n_beams, dim=1)
                beam_probability *= topk_output  # Updated probability of each beam
                beam_output = torch.stack((beam_output, topk_indices))
                decoder_input = topk_output.view(1, batch_size * n_beams)
            elif batch_size * n_beams == decoder_output.shape()[0]:
                # (batch_size * n_beams, vocab_size) -> (batch_size, n_beams, vocab_size)
                decoder_output = decoder_output.view(batch_size, n_beams, self.vocab_size)
                # topk_output -> (batch_size, n_beams, n_beams)
                # topk_indices -> (batch_size, n_beams, n_beams)
                topk_output, topk_indices = decoder_output.topk(n_beams, dim=2)
                # For each output, multiply the probability of the beam prior with the probability
                # of the next token
                # topk_output -> (batch_size, n_beams, n_beams)
                topk_output = beam_probability * topk_output
                # topk_output -> (batch_size, n_beams * n_beams)
                topk_output = topk_output.view(batch_size, n_beams * n_beams)
                topk_indices = topk_indices.view(batch_size, n_beams * n_beams)
                # Get the next topk beams
                # beam_probability -> (batch_size, n_beams)
                # beam_indices -> (batch_size, n_beams)
                beam_probability, beam_indices = topk_output.topk(n_beams, dim=1)
                # TODO: Given the beam_indices, we need to wrap back to what the beam_output is
                # For each n_beams^2 beam_indices get the prior beam in n_beams
                beam_indices = beam_indices / n_beams
                beam_output = beam_output[:, beam_indices]
                new_output = topk_indices[:, beam_indices]
                beam_output = torch.stack((beam_output, new_output))

            # Check if every batch has an eos_token
            if fixed_length is None:  # CASE: Stop unrolling if EOS is found
                eos_tokens.update(self._get_eos_indexes(beam_output))
                if len(eos_tokens) == batch_size * n_beams:
                    break
                if beam_output.shape(0) == 1000:  # CASE: Something has probably gone wrong
                    logger.warn('Decoder has not predicted EOS in 1000 iterations. Breaking.')
                    break
            elif beam_output.shape(0) == fixed_length:  # CASE: Stop unrolling at a fixed length
                break

        return beam_output, beam_probability

    def unroll(self, encoder_hidden, encoder_outputs, batch_size, fixed_length=None):
        """
        Using encoder hidden and encoder outputs make a prediction for the decoded sequence.
        This uses the decoder output to guess the next sequence.

        Args:
            decoder_hidden (tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
            encoder_outputs (torch.FloatTensor [seq_len, batch_size, rnn_size]): variable containing
                the encoded features of the input sequence
            batch_size (int) size of the batch
            filter (callable): Given the current prediction filter the vocabulary 
        Returns:
            decoder_outputs (torch.FloatTensor [fixed_length - 1, batch_size, vocab_size]): outputs
                of the decoder at each timestep
            decoder_hidden (tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
            attention_weights (torch.FloatTensor [fixed_length - 1, batch_size, input_len]) attention
                weights for every decoder_output 
        """
        # https://arxiv.org/pdf/1609.08144.pdf
        # https://arxiv.org/abs/1508.04025
        fixed_length = self.fixed_length if fixed_length is None else fixed_length
        decoder_hidden = None if self.use_attention else encoder_hidden
        decoder_input = self._init_start_input(batch_size)
        decoder_outputs = []
        predictions = []
        eos_tokens = set()
        attention_weights = [] if self.use_attention else None
        cuda = lambda t: t.cuda() if encoder_outputs.is_cuda else t

        while True:
            decoder_output, decoder_hidden, step_attention_weights = self.step(
                decoder_input, decoder_hidden, encoder_outputs)

            # (batch_size, 1, vocab_size) -> (batch_size, vocab_size)
            decoder_output = decoder_output.squeeze(1)

            if self.include_vocab is not None:
                # mask the vocabulary before making the prediction
                batch_mask = []
                for i in range(batch_size):
                    # (batch_size, vocab_size) -> (vocab_size)
                    prediction_row = [p[i].data[0] for p in predictions]

                    # include_vocab_indices is an array of vocabulary indices to include
                    # (batch_size, vocab_size) -> (vocab_size)
                    include_vocab_indices = self.include_vocab(prediction_row)
                    if include_vocab_indices is None or len(include_vocab_indices) == 0:
                        # By default only include PADDING_INDEX
                        assert EOS_INDEX in prediction_row
                        include_vocab_indices = [PADDING_INDEX]

                    mask = torch.FloatTensor(self.vocab_size).fill_(float('inf'))
                    for vocab_index in include_vocab_indices:
                        mask[vocab_index] = 1
                    batch_mask.append(mask)

                decoder_output = decoder_output * Variable(
                    cuda(torch.stack(batch_mask)), requires_grad=False)

            decoder_outputs.append(decoder_output)
            # (batch_size, vocab_size) -> (batch_size)
            prediction = decoder_output.max(1)[1].view(batch_size)
            predictions.append(prediction)
            # Feed the predictions as the next decoder_input
            decoder_input = prediction.view(1, batch_size)
            if self.use_attention:
                # (batch_size, 1, input_len) -> (batch_size, input_len)
                step_attention_weights = step_attention_weights.squeeze(1)
                attention_weights.append(step_attention_weights)

            # Check if every batch has an eos_token
            if fixed_length is None:  # CASE: Stop unrolling if EOS is found
                eos_tokens.update(self._get_eos_indexes(prediction))
                if len(eos_tokens) == batch_size:
                    break
                if len(decoder_outputs) == 1000:  # CASE: Something has probably gone wrong
                    logger.warn('Decoder has not predicted EOS in 1000 iterations. Breaking.')
                    break
            elif len(decoder_outputs) == fixed_length:  # CASE: Stop unrolling at a fixed length
                break

        decoder_outputs = torch.stack(decoder_outputs)
        if self.use_attention:
            attention_weights = torch.stack(attention_weights)

        return decoder_outputs, decoder_hidden, attention_weights

    def forward(self,
                fixed_length=None,
                encoder_hidden=None,
                encoder_outputs=None,
                target_output=None):
        """
        Using encoder hidden and encoder outputs make a prediction for the decoded sequence

        Args:
            target_output (torch.LongTensor [N, batch_size]): tensor containing the target
                sequence
            encoder_outputs (torch.FloatTensor [seq_len, batch_size, rnn_size]): variable containing
                the encoded features of the input sequence
            encoder_hidden (tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
        Returns:
            decoder_outputs (torch.FloatTensor [N - 1, batch_size, vocab_size]): outputs
                of the decoder at each timestep
            decoder_hidden(tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
            attention_weights
        """
        if fixed_length is not None and fixed_length < 2:
            raise ValueError('Max length of 1 will only generate <s> token. Below 1, it not valid.')

        if self.use_attention and encoder_outputs is None:
            raise ValueError('Argument encoder_outputs cannot be None when attention is used.')

        batch_size = self._get_batch_size(target_output, encoder_hidden)

        if not self.scheduled_sampling and target_output is not None:
            decoder_input = torch.cat([self._init_start_input(batch_size), target_output[0:-1]])
            # https://arxiv.org/pdf/1609.08144.pdf
            # https://arxiv.org/abs/1508.04025
            decoder_hidden = None if self.use_attention else encoder_hidden
            decoder_outputs, decoder_hidden, attention_weights = self.step(
                decoder_input, decoder_hidden, encoder_outputs)
            # (batch_size, vocab_size, output_len) -> (output_len, batch_size, vocab_size)
            decoder_outputs = decoder_outputs.transpose(0, 1).contiguous()
            if self.use_attention:
                attention_weights = attention_weights.transpose(0, 1).contiguous()
            return decoder_outputs, decoder_hidden, attention_weights

        return self.unroll(encoder_hidden, encoder_outputs, batch_size, fixed_length=fixed_length)
