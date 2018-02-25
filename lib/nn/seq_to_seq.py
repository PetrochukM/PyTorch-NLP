import torch
import torch.nn as nn

from lib.configurable import configurable
from lib.nn.seq_encoder import SeqEncoder
from lib.nn.seq_decoder import SeqDecoder

# TODO: Encoder and decoder are dependent on each other resolve that by not taking them as inputs


class SeqToSeq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.
    """

    @configurable
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 embedding_size=100,
                 rnn_size=100,
                 n_layers=2,
                 rnn_cell='gru',
                 tie_weights=False,
                 include_vocab=None):
        """
        Args:
            encoder (EncoderRNN): object of EncoderRNN
            decoder (DecoderRNN): object of DecoderRNN
        """
        super(SeqToSeq, self).__init__()

        self.encoder = SeqEncoder(
            input_vocab_size,
            embedding_size=embedding_size,
            rnn_size=rnn_size,
            n_layers=n_layers,
            rnn_cell=rnn_cell)
        self.decoder = SeqDecoder(
            output_vocab_size,
            embedding_size=embedding_size,
            rnn_size=rnn_size,
            n_layers=n_layers,
            rnn_cell=rnn_cell,
            include_vocab=include_vocab)

        if tie_weights:
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, source, source_lengths, target=None, target_lengths=None):
        """
        Args:
            batch (torchtext.data.Batch): Batch object with an input and output field containing the
                encoded features of the input sequence and output sequence
        Returns:
            decoder_outputs (torch.FloatTensor [max_length - 1, batch_size, rnn_size]): outputs
                of the decoder at each timestep
            decoder_hidden(tuple or tensor): variable containing the features in the hidden state
                dependant on torch.nn.GRU or torch.nn.LSTM
            ret_dict (dict): dictionary containing additional information as follows: {
                *KEY_LENGTH* : list of integers representing lengths of output sequences,
                *KEY_SEQUENCE* : list of sequences, where each sequence is a list of predicted token
                    IDs,
                *KEY_INPUT* : target outputs if provided for decoding,
                *KEY_ATTN_SCORE* : list of sequences, where each list is of attention weights}
        """
        encoder_outputs, encoder_hidden = self.encoder(source, source_lengths)

        # NOTE: Decoder is set predict to target length due in order to have the same matrix size
        # when computing the loss function.
        if target is not None:
            max_length = torch.max(target_lengths)
        else:
            max_length = None

        return self.decoder(
            fixed_length=max_length,
            target_output=target,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs)
