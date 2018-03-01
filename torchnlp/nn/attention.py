import torch
import torch.nn as nn

# TODO: Credit IBM for this snipit


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data:
            :math:`y = Ax + b`.
    """

    def __init__(self, dimensions, attention_type='general'):
        """
        Args:
            dimensions (int): The number of expected features in the output
        """
        super(Attention, self).__init__()

        self.attention_type = attention_type
        assert (self.attention_type in ["dot", "general"]), "Invalid attention type selected."

        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()

    def forward(self, input_, context):
        """
        Args:
            input_ (torch.FloatTensor [batch_size, output_len, dimensions]): the attention input.
            context (torch.FloatTensor [batch_size, input_len, dimensions]): tensor containing
                features of the encoded input sequence.
        Returns:
            output (torch.LongTensor [batch_size, output_len, dimensions]): tensor containing the
                attended output features.
            attention_weights (torch.FloatTensor [batch_size, output_len, input_len]): tensor
                containing attention weights.
        """
        batch_size, output_len, dimensions = input_.size()
        input_len = context.size(1)

        if self.attention_type == "general":
            input_ = input_.view(batch_size * output_len, dimensions)
            input_ = self.linear_in(input_)
            input_ = input_.view(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, input_len, dimensions) ->
        # (batch_size, output_len, input_len)
        attention_scores = torch.bmm(input_, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, input_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, input_len)

        # (batch_size, output_len, input_len) * (batch_size, input_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, input_), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights
