import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dict_dim, hidden_size=1000, embedding_dim=620):

        """
        input_dict_dim = longueur du dictionnaire du langage source K
        hidden_size = nombre d'états cachés n
        embedding_dim = dimension m de l'espace de représentation des mots du dictionnaire
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_dict_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, bidirectional=True)

    def forward(self, src):
        """
        :return: outputs (the hiddens states)
        :rtype: Tensor of shape (src_len, batch_size, hidden_size*2),
        """
        embedded = self.embedding(src)
        # embedded shape = (batch_size, len_sequence, embedding size)
        outputs, hidden = self.rnn(embedded)
        # outputs shape = (batch_size, len_sequence, 2*hidden size)

        # output shape : (num_layers * num_directions, batch_size, hidden_size)
        # outputs are always from the last layer
        assert tuple(outputs.shape) == (src.shape[0], src.shape[1], self.hidden_size*2)
        return outputs



class Attention(nn.Module):
    def __init__(self, hidden_size=1000, hidden_unit=1000):
        super().__init__()

        self.attn_Ua = nn.Linear((hidden_size*2), hidden_unit)
        self.attn_Wa = nn.Linear(hidden_size, hidden_unit)
        self.v = nn.Linear(hidden_unit, 1, bias=False)

    def forward(self, encoder_hiddens, last_dec_hidden):
        """
        :param encoder_hiddens: all the hidden states of the encoder
        :shape: (src_len, batch_size, hidden_size*2)
        :param last_dec_hidden: the last state of the decoder, s_{i-1}
        :shape: (batch_size, hidden_size)

        :return: The softmax output of how well the hidden states of the
            encoder {encoder_hiddens} "matches" with the last decoder states
            {last_dec_hidden}.
        :shape:  (batch_size, src_len)
        """
        batch_size = encoder_hiddens.shape[1]
        src_len = encoder_hiddens.shape[0]

        output_enc = self.attn_Ua(encoder_hiddens)

        # For each step t we "compare" the state encoder_hiddens[:, t] with
        # last_dec_hidden. As such, to add the two results of the linear
        # layers, their inputs must have the same shape
        last_dec_hidden = last_dec_hidden.unsqueeze(0).repeat(src_len, 1,1)
        output_dec = self.attn_Wa(last_dec_hidden)

        output = output_enc + output_dec
        energy = torch.tanh(output)

        attention = self.v(energy).squeeze(2)
        attention = attention.permute(1,0)

        r = F.softmax(attention, dim=1)
        assert tuple(r.shape) == (batch_size,src_len), r.shape
        return r


class Decoder(nn.Module):
    def __init__(self, output_dict_size, attention, hidden_size=1000, embedding_dim=620):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dict_size = output_dict_size
        self.attention = attention

        self.embedding = nn.Embedding(output_dict_size, embedding_dim)
        self.rnn = nn.GRU((hidden_size * 2) + embedding_dim, hidden_size)

        self.fc_out = nn.Linear((hidden_size * 3) + embedding_dim, output_dict_size)

    def forward(self, input_, last_dec_hidden, encoder_hiddens):
        """
        :param input_: the last word predicted by the decoder (or SOS at the
            start)
        :shape: (batch_size)
        :param last_dec_hidden: the last state of the decoder, s_{i-1}
        :shape: (batch, hidden_size)
        :param encoder_hiddens: all the hidden states of the encoder
        :shape: (src_len, batch, hidden_size*2)

        :return: predictions, hidden
            :shape predictions: (batch size, output_size)
            :shape hidden: (batch size, hidden_size)
        """
        input_ = input_.unsqueeze(0)
        embedded = self.embedding(input_)
        assert embedded.shape[1] == input_.shape[1] == last_dec_hidden.shape[0], (embedded.shape,input_.shape,last_dec_hidden.shape)

        # The coefficient for each hidden state of the encoder (batch, 1, src_len)
        alpha = self.attention(encoder_hiddens, last_dec_hidden).unsqueeze(1)
        self.last_alpha = alpha

        encoder_hiddens = encoder_hiddens.permute(1, 0, 2)
        c_i = torch.bmm(alpha, encoder_hiddens).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, c_i), dim=2)


        output, hidden = self.rnn(rnn_input, last_dec_hidden.unsqueeze(0))
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        c_i = c_i.squeeze(0)

        prediction = self.fc_out(torch.cat((output, c_i, embedded), dim = 1))
        assert prediction.shape[0] == last_dec_hidden.shape[0]
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.W_0 = nn.Linear(self.encoder.hidden_size * 2, self.decoder.hidden_size)

    def forward(self, src, trad, teacher_forcing_ratio=0.5):
        """
        :param src: The source sequence
        :shape: (src len, batch size)
        :param trad: The translated sequence
        :type trad: (trg len, batch size)

        :return: [description]
        :rtype: [type]
        """
        batch_size = src.shape[1]

        # trad shape = (batch_size, max target sequence length, 1)
        trad_len = trad.shape[0]
        trad_vocab_size = self.decoder.output_dict_size

        # tensor to store the decoded sentence
        outputs = torch.zeros(trad_len, batch_size, trad_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence
        encoder_hidden = self.encoder(src)

        last_dec_hidden_state = self.W_0(encoder_hidden[-1, :, :])
        assert last_dec_hidden_state.shape == (batch_size, self.decoder.hidden_size), last_dec_hidden_state.shape

        # first input to the decoder is the <sos> tokens
        input_ = trad[0]
        for t in range(1, trad_len):
            # insert input word embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input_, last_dec_hidden_state, encoder_hidden)

            # place predictions in a tensor holding predictions for each word
            outputs[t] = output

            # get the index of the word with the highest predicted probability
            word_id = output.argmax(1)

            # if teacher forcing, use the ground truth trad[t] word as next input
            # if not, use predicted token
            input_ = trad[t] if random.random() < teacher_forcing_ratio else word_id
            last_dec_hidden_state = hidden
        return outputs