# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict
import numpy as np


from .transformer import TransformerEncoder, TransformerDecoder
from .rnn import RnnEncoder
from .util import find_lengths

from .reinforce_wrappers import RnnSenderReinforce, RnnReceiverReinforce 

class Critic(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Critic, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        # clamp, tanh, sigmoid?
        return  torch.clamp(x, 0, 1)

class RnnA2CAgent(nn.Module):
    """
    This wrapper works for both the sender and receiver: it comprises a sender and receiver modules that should be 
    used by appropriately calling `sender_act` and `receiver_actreceiver_act`.

    Largely, this only wraps the Reinforce wrappers, and provides two methods for evaluating actions.

    The sender and receiver could be split, but we keep it as is for now so that we can introduce
    selfplay later on.
    """
    def __init__(self, sender, receiver, vocab_size, embed_dim, hidden_size, max_len, num_layers=1, cell='rnn', force_eos=True, is_sender=True):
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param embed_dim: the size of the embedding used to embed the output symbols
        :param hidden_size: the RNN cell's hidden state size
        :param max_len: maximal length of the output messages
        :param cell: type of the cell used (rnn, gru, lstm)
        :param force_eos: if set to True, each message is extended by an EOS symbol. To ensure that no message goes
        beyond `max_len`, Sender only generates `max_len - 1` symbols from an RNN cell and appends EOS.
        """
        super(RnnA2CAgent, self).__init__()
        self.sender = RnnSenderReinforce(
            sender, vocab_size, embed_dim, hidden_size, max_len, num_layers, cell, force_eos
        )
        self.receiver = RnnReceiverReinforce(
            receiver, vocab_size, embed_dim, hidden_size, cell, num_layers,
        )
        self.critic = Critic(hidden_size, 1)
        self.is_sender = is_sender

        self.reset_parameters()

    def reset_parameters(self):
        if self.sender is not None:
            self.sender.reset_parameters()

    def forward(self, x_or_message, _receiver_input = None, _lengths = None):
        # Warning("Use `send` or `receive`, otherwise `is_sender` is used to determine usage.")
        if self.is_sender:
            return self.sender_act(x_or_message)
        else:
            return self.receiver_act(x_or_message, _receiver_input, _lengths)

    def sender_act(self, x):
        return self.sender(x)

    def receiver_act(self, message, input=None, lengths=None):
        return self.receiver(message, input, lengths)

    def evaluate_sender(self, sender_input):
        # input -> value
        # The sender critic makes it decision: Do I know how to understand this encoded item?
        x = self.sender.agent(sender_input)
        values = self.critic(x.detach())
        return values

    def evaluate_receiver(self, message):
        # message -> value
        # The receiver critic makes it decision: Do I know how to understand this encoded message?
        encoded = self.receiver.encoder(message)
        value = self.critic(encoded.detach())
        return value


class SenderReceiverRnnA2C(nn.Module):
    """
    Implements Sender/Receiver game with training done via AC2. 
    
    
    Both agents are supposed to
    return 3-tuples of (output, log-prob of the output, entropy).
    The game implementation is responsible for handling the end-of-sequence term, so that the optimized loss
    corresponds either to the position of the eos term (assumed to be 0) or the end of sequence.

    Sender and Receiver can be obtained by applying the corresponding wrappers.
    `SenderReceiverRnnReinforce` also applies the mean baseline to the loss function to reduce the variance of the
    gradient estimate.

    >>> sender = nn.Linear(3, 10)
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, embed_dim=5, hidden_size=10, max_len=10, cell='lstm')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, embed_dim=10, hidden_size=5)
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    ...     return F.mse_loss(sender_input, receiver_output, reduction='none').mean(dim=1), {'aux': 5.0}

    >>> game = SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0,
    ...                                   length_cost=1e-2)
    >>> input = torch.zeros((16, 3)).normal_()
    >>> optimized_loss, aux_info = game(input, labels=None)
    >>> sorted(list(aux_info.keys()))  # returns some debug info, such as entropies of the agents, message length etc
    ['aux', 'loss', 'mean_length', 'original_loss', 'receiver_entropy', 'sender_entropy']
    >>> aux_info['aux']
    5.0
    """
    def __init__(self, sender, receiver, sender_entropy_coeff, receiver_entropy_coeff,
                 length_cost=0.0, sender_value_coeff=0.5, receiver_value_coeff=0.5, acktr=False):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        """
        super(SenderReceiverRnnA2C, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.length_cost = length_cost

        self.sender_value_coeff = sender_value_coeff
        self.receiver_value_coeff = receiver_value_coeff

        self.acktr = False

        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)

    def forward(self, sender_input, labels, receiver_input=None):
        message, log_prob_s, entropy_s = self.sender.sender_act(sender_input)
        message_lengths = find_lengths(message)
        receiver_output, log_prob_r, entropy_r = self.receiver.receiver_act(message, receiver_input, message_lengths)

        sender_values = self.sender.evaluate_sender(sender_input)
        receiver_values = self.receiver.evaluate_receiver(message)

        correct = (receiver_output == labels).detach().float()
        sender_advantage = correct - sender_values
        receiver_advantage = correct - receiver_values

        sender_value_loss = F.mse_loss(sender_values, correct) # sender_advantage.pow(2).mean()
        receiver_value_loss = F.mse_loss(receiver_values, correct) # receiver_advantage.pow(2).mean()
        weighted_value_loss = (
            sender_value_loss * self.sender_value_coeff + receiver_value_loss * self.receiver_value_coeff
        )

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after.
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest.
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()
        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        # off by default (length_cost = 0.)
        length_loss = message_lengths.float() * self.length_cost
        policy_length_loss = (
            (length_loss.float() - self.mean_baseline['length']) * effective_log_prob_s
        ).mean()

        policy_loss = (
            # sender policy loss.
            -(sender_advantage.detach() * effective_log_prob_s).mean()
            # receiver policy loss.
            + -(receiver_advantage.detach() * log_prob_r).mean()
        )

        optimized_loss = policy_loss + policy_length_loss + weighted_value_loss - weighted_entropy

        rest = {'acc': (receiver_output == labels).detach().float().mean()}
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_value_loss'] = sender_value_loss.mean().item()
        rest['receiver_value_loss'] = receiver_value_loss.mean().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = policy_loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()
    
        return optimized_loss, rest
