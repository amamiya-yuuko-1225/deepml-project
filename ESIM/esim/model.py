"""
Definition of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn

from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from .utils import get_mask, replace_masked
import torch.nn.functional as F


class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths)\
            .to(self.device)

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)

        attended_premises, attended_hypotheses =\
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)

        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0


# ---------------- ABCNN implementation (add to model.py) ----------------

class ABCNN(nn.Module):
    """
    Attention-Based Convolutional Neural Network for sentence pair modeling.
    Supports ABCNN-1, ABCNN-2, and ABCNN-3 (both 1 + 2).
    Reference: Yin et al., "ABCNN: Attention-Based Convolutional Neural Network
    for Modeling Sentence Pairs" (2015).
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_filters: int = 300,
                 kernel_size: int = 3,
                 abcnn_type: str = "ABCNN3",   # "ABCNN1" | "ABCNN2" | "ABCNN3"
                 embeddings: torch.Tensor = None,
                 padding_idx: int = 0,
                 dropout: float = 0.5,
                 num_classes: int = 3,
                 device: str = "cpu"):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel_size for same-length padding."
        assert abcnn_type in {"ABCNN1", "ABCNN2", "ABCNN3"}

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.abcnn_type = abcnn_type
        self.dropout = dropout
        self.num_classes = num_classes
        self.device = device

        # Word embedding layer
        self._word_embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=padding_idx,
            _weight=embeddings
        )

        # In ABCNN-1 we augment input with an attention feature map (extra channel).
        in_channels = 2 if self.abcnn_type in {"ABCNN1", "ABCNN3"} else 1

        # 2D convolution over (seq_len x embedding_dim). We collapse embed dim by kernel height.
        # Input shape we use: (B, C, L, E); Conv2d kernel is (k, E), padding (k//2, 0) -> preserves L.
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_filters,
            kernel_size=(self.kernel_size, self.embedding_dim),
            padding=(self.kernel_size // 2, 0),
            bias=True
        )

        # Projection after concatenating pooled representations of premise & hypothesis.
        # We combine mean and max pooling (2*F) and optionally attention-pooled vectors (+2*F if ABCNN-2 / ABCNN-3).
        base_feat = 2 * self.num_filters  # mean + max for each side, then concat => 4F total at the end
        att_feat = 2 * self.num_filters if self.abcnn_type in {"ABCNN2", "ABCNN3"} else 0
        total_feat = 2 * (self.num_filters + self.num_filters)  # (prem_mean,max) + (hyp_mean,max) = 4F
        if att_feat > 0:
            total_feat += 2 * self.num_filters  # add attention pooled prem + hyp => +2F

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(total_feat, self.num_filters),
            nn.Tanh(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.num_filters, self.num_classes)
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def cosine_sim_3d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
        """
        Compute pairwise cosine similarity matrix between two sequences.
        x: (B, Lx, D), y: (B, Ly, D)
        returns A: (B, Lx, Ly)
        """
        x_norm = x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
        y_norm = y / (y.norm(p=2, dim=-1, keepdim=True) + eps)
        return torch.bmm(x_norm, y_norm.transpose(1, 2))  # (B, Lx, Ly)

    def build_abcnn1_inputs(self, x_embed, y_embed, x_mask, y_mask):
        """
        ABCNN-1: build attention feature maps for each side and stack as channels.
        x_embed, y_embed: (B, L, E)
        returns x_in, y_in shaped as (B, C=2, L, E)
        """
        A = self.cosine_sim_3d(x_embed, y_embed)  # (B, Lx, Ly)
        # Mask padded positions in A
        if x_mask is not None and y_mask is not None:
            A = A * (x_mask.unsqueeze(2) * y_mask.unsqueeze(1)).float()

        # Attention features: soft alignment vectors
        # For each token in x, attention-weighted sum over y, and vice versa.
        alpha_x = F.softmax(A, dim=2)    # sum over y positions
        alpha_y = F.softmax(A.transpose(1, 2), dim=2)  # sum over x positions

        attn_x = torch.bmm(alpha_x, y_embed)  # (B, Lx, E)
        attn_y = torch.bmm(alpha_y, x_embed)  # (B, Ly, E)

        # Stack channels: original embedding + attention map -> (B, 2, L, E)
        x_in = torch.stack([x_embed, attn_x], dim=1)
        y_in = torch.stack([y_embed, attn_y], dim=1)
        return x_in, y_in

    def conv_encode(self, x_in):
        """
        Apply convolution + ReLU to an input with shape (B, C, L, E).
        Returns feature map with shape (B, F, L), collapsing embedding dimension.
        """
        # Conv2d with kernel (k, E) collapses last dimension to 1
        z = self.conv(x_in)                # (B, F, L, 1)
        z = F.relu(z).squeeze(-1)          # (B, F, L)
        return z

    def abcnn2_attention_pool(self, fx: torch.Tensor, fy: torch.Tensor,
                              x_mask: torch.Tensor, y_mask: torch.Tensor):
        """
        ABCNN-2: attention weights computed from conv feature maps, used to pool.
        fx, fy: (B, F, L)
        Masks: (B, L) bool tensors (True for valid tokens).
        Returns attention-pooled vectors: (B, F) for each side.
        """
        # Transpose to (B, L, F) for cosine
        x_t = fx.transpose(1, 2)  # (B, Lx, F)
        y_t = fy.transpose(1, 2)  # (B, Ly, F)

        A = self.cosine_sim_3d(x_t, y_t)   # (B, Lx, Ly)
        if x_mask is not None and y_mask is not None:
            A = A * (x_mask.unsqueeze(2) * y_mask.unsqueeze(1)).float()

        # Row-wise / col-wise attention weights
        ax = F.softmax(A, dim=2)           # (B, Lx, Ly)
        ay = F.softmax(A.transpose(1, 2), dim=2)  # (B, Ly, Lx)

        # Weighted sums: (B, Lx, F) and (B, Ly, F)
        x_ctx = torch.bmm(ax, y_t)         # (B, Lx, F)
        y_ctx = torch.bmm(ay, x_t)         # (B, Ly, F)

        # Aggregate along time with masks -> weighted mean (simple)
        def masked_mean(t, mask):
            # t: (B, L, F), mask: (B, L) bool
            if mask is None:
                return t.mean(dim=1)
            m = mask.float().unsqueeze(-1)     # (B, L, 1)
            s = (t * m).sum(dim=1)             # (B, F)
            z = m.sum(dim=1).clamp(min=1e-6)   # (B, 1)
            return s / z

        x_att_pooled = masked_mean(x_ctx, x_mask)  # (B, F)
        y_att_pooled = masked_mean(y_ctx, y_mask)  # (B, F)
        return x_att_pooled, y_att_pooled

    @staticmethod
    def masked_mean_max(feat: torch.Tensor, mask: torch.Tensor):
        """
        Mean and max pooling along time with masks.
        feat: (B, F, L), mask: (B, L) bool
        returns (mean: B,F), (max: B,F)
        """
        if mask is None:
            mean_vec = feat.mean(dim=-1)
            max_vec, _ = feat.max(dim=-1)
            return mean_vec, max_vec

        # mean
        m = mask.float().unsqueeze(1)  # (B,1,L)
        sum_vec = (feat * m).sum(dim=-1)  # (B,F)
        len_vec = m.sum(dim=-1).clamp(min=1e-6)  # (B,1)
        mean_vec = sum_vec / len_vec

        # max (replace masked with large negative)
        masked_feat = replace_masked(feat.transpose(1, 2), mask, -1e7).transpose(1, 2)
        max_vec, _ = masked_feat.max(dim=-1)  # (B,F)
        return mean_vec, max_vec

    def forward(self,
                premises: torch.Tensor,
                premises_lengths: torch.Tensor,
                hypotheses: torch.Tensor,
                hypotheses_lengths: torch.Tensor):
        """
        Inputs:
            premises, hypotheses: (B, L) LongTensor of token ids
            premises_lengths, hypotheses_lengths: (B,) lengths
        Returns:
            logits: (B, num_classes)
            probabilities: (B, num_classes)
        """
        # Build masks (True = valid token)
        prem_mask = get_mask(premises, premises_lengths).to(self.device)  # (B,L)
        hyp_mask  = get_mask(hypotheses, hypotheses_lengths).to(self.device)

        # Embedding lookup -> (B, L, E)
        prem_emb = self._word_embedding(premises)
        hyp_emb  = self._word_embedding(hypotheses)

        if self.dropout:
            prem_emb = F.dropout(prem_emb, p=self.dropout, training=self.training)
            hyp_emb  = F.dropout(hyp_emb, p=self.dropout, training=self.training)

        # Prepare inputs for convolution
        if self.abcnn_type in {"ABCNN1", "ABCNN3"}:
            # ABCNN-1: add attention feature map as extra channel
            x_in, y_in = self.build_abcnn1_inputs(prem_emb, hyp_emb, prem_mask, hyp_mask)
        else:
            # Shape as (B, C=1, L, E)
            x_in = prem_emb.unsqueeze(1)
            y_in = hyp_emb.unsqueeze(1)

        # Convolutional encoding -> (B, F, L)
        fx = self.conv_encode(x_in)
        fy = self.conv_encode(y_in)

        if self.dropout:
            fx = F.dropout(fx, p=self.dropout, training=self.training)
            fy = F.dropout(fy, p=self.dropout, training=self.training)

        # Base mean/max pooling
        prem_mean, prem_max = self.masked_mean_max(fx, prem_mask)  # (B,F) each
        hyp_mean,  hyp_max  = self.masked_mean_max(fy, hyp_mask)

        reps = [prem_mean, prem_max, hyp_mean, hyp_max]

        # ABCNN-2: attention-based pooling on conv features
        if self.abcnn_type in {"ABCNN2", "ABCNN3"}:
            prem_att, hyp_att = self.abcnn2_attention_pool(fx, fy, prem_mask, hyp_mask)
            reps.extend([prem_att, hyp_att])

        # Final representation and classification
        v = torch.cat(reps, dim=-1)  # (B, D)
        logits = self.classifier(v)
        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities
# ---------------- end ABCNN ----------------
