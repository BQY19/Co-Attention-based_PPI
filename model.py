import torch
import torch.nn as nn
from math import sqrt


def get_attn_pad_mask(mask1, mask2):
    len1 = mask1.size(1)
    len2 = mask2.size(1)
    mask2 = mask2.unsqueeze(-1).repeat(1, 1, len1)
    mask1 = mask1.unsqueeze(1).repeat(1, len2, 1)
    pad_attn_mask = mask1 * mask2
    return pad_attn_mask.data.eq(0)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dp = nn.Dropout(dropout)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn1 = self.sm(scores)
        # attn2 = self.sm(scores.transpose(-1, -2))
        context = torch.matmul(self.dp(attn1), V)
        return context


class ScaledDotProductAttention_2(nn.Module):
    def __init__(self, d_k, dropout):
        super(ScaledDotProductAttention_2, self).__init__()
        self.d_k = d_k
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.sm = nn.Softmax(dim=-1)
        self.q = nn.Linear(766, 766, bias=False)
        self.k = nn.Linear(766, 766, bias=False)

    def forward(self, Q, K, attn_mask):  # pro2,pro1
        scores = torch.matmul(Q, K.transpose(-1, -2)) / sqrt(self.d_k)
        scores.masked_fill_(attn_mask.transpose(2, 3), -1e9)
        attn1 = self.sm(scores)
        attn2 = self.sm(scores.transpose(-1, -2))  # [batch_size, head, len_k, len_q]
        # attn2=self.sm((torch.matmul(K, Q.transpose(-1, -2)) / sqrt(self.d_k)).masked_fill_(attn_mask, -1e9))
        # Q=self.q(Q.transpose(-1, -2)).transpose(-1, -2)
        # K=self.k(K.transpose(-1, -2)).transpose(-1, -2)
        context1 = torch.matmul(self.dp1(attn1), K)  # [batch_size, len_q, d_v]
        context2 = torch.matmul(self.dp2(attn2), Q)  # [batch_size, len_k, d_v]
        return context1, context2, attn1, attn2


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.fc1 = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.fc2 = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.nm1 = nn.LayerNorm(d_model)
        self.nm2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dp1 = nn.Dropout(p=dropout)
        self.dp2 = nn.Dropout(p=dropout)
        self.sdpa = ScaledDotProductAttention_2(d_k, dropout)

    def forward(self, pro1, pro2, mask1_2):
        batch_size = pro1.size(0)  # batch_size

        pro1 = self.W_Q(pro1).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # Q: [batch_size, n_heads, max_len, d_k]
        pro2 = self.W_K(pro2).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # K: [batch_size, n_heads, max_len, d_k]

        pro1, pro2, coattn1, coattn2 = self.sdpa(pro1, pro2, mask1_2)
        pro1 = pro1.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        pro2 = pro2.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        pro1 = self.fc1(pro1)  # [batch_size, max_len, d_model]
        pro2 = self.fc1(pro2)


        return self.dp1(self.nm1(pro1)), self.dp2(self.nm2(pro2))


class MultiHeadAttention_(nn.Module):  #
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        super(MultiHeadAttention_, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_Q_ = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K_ = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.fc1 = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.fc2 = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.nm1 = nn.LayerNorm(d_model)
        self.nm2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_k = d_k  # d_model//nheads?
        self.d_v = d_v
        self.dp1 = nn.Dropout(p=dropout)
        self.dp2 = nn.Dropout(p=dropout)
        self.sdpa1 = ScaledDotProductAttention(d_k, dropout)
        self.sdpa2 = ScaledDotProductAttention(d_k, dropout)

    def forward(self, pro1i, pro2i, mask1_2):
        batch_size = pro1i.size(0)  # batch_size

        pro1 = self.W_Q(pro1i).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                      2)  # Q: [batch_size, n_heads, max_len, d_k]
        pro2 = self.W_K(pro2i).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                      2)  # K: [batch_size, n_heads, max_len, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)                   # attn_mask : [batch_size, n_heads, max_len, max_len]

        pro1_ = self.W_Q_(pro1i).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                        2)  # Q: [batch_size, n_heads, max_len, d_k]
        pro2_ = self.W_K_(pro2i).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                        2)  # K: [batch_size, n_heads, max_len, d_k]

        pro2 = self.sdpa1(pro2, pro1, pro1, mask1_2)  
        pro1 = self.sdpa2(pro1_, pro2_, pro2_, mask1_2.transpose(2, 3))
        pro1 = pro1.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        pro2 = pro2.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        pro1 = self.fc1(pro1)  # [batch_size, max_len, d_model]
        pro2 = self.fc2(pro2)

        return self.dp1(self.nm1(pro1)), self.dp2(self.nm2(pro2))


class HieDPA(nn.Module):
    def __init__(self, d_model, d_k, dropout):
        super(HieDPA, self).__init__()
        self.v = nn.Linear(d_model, d_k, bias=False)
        self.b = nn.Linear(d_model, d_model, bias=False)  ##bi→bias?
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.a_wv = nn.Parameter(torch.ones([1, d_k]), requires_grad=True)
        self.a_wq = nn.Parameter(torch.ones([1, d_k]), requires_grad=True)
        self.dp1 = nn.Dropout(dropout)
        self.bm1 = nn.BatchNorm1d(d_model)
        self.dp2 = nn.Dropout(dropout)
        self.bm2 = nn.BatchNorm1d(d_model)  # bm1,bm2?  d_model=dim
        self.sm = nn.Softmax(dim=-1)
        self.th = nn.Tanh()
        self.d_model = d_model

    def forward(self, Q, V, attn_mask):
        Wq = self.q(Q).transpose(1, 2);
        Wv = self.v(V).transpose(1, 2)  # Q: [batch_size, d_k, max_len1]   K: [batch_size, d_k, max_len2]
        bk = self.b(V).transpose(1, 2)
        scores = self.th(torch.matmul(Q, bk))  # [B, len1, len2]
        scores = scores.masked_fill(attn_mask.transpose(1, 2), 0)
        Hv = self.th(Wv + torch.matmul(Wq, scores))  # [B, k, len2]
        Hq = self.th(Wq + torch.matmul(Wv, scores.transpose(-1, -2)))  # [B, k, len1]

        # ppi matrix
        # scores_ = torch.matmul(Hq, Hv.transpose(1,2))  / sqrt(self.d_k)
        # scores_.masked_fill_(attn_mask.transpose(1, 2), -1e9)
        # attn = self.sm(scores_)

        av = self.sm(torch.matmul(self.a_wv, Hv)).squeeze()  # [B, len2]
        aq = self.sm(torch.matmul(self.a_wq, Hq)).squeeze()  # [B, len1]

        V1 = torch.sum(av.unsqueeze(-1).repeat(1, 1, self.d_model) * V, 1)
        Q1 = torch.sum(aq.unsqueeze(-1).repeat(1, 1, self.d_model) * Q, 1)  # [B, D]

        return self.dp1(self.bm1(Q1)), self.dp2(self.bm2(V1)), aq, av


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=True))
        # self.nm = nn.LayerNorm(d_model)
        self.dp = nn.Dropout(dropout)

    def forward(self, inputs):
        output = self.dp(self.fc(inputs))
        return output


class RCNN(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, c_layers, dropout):
        super(RCNN, self).__init__()
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, padding=1)
        self.rnn = nn.LSTM(128, 64, batch_first=True)

    def forward(self, pro1, pro2):
        pro1_conv = self.conv1d(pro1.transpose(1, 2)).transpose(1, 2)
        pro2_conv = self.conv1d(pro2.transpose(1, 2)).transpose(1, 2)
        pro1_relu = self.relu(pro1_conv)
        pro2_relu = self.relu(pro2_conv)
        pro1_rnn, _ = self.rnn(pro1_relu)
        pro2_rnn, _ = self.rnn(pro2_relu)
        pro1_output = pro1_rnn
        pro2_output = pro2_rnn

        return pro1_output, pro2_output

class CoAttnLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads, dropout):
        super(CoAttnLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.pos_ffn1 = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.pos_ffn2 = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.nm1 = nn.LayerNorm(d_model)
        self.nm2 = nn.LayerNorm(d_model)
        self.head = n_heads

    def forward(self, pro1, pro2, mask1_2):  # Res?
        res1, res2 = pro1, pro2
        pro1, pro2 = self.attn(pro1, pro2, mask1_2)
        pro1 = self.pos_ffn1(pro1)
        pro2 = self.pos_ffn2(pro2)
        return self.nm1(res1 + pro1), self.nm2(res2 + pro2)


class CoAttn(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, c_layers, n_heads, dropout):
        super(CoAttn, self).__init__()
        self.layers = nn.ModuleList([CoAttnLayer(d_model, d_k, d_v, d_ff, n_heads, dropout) for _ in range(c_layers)])
        self.head = n_heads

    def forward(self, pro1, pro2, mask1_2):
        mask1_2 = mask1_2.unsqueeze(1).repeat(1, self.head, 1, 1)
        for i, layer in enumerate(self.layers):
            pro1, pro2 = layer(pro1, pro2, mask1_2)
        return pro1, pro2


class PPI_site(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, c_layers, n_heads, dropout):
        super(PPI_site, self).__init__()
        self.l1 = nn.Linear(1024, d_model)
        self.l2 = nn.Linear(1024, d_model)
        self.coattn = CoAttn(d_model, d_k, d_v, d_ff, c_layers, n_heads, dropout)
        self.hieattn = HieDPA(d_model, d_k, dropout)
        self.rcnn = RCNN(d_model, d_k, d_v, d_ff, c_layers, dropout)

        self.ppi = nn.Sequential(
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, pro1, pro2, mask1, mask2):
        mask1_2 = get_attn_pad_mask(mask1, mask2)
        pro1 = self.l1(pro1)
        pro2 = self.l2(pro2)
        # pro1, pro2 = self.coattn(pro1, pro2, mask1_2)
        pro1, pro2 = self.coattn(pro1, pro2, mask1_2)
        # pro1,pro2=self.rcnn(pro1,pro2)
        concat = torch.cat([pro1, pro2], 1)
        logits = self.ppi(concat)
        return logits, pro1, pro2


class PPI(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, c_layers, n_heads, dropout):
        super(PPI, self).__init__()
        self.l1 = nn.Linear(1024, d_model)
        self.l2 = nn.Linear(1024, d_model)
        self.coattn = CoAttn(d_model, d_k, d_v, d_ff, c_layers, n_heads, dropout)
        self.hieattn = HieDPA(d_model, d_k, dropout)
        self.rcnn = RCNN(d_model, d_k, d_v, d_ff, c_layers, dropout)
        self.ppi = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, pro1, pro2, mask1, mask2):
        mask1_2 = get_attn_pad_mask(mask1, mask2)
        pro1 = self.l1(pro1)
        pro2 = self.l2(pro2)
        pro1, pro2 = self.coattn(pro1, pro2, mask1_2)
        pro1, pro2, pro1_attn, pro2_attn = self.hieattn(pro1, pro2, mask1_2)
        concat = torch.cat([pro1, pro2], -1)
        logits = self.ppi(concat)
        return logits, pro1, pro2