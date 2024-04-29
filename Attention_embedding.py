# 导入需要的库
import time
import torch
import biom
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, mean_absolute_error, r2_score
from sklearn import metrics
import gc
import os
import matplotlib.pyplot as plt
import yaml
import wandb


# GPU选择
def gpu(i=0):
    """Defined in :numref:`sec_use_gpu`"""
    return torch.device(f'cuda:{i}')


def num_gpus():
    """Defined in :numref:`sec_use_gpu`"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    return torch.cuda.device_count()


def try_all_gpus(numb):
    """Return all available GPUs, or [cpu(),] if no GPU exists.
    Defined in :numref:`sec_use_gpu`"""
    return [gpu(i) for i in [numb]]


# 导入数据
def read_imdb(biom_table, metadata, group):
    """读取OTU table和样本标签"""
    labels = []
    table = biom.load_table(biom_table)
    fid = table.ids(axis="observation")

    mapping_file = pd.read_csv(metadata,
                            sep="\t",
                            index_col=0,
                            low_memory=False)
    labels = list(mapping_file.loc[table.ids(axis='sample')][group].values)
    table = table.rankdata(axis='sample', inplace=False)
    table = table.matrix_data.multiply(1 / table.max(axis='sample'))
    # table = table.norm(axis='sample', inplace=False).matrix_data
    table = table.toarray().T
    return table, fid, labels


class Fid:
    """Vocabulary for text."""

    def __init__(self, tokens, reserved_tokens=[]):
        """Defined in :numref:`sec_text-sequence`"""
        # The list of tokens
        self.idx_to_token = list(
            sorted(
                set(reserved_tokens + ['<unk>'] + [
                    token
                    for token in tokens
                ])))
        # cls向量放置在第0号索引位置
        self.idx_to_token = ['cls'] + self.idx_to_token
        self.token_to_idx = {
            token: idx
            for idx, token in enumerate(self.idx_to_token)
        }

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']


class load_data_imdb(Dataset):
    """

    Returns
    -------
    _type_
        _description_ 
    """

    def __init__(self, biom_table, metadata, group, num_steps=500):

        otu, fid, labels = read_imdb(biom_table, metadata, group)
        self.fid = Fid(fid, reserved_tokens=['<pad>'])
        self.features, self.abundance, self.mask = self.truncate_pad(otu, fid, num_steps)
        self.labels = torch.tensor(labels)
        self.otu = otu

    def truncate_pad(self, otu, fid, num_steps):
        features = np.zeros((otu.shape[0], num_steps), dtype=int)
        cls_column = np.zeros((features.shape[0], 1)).astype(int)
        abundance = np.zeros((otu.shape[0], num_steps))
        cls_abundance = np.ones((features.shape[0], 1)).astype(int)
        for i in range(0, otu.shape[0]):
            nonzero_count = np.count_nonzero(otu[i,])
            if nonzero_count >= num_steps:
                sorted_indices = np.argsort(otu[i,])[::-1]
                features[i, ] = np.array([self.fid[line]
                                          for line in
                                          fid[sorted_indices[:num_steps]]])
                abundance[i, ] = otu[i, sorted_indices[:num_steps]]
            else:
                indices = np.nonzero(otu[i,])
                features[i, :nonzero_count] = np.array([self.fid[line]
                                                        for line in
                                                        fid[indices]])
                features[i, nonzero_count:] = self.fid['<pad>']
                abundance[i, :nonzero_count] = otu[i, indices]
        features = np.concatenate((cls_column, features), axis=1)
        abundance = np.concatenate((cls_abundance, abundance), axis=1)

        input_mask = torch.ones(features.shape[0], 
                                features.shape[1], 
                                dtype=torch.long)
        input_mask[features == self.fid['<pad>']] = 0 # mask padding tokens
        input_mask[features == 0] = 0 # mask cls tokens
        return torch.tensor(features), torch.tensor(abundance), input_mask

    def __call__(self):
        return self.fid

    def __getitem__(self, index):
        return self.features[index, ], self.abundance[index, ], self.labels[index], self.mask[index]

    def __len__(self):
        """ Returns the number of sample. """
        return self.otu.shape[0]


class TokenEmbedding:
    """Token Embedding."""

    def __init__(self, data_dir):
        """Defined in :numref:`sec_synonyms`"""
        self.idx_to_token, self.idx_to_vec = self._load_embedding(data_dir)
        self.unknown_idx = 0
        self.token_to_idx = {
            token: idx
            for idx, token in enumerate(self.idx_to_token)
        }

    def _load_embedding(self, data_dir):
        # idx_to_token : OTU的名称
        # idx_to_vec : OTU的名称对应的
        idx_to_token, idx_to_vec = ['<unk>'], []
        with open(data_dir, 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [
            self.token_to_idx.get(token, self.unknown_idx) for token in tokens
        ]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        # 找到0元素的位置
        zero_index = [i for i, x in enumerate(indices) if x == 0]
        # 对vecs进行操作，将0元素位置的向量位置替换成随机向量数值在（-0.1，0.1）
        vector = torch.empty(len(zero_index), vecs.shape[1])
        vecs[zero_index, ] = nn.init.xavier_uniform_(vector)
        
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


# 搭建模型
class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()

        self.d_k = d_k

    def forward(self, q, k, v, attn_mask):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k), |v| : (batch_size, n_heads, v_len, d_v)

        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))

        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)

        attn_score.masked_fill_(attn_mask, -1e9)

        # |attn_score| : (batch_size, n_heads, q_len, k_len)

        attn_weights = nn.Softmax(dim=-1)(attn_score)

        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        output = torch.matmul(attn_weights, v)

        # |output| : (batch_size, n_heads, q_len, d_v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads

        self.d_k = self.d_v = d_model // n_heads

        self.WQ = nn.Linear(d_model, d_model)

        self.WK = nn.Linear(d_model, d_model)

        self.WV = nn.Linear(d_model, d_model)

        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)

        self.linear = nn.Linear(n_heads * self.d_v, d_model)

    def forward(self, Q, K, V, attn_mask):
        # |Q| : (batch_size, q_len, d_model), |K| : (batch_size, k_len, d_model), |V| : (batch_size, v_len, d_model)

        # |attn_mask| : (batch_size, seq_len(=q_len), seq_len(=k_len))

        batch_size = Q.size(0)

        q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads,
                                  self.d_k).transpose(1, 2)

        k_heads = self.WK(K).view(batch_size, -1, self.n_heads,
                                  self.d_k).transpose(1, 2)

        v_heads = self.WV(V).view(batch_size, -1, self.n_heads,
                                  self.d_v).transpose(1, 2)

        # |q_heads| : (batch_size, n_heads, q_len, d_k), |k_heads| : (batch_size, n_heads, k_len, d_k), |v_heads| : (batch_size, n_heads, v_len, d_v)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))

        attn, attn_weights = self.scaled_dot_product_attn(
            q_heads, k_heads, v_heads, attn_mask)

        # |attn| : (batch_size, n_heads, q_len, d_v)

        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1,
                                                      self.n_heads * self.d_v)

        # |attn| : (batch_size, q_len, n_heads * d_v)

        output = self.linear(attn)

        # |output| : (batch_size, q_len, d_model)

        return output, attn_weights


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)

        self.dropout1 = nn.Dropout(p_drop)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout2 = nn.Dropout(p_drop)

        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, attn_mask, weight):
        # |inputs| : (batch_size, seq_len, d_model)

        # |attn_mask| : (batch_size, seq_len, seq_len)
        # inputs_1 = inputs.permute(2, 0, 1) * weight
        # inputs_1 = inputs_1.permute(1, 2, 0)
        attn_outputs, attn_weights = self.mha(inputs, inputs, inputs,
                                              attn_mask)

        attn_outputs = self.dropout1(attn_outputs)

        attn_outputs = self.layernorm1(inputs + attn_outputs)

        # |attn_outputs| : (batch_size, seq_len(=q_len), d_model)

        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        return attn_outputs, attn_weights


class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers.
    Args:
        vocab_size (int)    : vocabulary size (vocabulary: collection mapping token to numerical identifiers)
        seq_len    (int)    : input sequence length
        d_model    (int)    : number of expected features in the input
        n_layers   (int)    : number of sub-encoder-layers in the encoder
        n_heads    (int)    : number of heads in the multiheadattention models
        p_drop     (float)  : dropout value
        d_ff       (int)    : dimension of the feedforward network model
        pad_id     (int)    : pad token id
    Examples:
    >>> encoder = TransformerEncoder(vocab_size=1000, seq_len=512)
    >>> inp = torch.arange(512).repeat(2, 机器学习)
    >>> encoder(inp)
    """

    def __init__(self,
                 otu_size,
                 seq_len,
                 d_model=128,
                 n_layers=6,
                 n_heads=8,
                 p_drop=0.1,
                 d_ff=128,
                 pad_id=0):

        super(TransformerEncoder, self).__init__()
        # layers
        self.embedding = nn.Embedding(otu_size, d_model, padding_idx=pad_id)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, p_drop, d_ff)
            for _ in range(n_layers)
        ])
        # layers to classify
        # self.linear1 = nn.Linear(seq_len-1, d_model)
        # self.linear2 = nn.Linear(d_model, 2 * d_model)
        self.linear3 = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.pad_id = pad_id
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x, weight, mask, classification=True):

        inputs = self.embedding(x)
        inputs = inputs.permute(2, 0, 1) * weight
        # |inputs| : (batch_size, seq_len, d_model)
        inputs = inputs.permute(1, 2, 0)

        attention_weights = []
        attn_pad_mask = self.get_attention_padding_mask(
            x, x, self.pad_id)

        for layer in self.layers:
            inputs, attn_weights = layer(inputs.float(), attn_pad_mask, weight)
            # |outputs| : (batch_size, seq_len, d_model)
            # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)

            attention_weights.append(attn_weights)
        # |outputs| : (batch_size * seq_len * d_model)
        # outputs, _ = torch.max(inputs[:, 1:, :].float(), dim=1)
        # outputs = self.dropout(outputs)
        # |outputs| : (batch_size, d_model)
        outputs = self.dropout(inputs[:, 0, :]) # cls
        # abundance = self.linear1(weight.float())
        # outputs = self.linear1(outputs)
        # embedding_sum  = inputs * mask.unsqueeze(-1)
        # embedding = embedding_sum.sum(
        #     dim=1, keepdim=False) / mask.sum(1).unsqueeze(-1)
        # outputs = self.dropout(embedding)
        outputs = self.relu(outputs)
        outputs = self.linear3(outputs)
        if classification is False:
            outputs = self.relu(outputs)
        # |outputs| : (batch_size, 2)
        return outputs, attention_weights

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)
        return attn_pad_mask


def evaluate_auc_gpu(Y, prob, device=None):
    """_summary_

    Parameters
    ----------
    net : _type_
        _description_
    data_iter : _type_
        _description_
    device : _type_, optional
        _description_, by default None
    """
    Y = Y.numpy().astype('int')
    prob = prob.cpu().detach().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(Y, prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision, recall, threshold = metrics.precision_recall_curve(Y,
                                                                  prob,
                                                                  pos_label=1)
    numerator = 2 * recall * precision
    denom = recall + precision

    f1_scores = np.divide(numerator,
                          denom,
                          out=np.zeros_like(denom),
                          where=(denom != 0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = threshold[np.argmax(f1_scores)]
    # max_f1_thresh = 0.5
    aupr = metrics.auc(recall, precision)
    y_hat = np.array([1 if i > max_f1_thresh else 0 for i in prob])
    cm = confusion_matrix(Y, y_hat)
    f1_scores = f1_score(Y, y_hat, average='macro')
    mcc = matthews_corrcoef(Y, y_hat)

    return auc, max_f1_thresh, aupr, cm, f1_scores, mcc 


def evaluate_r2_gpu(Y, pred, device=None):
    Y = Y.numpy()
    pred = pred.cpu().detach().numpy()
    mae = mean_absolute_error(Y, pred)
    r2 = r2_score(Y, pred)

    return mae, r2


class Timer:
    """Record multiple running times."""

    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def train_batch_ch13(net, X, y, abundance, mask, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13).
    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0], dtype=torch.int64)
    abundance = abundance.to(devices[0])
    mask = mask.to(devices[0])
    net.train()
    trainer.zero_grad()
    if type(loss) is nn.MSELoss:
        pred, _ = net(X, abundance, mask, classification=False)
        pred = torch.squeeze(pred, dim=1)
    else:
        pred, _ = net(X, abundance, mask)
        pred = torch.squeeze(pred, dim=1)
        # l = loss(pred, y, devices)
        pred = torch.sigmoid(pred)
    l = loss(pred.float(), y.float())
    l.backward()
    trainer.step()
    train_loss_sum = l.sum()
    return train_loss_sum, pred


def train_log(epoch, **kwargs):
    wandb.log(kwargs)


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator:
    """For plotting data in animation."""

    def __init__(self,
                 xlabel=None,
                 ylabel=None,
                 legend=None,
                 xlim=None,
                 ylim=None,
                 xscale='linear',
                 yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'),
                 nrows=1,
                 ncols=1,
                 figsize=(5, 5)):
        """Defined in :numref:`sec_utils`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        # use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim,
                                            ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y, plotfile):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        self.fig.savefig(plotfile)
        
        
def train_model(net, train_iter, test_iter, loss,trainer, scheduler, num_epochs, devices,
                embedding_birnn, plotfile_loss, plotfile_auc):
    """Train a model with multiple GPUs.
    Defined in :numref:`sec_image_augmentation`"""
    animator_1 = Animator(xlabel='epoch', xlim=[1, num_epochs],legend=['train loss', 'test loss'])
    # net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    net = net.to(devices[0])
    test_list = []
    train_list = []
    test_loss_list = []
    fid_embedding = net.embedding.weight.data[1:].clone()
    true_age = []
    pred_age = []
    if type(loss) is nn.MSELoss:
        plotfile_mae = plotfile_auc
        animator_2 = Animator(xlabel='epoch', xlim=[1, num_epochs],legend=['train mae', 'test mae'])
        for epoch in range(num_epochs):
            train_loss = 0
            for i, (features, abundance, labels, mask) in enumerate(train_iter):
                l, pred = train_batch_ch13(
                    net, features, labels, abundance, mask, loss, trainer, devices)
                net.embedding.weight.data[1:] = fid_embedding
                train_loss += l
                if i == 0:
                    prob1 = pred
                    Label = labels
                else:
                    prob1 = torch.cat((prob1, pred), dim=0)
                    Label = torch.cat((Label, labels), dim=0)
            # scheduler.step()
            train_loss /= (i + 1)
            animator_1.add(epoch + 1, (train_loss.cpu().detach().numpy(), None), plotfile_loss)
            train_mae, train_r2= evaluate_r2_gpu(
                Label, prob1)
            animator_2.add(epoch + 1, (train_mae, None), plotfile_mae)
            true_age_test=[]
            pred_age_test=[]
            with torch.no_grad():
                net.eval()
                test_loss = 0
                for i, (features, abundance, labels, mask) in enumerate(test_iter):
                    X = features.to(devices[0])
                    abundance = abundance.to(devices[0])
                    mask = mask.to(devices[0])
                    pred, _ = net(X, abundance, mask, classification=False)
                    pred = torch.squeeze(pred, dim=1)
                    y = labels.to(devices[0], dtype=torch.int64)
                    true_age_test.append(labels.detach().cpu().numpy())
                    pred_age_test.append(pred.detach().cpu().numpy())
                    l = loss(pred.float(), y.float())
                    test_loss += l
                    if i == 0:
                        prob2 = pred
                        Label = labels
                    else:
                        prob2 = torch.cat((prob2, pred), dim=0)
                        Label = torch.cat((Label, labels), dim=0)
                true_age_test=np.concatenate(true_age_test)
                pred_age_test=np.concatenate(pred_age_test)
                true_age.append(true_age_test)
                pred_age.append(pred_age_test)
                test_loss /= (i + 1)
            animator_1.add(epoch + 1, (None, test_loss.cpu().detach().numpy()), plotfile_loss)
            test_mae, test_r2 = evaluate_r2_gpu(Label, prob2)
            animator_2.add(epoch + 1, (None, test_mae), plotfile_mae)
            train_list.append(train_mae)
            test_list.append(test_mae)
            test_loss_list.append(test_loss)
            # f1_list.append()
            # 导出最优模型参数
            if len(test_loss_list) == 1:
                train_r2_1 = train_r2
                test_r2_1 = test_r2
                train_loss_1 = train_loss
                test_loss_1 = test_loss
                train_mae_1 = train_mae
                test_mae_1 = test_mae
                pro_ = prob2.cpu().detach().numpy()
                torch.save(net.state_dict(), embedding_birnn)
            else:
                if test_list[-1] < test_mae_1:
                    epoch1 = epoch
                    train_r2_1 = train_r2
                    test_r2_1 = test_r2
                    train_loss_1 = train_loss
                    test_loss_1 = test_loss
                    train_mae_1 = train_mae
                    test_mae_1 = test_mae
                    torch.save(net.state_dict(), embedding_birnn)
        print(f'train loss {train_loss_1}, test loss {test_loss_1}, train mae '
            f'{train_mae_1:.3f}, test mae {test_mae_1:.3f}, train r2 '
            f'{train_r2_1:.3f}, test r2 {test_r2_1:.3f}')
        del net, features, labels, prob1, prob2, Label
        gc.collect()
        torch.cuda.empty_cache()
    else:
        plotfile_auc = plotfile_auc
        animator_2 = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train auc', 'test auc'])
        for epoch in range(num_epochs):
            train_loss = 0
            for i, (features, abundance, labels, mask) in enumerate(train_iter):
                l, pred = train_batch_ch13(
                    net, features, labels, abundance, mask, loss, trainer, devices)
                net.embedding.weight.data[1:] = fid_embedding
                train_loss += l
                if i == 0:
                    prob1 = pred
                    Label = labels
                else:
                    prob1 = torch.cat((prob1, pred), dim=0)
                    Label = torch.cat((Label, labels), dim=0)
            # scheduler.step()
            train_loss /= (i + 1)
            animator_1.add(epoch + 1, (train_loss.cpu().detach().numpy(), None), plotfile_loss)
            train_auc, train_f1_thresh, train_aupr, train_cm, train_f1, train_mcc = evaluate_auc_gpu(
                Label, prob1)
            animator_2.add(epoch + 1, (train_auc, None), plotfile_auc)
            with torch.no_grad():
                net.eval()
                test_loss = 0
                for i, (features, abundance, labels, mask) in enumerate(test_iter):
                    X = features.to(devices[0])
                    abundance = abundance.to(devices[0])
                    mask = mask.to(devices[0])
                    pred, _ = net(X, abundance, mask)
                    pred = torch.squeeze(pred, dim=1)
                    y = labels.to(devices[0], dtype=torch.int64)
                    pred = torch.sigmoid(pred)
                    l = loss(pred.float(), y.float())
                    test_loss += l
                    if i == 0:
                        prob2 = pred
                        Label = labels
                    else:
                        prob2 = torch.cat((prob2, pred), dim=0)
                        Label = torch.cat((Label, labels), dim=0)
                test_loss /= (i + 1)
            animator_1.add(epoch + 1, (None, test_loss.cpu().detach().numpy()), plotfile_loss)
            test_auc, test_f1_thresh, test_aupr, test_cm, test_f1, test_mcc = evaluate_auc_gpu(
                Label, prob2)
            animator_2.add(epoch + 1, (None, test_auc), plotfile_auc)
            train_list.append(train_auc)
            test_list.append(test_auc)
            test_loss_list.append(test_loss)
            # f1_list.append()
            # 导出最优模型参数
            if len(test_loss_list) == 1:
                train_auc_1 = train_auc
                test_auc_1 = test_auc
                train_loss_1 = train_loss
                test_loss_1 = test_loss
                train_f1_ = train_f1
                test_f1_ = test_f1
                mcc = test_mcc
                cm = test_cm
                pro_ = prob2.cpu().detach().numpy()
                torch.save(net.state_dict(), embedding_birnn)
            else:
                if test_list[-1] > test_auc_1:
                    train_auc_1 = train_auc
                    test_auc_1 = test_auc
                    train_loss_1 = train_loss
                    test_loss_1 = test_loss
                    train_f1_ = train_f1
                    test_f1_ = test_f1
                    mcc = test_mcc
                    cm = test_cm
                    pro_ = prob2.cpu().detach().numpy()
                    torch.save(net.state_dict(), embedding_birnn)
        print(f'train loss {train_loss_1}, test loss {test_loss_1}, train auc '
            f'{train_auc_1:.3f}, test auc {test_auc_1:.3f}',
            f'train f1 {train_f1_}, test f1 {test_f1_}',
            f'mcc {mcc:.3f}, confusion_matrix {cm}')
        del net, features, labels, prob1, prob2, Label
        gc.collect()
        torch.cuda.empty_cache()



def init_transformer_weights(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    if len(w.shape) < 2:
                        nn.init.xavier_normal_(w.unsqueeze(0))
                    else:
                        nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    if len(w.shape) < 2:
                        nn.init.kaiming_normal_(w.unsqueeze(0))
                    else:
                        nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


# foca loss for banlance cross loss
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.6, gamma=2, reduction='mean', devices=None):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = torch.tensor(gamma)
        self.reduction = reduction
        self.devices = devices

    def forward(self, inputs, targets):
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(self.devices[0])
            self.gamma = self.gamma.to(self.devices[0])
        pt = inputs
        alpha = self.alpha
        F_loss = - alpha * (1 - pt) ** self.gamma * targets * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - targets) * torch.log(1 - pt)
        if self.reduction == 'mean':
            F_loss = torch.mean(F_loss)
        elif self.reduction == 'sum':
            F_loss = torch.sum(F_loss)
        return F_loss


class MultiCEFocalLoss(torch.nn.Module):

    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean', devices=None):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num
        self.devices = devices

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)  # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num)  # 获取target的one hot编码
        ids = target.view(-1, 1)
        if predict.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(self.devices[0])
        alpha = self.alpha[ids.data.view(-1)]  # 注意，这里的alpha是给定的一个list(tensor
        # ),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def Attention_biom(
        metadata,
        group,
        train_biom,
        test_biom,
        embedding_birnn,
        plotfile_loss,
        plotfile_auc,
        num_steps=400,
        p_drop=0,
        d_ff=64,  # 中间层向量的大小
        batch_size=128,
        d_model=100,
        n_layers=2,
        n_heads=2,
        numb=1,
        lr=0.0005,
        weight_decay=0,
        num_epochs=100,
        loss=None,
        alpha=0.6,
        glove_embedding=None):
    """_summary_

    Parameters
    ----------
    glove_embedding : _type_
        _description_
    metadata : _type_
        _description_
    train_text : _type_
        _description_
    train_biom : _type_
        _description_
    test_text : _type_
        _description_
    test_biom : _type_
        _description_
    embedding_birnn : _type_
        _description_
    plotfile : _type_
        _description_
    embed_size : int, optional
        _description_, by default 100
    batch_size : int, optional
        _description_, by default 64
    num_steps : int, optional
        _description_, by default 500
    num_hiddens : int, optional
        _description_, by default 100
    num_layers : int, optional
        _description_, by default 2
    lr : float, optional
        _description_, by default 0.01
    num_epochs : int, optional
        _description_, by default 5
    """
    # with open(config_file) as f:
    #     config_defaults = yaml.load(f.read(),
    #                                 Loader=yaml.FullLoader)['parameters']
    # wandb.init(config=config_defaults, settings=wandb.Settings(start_method='fork'))
    # config = wandb.config
    # batch_size = config.batch_size
    # d_model = config.d_model  # embedding size
    # n_layers = config.n_layers
    # n_heads = config.n_heads
    # numb = config.numb
    # lr = config.lr
    # p_drop = config.p_drop
    # d_ff = config.d_ff
    # weight_decay = config.weight_decay

    devices = try_all_gpus(numb)

    if glove_embedding is not None:
        glove_embedding = f"{glove_embedding}_{d_model}.txt"
    train_data = load_data_imdb(train_biom,
                                metadata,
                                group,
                                num_steps)
    test_data = load_data_imdb(test_biom,
                               metadata,
                               group,
                               num_steps)
    train_iter = DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True)
    test_iter = DataLoader(test_data,
                           batch_size=batch_size,
                           shuffle=False)

    fid_dict = train_data()
    net = TransformerEncoder(otu_size=len(fid_dict),
                             seq_len=num_steps+1,
                             d_model=d_model,
                             n_layers=n_layers,
                             n_heads=n_heads,
                             p_drop=p_drop,
                             d_ff=d_ff,
                             pad_id=fid_dict['<pad>'])
    net.apply(init_transformer_weights)

    if glove_embedding is not None:
        # import embedding vector
        glove_embedding = TokenEmbedding(glove_embedding)
        embeds = glove_embedding[fid_dict.idx_to_token]
        net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.data[fid_dict['<pad>']] = 0
    net.embedding.weight.requires_grad = False


    # 训练
    trainer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer, gamma=0.98)
    if loss is None:
        loss = nn.CrossEntropyLoss(reduction='mean')
    if loss == "FocalLoss":
        loss = FocalLoss(alpha=alpha, devices=devices)
    if loss == "BCE_loss":
        loss = nn.BCELoss(weight=None, reduction='mean')
    if loss == "MAE_loss":
        loss = nn.L1Loss(reduction='mean')
    if loss == "MSE_loss":
        loss = nn.MSELoss(reduction='mean')
    train_model(net, train_iter, test_iter, loss,trainer, scheduler, num_epochs, devices,embedding_birnn, plotfile_loss, plotfile_auc)
