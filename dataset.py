import gc
from typing import Dict
import pandas as pd
import biom
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset



def read_imdb(biom_table, metadata, group, microbes):
    """读取OTU table和样本标签"""
    labels = []
    table = biom.load_table(biom_table)
    fid = table.ids(axis="observation")

    # Filter the table to only include the specified microbes
    # Compute the intersection of the microbes in the table and the specified microbes
    microbes_to_include = set(fid).intersection(set(microbes))
    table = table.filter(microbes_to_include, axis='observation', inplace=False)

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


class ImdbDataset(Dataset):
    """

    Returns
    -------
    _type_
        _description_ 
    """

    def __init__(self, biom_table, metadata, group, microbes, num_steps=500):

        otu, fid, labels = read_imdb(biom_table, metadata, group, microbes)
        self.fid = Fid(fid, reserved_tokens=['<pad>'])
        self.features, self.abundance, self.mask = self.truncate_pad(otu, fid, num_steps)
        self.labels = torch.tensor(labels)
        self.otu = otu

    def truncate_pad(self, otu, fid, num_steps):
        features = np.zeros((otu.shape[0], num_steps), dtype=int)
        abundance = np.zeros((otu.shape[0], num_steps))
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

        input_mask = torch.ones(features.shape[0], 
                                features.shape[1], 
                                dtype=torch.long)
        input_mask[features == self.fid['<pad>']] = 0 # mask padding tokens
        return torch.tensor(features), torch.tensor(abundance, dtype=torch.float32), input_mask

    def __call__(self):
        return self.fid

    def __getitem__(self, index):
        return self.features[index, ], self.abundance[index, ], self.labels[index], self.mask[index]

    def __len__(self):
        """ Returns the number of sample. """
        return self.otu.shape[0]

    def collate_fn(self, examples) -> Dict[str, torch.Tensor]:
        """Function to transform tokens string to IDs; it depends on the model
        used."""
        token_ids = [ex[0] for ex in examples]
        token_abundance = [ex[1] for ex in examples]
        mask = [ex[3] for ex in examples]
        labels = [ex[2] for ex in examples]

        encoded_inputs = {
            'token_ids': torch.stack(token_ids),
            'token_abundance': torch.stack(token_abundance),
            'mask': torch.stack(mask),
            'labels': torch.stack(labels)
        }
        return encoded_inputs

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



if __name__ == '__main__':
    # read microbes_intersection from file
    with open('data/microbes_intersection.txt', 'r') as f:
        microbes = f.read().splitlines()

    train_data = ImdbDataset(biom_table='data/crc_cross_dataset_new/train_1.biom',
                                metadata='data/crc_cross_dataset_new/metadata.txt',
                                group='group',
                                microbes=microbes,
                                num_steps=500)
