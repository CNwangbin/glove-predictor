import torch
from torch import nn
from torch.nn import BCELoss, MSELoss
import torch.nn.functional as F
from dataset import TokenEmbedding

class LightAttentionPredictor(nn.Module):
    def __init__(self,
                 fid_dict=None,
                 d_model=100,
                 d_mlp_atten=50,
                 d_mlp_embed=50,
                 d_mlp_pred=50,
                 p_drop=0.1,
                 activation=nn.ReLU(),
                 classification=True,
                 glove_embedding_path="data/abundance-percentile_100.txt",
                 embedding_freeze=True):
        super(LightAttentionPredictor, self).__init__()
        if fid_dict is None:
            raise ValueError("fid_dict is None")
        self.fid_dict = fid_dict
        otu_size = len(fid_dict)
        pad_id=fid_dict['<pad>']

        # layers
        self.embedding = nn.Embedding(otu_size, d_model, padding_idx=pad_id)
        if glove_embedding_path is not None:
            self.embedding_glove_init(glove_embedding_path,embedding_freeze)
        else:
            if embedding_freeze:
                self.embedding.weight.requires_grad = False
        self.activation = activation
        self.mlp_atten = nn.Sequential(
            nn.Linear(d_model, d_mlp_atten),
            self.activation,
            nn.Dropout(p_drop),
            nn.Linear(d_mlp_atten, 1),
        )
        self.mlp_embed = nn.Sequential(
            nn.Linear(d_model, d_mlp_embed),
            self.activation,
            nn.Dropout(p_drop),
            nn.Linear(d_mlp_embed, d_model),
        )
        self.mlp_pred = nn.Sequential(
            nn.Linear(d_model, d_mlp_pred),
            self.activation,
            nn.Dropout(p_drop),
            nn.Linear(d_mlp_pred, 1),
        )
        self.classification = classification
        if classification is True:
            self.lossfn = BCELoss()
        else:
            self.lossfn = MSELoss()

    def forward(self, token_ids, token_abundance, mask, labels=None):
        # |token_ids| : (batch_size, num_steps)
        # |token_abundance| : (batch_size, num_steps)
        # |mask| : (batch_size, num_steps)
        embeddings = self.embedding(token_ids)
        atten_score = self.activation(self.mlp_atten(embeddings).squeeze(-1) * token_abundance)
        atten_score = atten_score.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(atten_score, dim=-1)
        # |attention_weights| : (batch_size, num_steps)
        embeddings = self.mlp_embed(embeddings)
        pooled_embedding = (embeddings * attention_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.mlp_pred(pooled_embedding).squeeze(-1)
        # |logits| : (batch_size, d_model)
        loss = None
        if labels is not None:
            labels = labels.float()
            if self.classification is False:
                loss = self.lossfn(logits, labels)
                return loss, logits, attention_weights
            else:
                outputs = torch.sigmoid(logits)
                loss = self.lossfn(outputs, labels)
                return loss, outputs, attention_weights

    def predict(self, token_ids, token_abundance, mask):
        pass

    def embedding_glove_init(self, glove_embedding_path="data/abundance-percentile_100.txt",embedding_freeze=True):
        glove_embedding = TokenEmbedding(glove_embedding_path)
        embeds = glove_embedding[self.fid_dict.idx_to_token]
        self.embedding.weight.data.copy_(embeds)
        self.embedding.weight.data[self.fid_dict['<pad>']] = 0
        if embedding_freeze:
            self.embedding.weight.requires_grad = False



if __name__ == '__main__':

    from dataset import TokenEmbedding, ImdbDataset, Fid
    from torch.utils.data import DataLoader

    with open('data/microbes_intersection.txt', 'r') as f:
        microbes = f.read().splitlines()

    train_data = ImdbDataset(biom_table='data/crc_cross_dataset_new/train_1.biom',
                                metadata='data/crc_cross_dataset_new/metadata.txt',
                                group='group',
                                microbes=microbes,
                                num_steps=500)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True,collate_fn=train_data.collate_fn)
    batch_data = next(iter(train_loader))
    fid_dict = train_data()

    model = LightAttentionPredictor(fid_dict=fid_dict,
                             d_model=100,
                             d_mlp_atten=50,
                             d_mlp_pred=50,
                             p_drop=0.1,
                             activation=nn.ReLU(),
                             classification=True,
                             glove_embedding_path="data/abundance-percentile_100.txt",
                             embedding_freeze=True)
    for batch_data in train_loader:
        loss, outputs, attention_weights = model(batch_data['token_ids'], batch_data['token_abundance'], batch_data['mask'], batch_data['labels'])

