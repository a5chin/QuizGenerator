import torch
from torch import nn
from transformers import BertModel


class Bert(nn.Module):
    def __init__(self, out_features) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(
            in_features=self.bert.pooler.dense.in_features, out_features=out_features
        )

    def forward(self, ids, mask) -> torch.Tensor:
        _, x = self.bert(ids, attention_mask=mask)
        x = self.fc(x)
        return x
