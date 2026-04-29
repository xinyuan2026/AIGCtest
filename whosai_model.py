import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class WhosAIModel(nn.Module):
    def __init__(self, base_model_name="roberta-large", num_classes=5):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.base.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # 取CLS向量
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits