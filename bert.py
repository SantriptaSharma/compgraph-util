from torchinfo import summary
from transformers import BertModel, BertConfig

model = BertModel(BertConfig(hidden_size = 1024, num_hidden_layers = 24, num_attention_heads = 16, intermediate_size = 4096))
summary(model)