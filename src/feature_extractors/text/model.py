import torch
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)


class TextERC(torch.nn.Module):
    def __init__(self, pretrained_model='roberta-base', num_classes=7):
        super().__init__()
        self.is_frozen = False
        config = RobertaConfig.from_pretrained(pretrained_model, num_labels=num_classes)
        self.embeddings_size = config.hidden_size

        self.roberta = RobertaModel.from_pretrained(pretrained_model, add_pooling_layer=False, config=config)
        self.classifier_head = RobertaClassificationHead(config)

    def forward(self, input_ids, attention_mask):
        text_features = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_features.last_hidden_state
        out = self.classifier_head(text_features)
        return out

    def freeze(self):
        if not self.is_frozen:
            for param in self.roberta.parameters():
                param.requires_grad = False
            self.is_frozen = True

    def unfreeze(self):
        if self.is_frozen:
            for param in self.roberta.parameters():
                param.requires_grad = True
            self.is_frozen = False
