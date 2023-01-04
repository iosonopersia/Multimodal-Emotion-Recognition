import torch
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)


class TextERC(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        config = RobertaConfig.from_pretrained('roberta-large', num_labels=num_classes)
        self.roberta = RobertaModel.from_pretrained('roberta-large', add_pooling_layer=False, config=config)
        self.classifier_head = RobertaClassificationHead(config)

    def forward(self, input_ids, attention_mask):
        text_features = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=None, # TODO
            # position_ids=None, # TODO
        )
        text_features = text_features.last_hidden_state
        out = self.classifier_head(text_features)
        return out

    def freeze(self):
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.roberta.parameters():
            param.requires_grad = True
