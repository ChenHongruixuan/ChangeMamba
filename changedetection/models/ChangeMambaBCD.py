import torch.nn as nn

from .ChangeDecoder import ChangeDecoder
from .builders import build_backbone, build_head, resolve_decoder_components, resize_to_input


class ChangeMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super().__init__()
        self.encoder = build_backbone(pretrained=pretrained, **kwargs)
        norm_layer, ssm_act_layer, mlp_act_layer, clean_kwargs = resolve_decoder_components(kwargs)
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )
        self.main_clf = build_head(out_channels=2)

    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        output = self.main_clf(self.decoder(pre_features, post_features))
        return resize_to_input(output, pre_data)
