import torch.nn as nn

from .ChangeDecoder_BRIGHT import ChangeDecoder
from .SemanticDecoder import SemanticDecoder
from .builders import build_backbone, build_head, resolve_decoder_components, resize_to_input


class ChangeMambaMMBDA(nn.Module):
    def __init__(self, output_building, output_damage, pretrained, **kwargs):
        super().__init__()
        self.encoder_1 = build_backbone(pretrained=pretrained, **kwargs)
        self.encoder_2 = build_backbone(pretrained=pretrained, **kwargs)
        norm_layer, ssm_act_layer, mlp_act_layer, clean_kwargs = resolve_decoder_components(kwargs)

        self.decoder_building = SemanticDecoder(
            encoder_dims=self.encoder_1.dims,
            channel_first=self.encoder_1.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )
        self.decoder_damage = ChangeDecoder(
            encoder_dims=self.encoder_2.dims,
            channel_first=self.encoder_2.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )

        self.main_clf = build_head(out_channels=output_damage)
        self.aux_clf = build_head(out_channels=output_building)

    def forward(self, pre_data, post_data):
        pre_features = self.encoder_1(pre_data)
        post_features = self.encoder_2(post_data)

        output_building = resize_to_input(self.aux_clf(self.decoder_building(pre_features)), pre_data)
        output_damage = resize_to_input(self.main_clf(self.decoder_damage(pre_features, post_features)), post_data)
        return output_building, output_damage
