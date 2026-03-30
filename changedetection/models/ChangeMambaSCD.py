import torch.nn as nn

from .ChangeDecoder import ChangeDecoder
from .SemanticDecoder import SemanticDecoder
from .builders import build_backbone, build_head, resolve_decoder_components, resize_to_input


class ChangeMambaSCD(nn.Module):
    def __init__(self, output_cd, output_clf, pretrained, **kwargs):
        super().__init__()
        self.encoder = build_backbone(pretrained=pretrained, **kwargs)
        norm_layer, ssm_act_layer, mlp_act_layer, clean_kwargs = resolve_decoder_components(kwargs)

        self.decoder_bcd = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )
        self.decoder_T1 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )
        self.decoder_T2 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )

        self.main_clf_cd = build_head(out_channels=output_cd)
        self.aux_clf = build_head(out_channels=output_clf)

    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        output_bcd = resize_to_input(self.main_clf_cd(self.decoder_bcd(pre_features, post_features)), pre_data)
        output_T1 = resize_to_input(self.aux_clf(self.decoder_T1(pre_features)), pre_data)
        output_T2 = resize_to_input(self.aux_clf(self.decoder_T2(post_features)), post_data)
        return output_bcd, output_T1, output_T2
