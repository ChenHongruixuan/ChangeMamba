from .decoder_factory import HierarchicalChangeDecoder


class ChangeDecoder(HierarchicalChangeDecoder):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__(
            encoder_dims=encoder_dims,
            channel_first=channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            fusion_modes_by_stage=[
                ("cat", "interleave", "split"),
                ("cat", "interleave", "split"),
                ("cat", "interleave", "split"),
                ("cat", "interleave", "split"),
            ],
            use_vss_by_stage=[True, True, True, True],
            **kwargs,
        )
