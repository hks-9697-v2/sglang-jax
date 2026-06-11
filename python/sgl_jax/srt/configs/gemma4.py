from transformers import PretrainedConfig


class Gemma4Config(PretrainedConfig):
    model_type = "gemma4"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = PretrainedConfig(**text_config)
        self.text_config = text_config
        if self.text_config is not None:
            tc = self.text_config
            tc.swa_head_dim = getattr(tc, "head_dim", None)
            tc.head_dim = getattr(tc, "global_head_dim", tc.swa_head_dim)
            tc.swa_num_key_value_heads = getattr(tc, "num_key_value_heads", None)
            tc.num_key_value_heads = getattr(tc, "num_global_key_value_heads", tc.swa_num_key_value_heads)

        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig(**vision_config)
        self.vision_config = vision_config

        super().__init__(**kwargs)
