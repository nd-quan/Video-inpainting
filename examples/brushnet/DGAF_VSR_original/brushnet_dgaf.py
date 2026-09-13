"""New BrushNet variant; the existing BrushNet implementation is untouched."""
import torch
from diffusers.models.brushnet import BrushNetModel


class DGAFBrushNetModel(BrushNetModel):
    """Condition = [input latent(4), BG(1), aligned clean(4), support(1)].

    Inherits native config/save/load and dense down/mid/up residual paths.
    The input convolution additionally receives noisy latent z_t(4), as in
    the original BrushNet. All new input-channel weights start at zero.
    """
    @classmethod
    def from_baseline(cls, path):
        base = BrushNetModel.from_pretrained(str(path))
        return cls.from_baseline_model(base)

    @classmethod
    def from_baseline_model(cls, base):
        if base.config.in_channels != 4 or base.config.conditioning_channels != 5:
            raise ValueError("Requires baseline BrushNet with 4 latent + 5 condition channels")
        config = dict(base.config)
        config["conditioning_channels"] = 10
        model = cls.from_config(config)
        state = base.state_dict()
        old_weight = state["conv_in_condition.weight"]
        expanded = torch.zeros_like(model.conv_in_condition.weight)
        expanded[:, :old_weight.shape[1]] = old_weight
        state["conv_in_condition.weight"] = expanded
        model.load_state_dict(state, strict=True)
        return model

    def forward(self, sample, timestep, encoder_hidden_states, brushnet_cond, **kwargs):
        if self.config.conditioning_channels != 10 or brushnet_cond.shape[1] != 10:
            raise ValueError("DGAF BrushNet requires 10 condition channels")
        return super().forward(sample, timestep, encoder_hidden_states,
                               brushnet_cond=brushnet_cond, **kwargs)
