from mit_ub.model import BACKBONES, ViT
from mit_ub.model.mlp import relu2


HEAD_DIM = 64

BACKBONES(
    ViT,
    name="chexpert-small",
    in_channels=1,
    dim=512,
    patch_size=16,
    depth=15,
    nhead=512 // HEAD_DIM,
    num_kv_heads=512 // HEAD_DIM,
    dropout=0.1,
    stochastic_depth=0.1,
    bias=False,
    qk_norm=True,
    activation=relu2,
    gate_activation=None,
)

BACKBONES(
    ViT,
    name="chexpert-small-conv",
    in_channels=1,
    dim=512,
    patch_size=16,
    depth=15,
    nhead=512 // HEAD_DIM,
    num_kv_heads=512 // HEAD_DIM,
    dropout=0.1,
    stochastic_depth=0.1,
    bias=False,
    qk_norm=True,
    activation=relu2,
    gate_activation=None,
    target_shape=(512, 512),
    high_res_depth=6,
)
