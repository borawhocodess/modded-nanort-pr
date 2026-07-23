from .muon import Muon


MUON_BLOCK_MODULES = ("attns", "nbr_attn", "rel_attn", "ffn")


def build_muon_param_groups(model) -> tuple[list, list]:
    """Split model parameters into Muon and AdamW groups.

    Muon group: 2-D weight matrices inside RelationalBlock attention, FFN and
    simplicial modules (blocks.*.attns.*.w{q,k,v,o}.weight,
    blocks.*.{nbr_attn,rel_attn}.w*.weight and blocks.*.ffn.w{1,2,3}.weight).

    AdamW group: everything else — 1-D params, encoders, decoders, norms,
    mask embeddings, classifier head.

    SIGReg projector params (``sigreg.*``) are excluded entirely. They stay fp32
    and are trained by a separate AdamW optimizer in rt/main.py.

    Returns (muon_params, adamw_params).
    """
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if name.startswith("sigreg."):
            continue
        parts = name.split(".")
        in_block = parts[0] == "blocks" and len(parts) >= 2
        in_muon_module = (
            in_block and len(parts) >= 3 and parts[2] in MUON_BLOCK_MODULES
        )
        is_weight_2d = param.ndim == 2

        if in_muon_module and is_weight_2d:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params
