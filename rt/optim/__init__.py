from .muon import Muon


def build_muon_param_groups(model) -> tuple[list, list]:
    """Split model parameters into Muon and AdamW groups.

    Muon group: 2-D weight matrices inside RelationalBlock attention and FFN
    modules (blocks.*.attns.*.w{q,k,v,o}.weight and blocks.*.ffn.w{1,2,3}.weight).

    AdamW group: everything else — 1-D params, encoders, decoders, norms,
    mask embeddings, classifier head.

    SIGReg projector params (``sigreg.*``) are excluded entirely — they are fp32
    and trained by a separate AdamW optimizer (see rt/main.py).

    Returns (muon_params, adamw_params).
    """
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if name.startswith("sigreg."):
            continue
        parts = name.split(".")
        in_block = parts[0] == "blocks" and len(parts) >= 2
        in_attn = in_block and len(parts) >= 3 and parts[2] == "attns"
        in_ffn = in_block and len(parts) >= 3 and parts[2] == "ffn"
        is_weight_2d = param.ndim == 2

        if (in_attn or in_ffn) and is_weight_2d:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params
