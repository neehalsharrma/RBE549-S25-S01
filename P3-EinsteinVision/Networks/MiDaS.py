import torch


def refresh_MiDaS():
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo


def load_ZoeDepth():
    repo = "isl-org/ZoeDepth"
    # Zoe_NK
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)