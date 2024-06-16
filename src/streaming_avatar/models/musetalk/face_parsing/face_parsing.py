import torch
from huggingface_hub import hf_hub_download

from .bisenet import BiSeNet


class FaceParsing:
    def __init__(self, device: torch.device):
        resnet_path = hf_hub_download(
            "vivym/face-parsing-bisenet",
            filename="resnet18-5c106cde.pth",
        )

        net = BiSeNet(resnet_path)
        net.to(device)

        model_path = hf_hub_download(
            "vivym/face-parsing-bisenet",
            filename="79999_iter.pth",
        )
        net.load_state_dict(torch.load(model_path, map_location=device))

        net.eval()

        self.net = net

    def parse(self, images: torch.Tensor) -> torch.Tensor:
        ...
