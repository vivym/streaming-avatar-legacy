from pathlib import Path
from typing import Generator

import numpy as np
import torch
import torch.nn as nn
import imageio.v3 as iio
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from face_alignment import FaceAlignment, LandmarksType
from face_alignment.detection.sfd import FaceDetector
from huggingface_hub import hf_hub_download
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from PIL import Image
from torchvision.transforms import functional as TrF, InterpolationMode
from tqdm import tqdm

Avatar = str | Path | Image.Image | list[Image.Image] | list[str | Path] | np.ndarray | torch.Tensor


def load_dwpose_model(device: torch.device) -> nn.Module:
    config_file = Path(__file__).parent / "dwpose" / "rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py"

    ckpt_path = hf_hub_download(
        "yzd-v/DWPose",
        filename="dw-ll_ucoco_384.pth",
    )

    model = init_model(config_file, checkpoint=ckpt_path, device=device)
    model.eval()
    return model


def load_face_detector(device: torch.device) -> FaceDetector:
    fa = FaceAlignment(
        landmarks_type=LandmarksType.TWO_D,
        device=str(device),
        face_detector="sfd",
    )
    return fa.face_detector


@torch.no_grad()
def _to_tensor(
    avatar: Avatar,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    _force_image: bool = False,
) -> Generator[torch.Tensor, None, None]:
    """
    Output spec:
        shape: (num_frames, height, width, 3)
        dtype: uint8
        color: RGB
    """

    if isinstance(avatar, (str, Path)):
        avatar_path = Path(avatar)
        assert avatar_path.exists(), f"Avatar path does not exist: {avatar_path}"
        assert avatar_path.is_file(), f"Avatar path is not a file: {avatar_path}"

        if avatar_path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"
        }:
            frame = iio.imread(avatar_path)

            if avatar_path.suffix.lower() == ".gif":
                frames = frame
            else:
                frames = frame[None]

            frames = torch.from_numpy(frames).to(dtype=dtype, device=device)

            for i in range(0, len(frames), batch_size):
                yield frames[i:i + batch_size]

        elif avatar_path.suffix.lower() in {
            ".mp4", ".webm", ".mov", ".avi", ".flv", ".mkv", ".wmv"
        }:
            frames = []
            for frame in iio.imiter(avatar_path, plugin="pyav"):
                frame = torch.from_numpy(frame).to(dtype=dtype, device=device)
                frames.append(frame)

                if len(frames) >= batch_size:
                    yield torch.stack(frames, dim=0)
                    frames.clear()

            if len(frames) > 0:
                yield torch.stack(frames, dim=0)
        else:
            raise ValueError(f"Unsupported avatar format: {avatar_path}")

    elif isinstance(avatar, Image.Image):
        frames = torch.from_numpy(np.array(avatar)).to(dtype=dtype, device=device)
        yield frames[None]

    elif isinstance(avatar, np.ndarray):
        frames = torch.from_numpy(avatar).to(dtype=dtype, device=device)
        if frames.ndim == 3:
            frames = frames[None]
        else:
            assert not _force_image, "Unsupported avatar format: {avatar}"

        for i in range(0, len(frames), batch_size):
            yield frames[i:i + batch_size]

    elif isinstance(avatar, torch.Tensor):
        if avatar.ndim == 3:
            frames = avatar[None]
        else:
            assert not _force_image, "Unsupported avatar format: {avatar}"
            frames = avatar

        frames = frames.to(dtype=dtype, device=device)

        for i in range(0, len(frames), batch_size):
            yield frames[i:i + batch_size]

    elif isinstance(avatar, (list, tuple)):
        all_frames = []
        for frame in avatar:
            for frames in _to_tensor(frame, batch_size, dtype, device, _force_image=True):
                assert len(frames) == 1, "Only single frames are supported in list mode"
                all_frames.append(frames[0])

                if len(all_frames) >= batch_size:
                    yield torch.stack(all_frames, dim=0)
                    all_frames.clear()

        if len(all_frames) > 0:
            yield torch.stack(all_frames, dim=0)

    else:
        raise ValueError(f"Unsupported avatar format: {avatar}")


@torch.no_grad()
def preprocess_avatar(
    avatar: Avatar,
    bbox_shift: int = 0,
    batch_size: int = 1,
    device: str | torch.device = torch.device("cpu"),
    vae_model_name_or_path: str = "stabilityai/sd-vae-ft-mse",
    verbose: bool = False,
) -> torch.Tensor:
    if isinstance(device, str):
        device = torch.device(device)

    dwpose_model = load_dwpose_model(device)

    face_detector = load_face_detector(device)

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(vae_model_name_or_path)

    def vae_encode(x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            posterior: DiagonalGaussianDistribution = vae.encode(x.to(vae.dtype)).latent_dist
            latents = posterior.sample() * vae.config.scaling_factor
        return latents

    range_minus_list = []
    range_plus_list = []
    for frames in tqdm(
        _to_tensor(avatar, batch_size=batch_size, dtype=torch.uint8, device=device),
        desc="Preprocessing avatar",
        disable=not verbose,
    ):
        detected_faces = face_detector.detect_from_batch(frames.permute(0, 3, 1, 2))

        bboxes = []
        for faces in detected_faces:
            if len(faces) > 0:
                # Only consider the first face
                face = faces[0]
                face = np.clip(face, 0, None)
                x1, y1, x2, y2 = map(int, face[:-1])
                bboxes.append((x1, y1, x2, y2))
            else:
                bboxes.append(None)

        landmark_bboxes = []
        cropped_frames = []
        for frame, face_bbox in zip(frames, bboxes):
            if face_bbox is None:
                landmark_bboxes.append(None)
                continue

            # RGB -> BGR
            results = inference_topdown(dwpose_model, frame[..., ::-1].cpu().numpy())
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_landmark = keypoints[0, 23:91].astype(np.int32)

            half_face_coord = face_landmark[29]
            range_minus = (face_landmark[30] - face_landmark[29])[1]
            range_plus = (face_landmark[29] - face_landmark[28])[1]
            range_minus_list.append(range_minus)
            range_plus_list.append(range_plus)

            if bbox_shift != 0:
                half_face_coord[1] += bbox_shift

            half_face_dist = np.max(face_landmark[:, 1]) - half_face_coord[1]
            upper_bond = half_face_coord[1] - half_face_dist

            x1 = np.min(face_landmark[:, 0])
            y1 = upper_bond
            x2 = np.max(face_landmark[:, 0])
            y2 = np.max(face_landmark[:, 1])

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                # invalid landmark bbox, use face bbox instead
                landmark_bboxes.append(face_bbox)
            else:
                landmark_bboxes.append((x1, y1, x2, y2))

            # h, w, c -> c, h, w
            cropped_frame = frame[y1:y2, x1:x2].permute(2, 0, 1)
            cropped_frame = cropped_frame.to(dtype=torch.float32).div_(255.0)
            cropped_frame = TrF.resize(
                cropped_frame,
                (256, 256),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            cropped_frames.append(cropped_frame)

        cropped_frames = torch.stack(cropped_frames, dim=0)
        normalized_frames = TrF.normalize(
            cropped_frames,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        masked_frames = normalized_frames.clone()
        masked_frames[:, :, 128:] = 0.0

        latents = vae_encode(normalized_frames)
        masked_latents = vae_encode(masked_frames)
        latents = torch.cat([latents, masked_latents], dim=1)

        # Prepare masks
        for landmark_bbox, frame in zip(landmark_bboxes, frames):
            if landmark_bbox is None:
                continue

            x1, y1, x2, y2 = landmark_bbox
            frame[y1:y2, x1:x2] = 0.0

    avg_range_minus = int(np.mean(range_minus_list))
    avg_range_plus = int(np.mean(range_plus_list))

    print("[INFO] bbox_shift adjustment range: [", -avg_range_minus, ",", avg_range_plus, "]")
