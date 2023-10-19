import subprocess
from typing import Union, List, Optional

import numpy as np
import torch

from .array_tensor_utils import any_stack, any_to_torch_tensor, any_to_numpy
from .file_utils import f_mkdir, f_join, f_remove

__all__ = ["save_video", "ffmpeg_save_video", "compress_video", "VideoTensorWriter"]


def save_video(
    video: Union[np.ndarray, torch.Tensor],
    fname: str,
    fps: Optional[int] = None,
    compress: bool = False,
):
    import torchvision.io
    from einops import rearrange

    fname = f_join(fname)
    video = any_to_torch_tensor(video)
    assert video.ndim == 4, f"must be 4D tensor, {video.shape}"
    assert (
        video.size(1) == 3 or video.size(3) == 3
    ), "shape should be either T3HW or THW3"

    if video.size(1) == 3:
        video = rearrange(video, "T C H W -> T H W C")
    output_fname = fname
    if compress:
        fname = fname.split(".")[0] + "_raw." + fname.split(".")[1]
    torchvision.io.write_video(fname, video, fps=fps)
    if compress:
        compress_video(fname, output_fname, delete_input=True)


def ffmpeg_save_video(
    video: Union[np.ndarray, torch.Tensor], fname: str, fps: Optional[int] = None
):
    """if ffmpeg: error while loading shared libraries: libopenh264.so.5:

    cannot open shared object file: No such file or directory, do `conda
    update ffmpeg`
    """
    import ffmpeg  # pip install ffmpeg-python
    from einops import rearrange

    video = any_to_numpy(video)
    assert video.ndim == 4, f"must be 4D array, {video.shape}"
    assert (
        video.shape[1] == 3 or video.shape[3] == 3
    ), "shape should be either T3HW or THW3"
    if video.shape[1] == 3:
        video = rearrange(video, "T C H W -> T H W C")

    out = ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="rgb24",
        s="{}x{}".format(video.shape[2], video.shape[1], r=fps or 30),
    ).output(
        fname,
        vcodec="libx264",
        crf=28,
        preset="fast",
        pix_fmt="yuv420p",
        loglevel="quiet",
    )
    process = out.run_async(pipe_stdin=True)
    try:
        for frame in video:
            process.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass

    process.stdin.close()
    process.wait()


def compress_video(in_mp4_path: str, out_mp4_path: str, delete_input: bool = True):
    ffmpeg = f"/usr/bin/ffmpeg"
    commands_list = [
        ffmpeg,
        "-v",
        "quiet",
        "-y",
        "-i",
        in_mp4_path,
        "-vcodec",
        "libx264",
        "-crf",
        "28",
        out_mp4_path,
    ]
    assert subprocess.run(commands_list).returncode == 0, commands_list
    if delete_input:
        f_remove(in_mp4_path)


class VideoTensorWriter:
    def __init__(self, folder=".", fps=40):
        self._folder = folder
        self._fps = fps
        self._frames = []

    @property
    def frames(self) -> List[np.ndarray]:
        return self._frames

    def add_frame(self, frame: Union[np.ndarray, torch.Tensor]):
        assert len(frame.shape) == 3
        self._frames.append(frame)

    def clear(self):
        self._frames = []

    def save(
        self,
        step: Union[int, str],
        save: bool = True,
        suffix: Optional[str] = None,
        fps: Optional[int] = None,
        compress: bool = True,
    ) -> str:
        """
        Requires:
            pip install av
        """
        fps = fps or self._fps
        fname = str(step) if suffix is None else f"{step}-{suffix}"
        in_fname = f"{fname}_raw.mp4" if compress else f"{fname}.mp4"
        in_path = f_join(self._folder, in_fname)
        out_path = f_join(self._folder, f"{fname}.mp4")
        if save:
            f_mkdir(self._folder)
            save_video(any_stack(self._frames, dim=0), in_path, fps=fps)
            if compress:
                compress_video(in_path, out_path, delete_input=True)
            self.clear()
        # clear in record env wrapper if not `save`
        return out_path
