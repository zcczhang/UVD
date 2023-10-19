import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import tqdm

import numpy as np
import time

from uvd.decomp.decomp import embedding_decomp, DEFAULT_DECOMP_KWARGS
from uvd.models import Preprocessor, get_preprocessor
import uvd.utils as U

from decord import VideoReader


def one_run(video_file: str, preprocessor: Preprocessor):
    t = time.time()
    vr = VideoReader(U.f_expand(video_file), height=224, width=224)
    videos = vr[:].asnumpy()
    load_v_t = time.time() - t

    t = time.time()
    embeddings = preprocessor.process(videos, return_numpy=True)
    preprocess_t = time.time() - t

    t = time.time()
    _, decomp_meta = embedding_decomp(
        embeddings=embeddings,
        fill_embeddings=False,
        return_intermediate_curves=False,
        window_length=100,
        **DEFAULT_DECOMP_KWARGS["embed"],
    )
    decomp_t = time.time() - t
    return load_v_t, preprocess_t, decomp_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file")
    parser.add_argument("--preprocessor_name", default="vip")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print("NO GPU FOUND")
    preprocessor = get_preprocessor(
        args.preprocessor_name, device="cuda" if use_gpu else None
    )
    one_run(args.video_file, preprocessor)

    benchmark_times = dict(load=[], preprocess=[], decomp=[])
    for _ in tqdm.trange(args.n):
        load_v_t, preprocess_t, decomp_t = one_run(args.video_file, preprocessor)
        benchmark_times["load"].append(load_v_t)
        benchmark_times["preprocess"].append(preprocess_t)
        benchmark_times["decomp"].append(decomp_t)

    print({k: (np.mean(v), np.std(v)) for k, v in benchmark_times.items()})
