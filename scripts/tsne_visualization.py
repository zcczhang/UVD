import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from uvd.decomp.decomp import (
    embedding_decomp,
)
from uvd.models.preprocessors import *
import uvd.utils as U

from decord import VideoReader


def vis_2d_tsne(embeddings: np.ndarray, labels: list):
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(embeddings)
    tsne_result_df = pd.DataFrame(
        {"tsne_1": tsne_result[:, 0], "tsne_2": tsne_result[:, 1], "label": labels}
    )
    fig, ax = plt.subplots(1)
    sns.scatterplot(x="tsne_1", y="tsne_2", hue="label", data=tsne_result_df, ax=ax, s=120)
    lim = (tsne_result.min() - 5, tsne_result.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_title(f"{preprocessor.__class__.__name__}")
    plt.show()


def vis_3d_tsne(embeddings: np.ndarray, labels: list):
    tsne = TSNE(n_components=3)
    tsne_result = tsne.fit_transform(embeddings)
    tsne_result_df = pd.DataFrame(
        {
            "tsne_1": tsne_result[:, 0],
            "tsne_2": tsne_result[:, 1],
            "tsne_3": tsne_result[:, 2],
            "label": labels,
        }
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    palette = sns.color_palette("viridis", as_cmap=True)
    unique_labels = tsne_result_df["label"].unique()
    colors = palette(np.linspace(0, 1, len(unique_labels)))
    color_dict = dict(zip(unique_labels, colors))

    for label in unique_labels:
        subset = tsne_result_df[tsne_result_df["label"] == label]
        ax.scatter(
            subset["tsne_1"],
            subset["tsne_2"],
            subset["tsne_3"],
            c=[color_dict[label]],
            label=label,
            s=120,
        )
    ax.set_title(f"{preprocessor.__class__.__name__}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_file",
        default=U.f_join(
         os.path.dirname(__file__), "examples/microwave-bottom_burner-light_switch-slide_cabinet.mp4"
        )
    )
    parser.add_argument("--preprocessor_name", default="vip")
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print("NO GPU FOUND")

    frames = VideoReader(args.video_file, height=224, width=224)[:].asnumpy()
    preprocessor = get_preprocessor(
        args.preprocessor_name, device="cuda" if use_gpu else None
    )
    embeddings = preprocessor.process(frames, return_numpy=True)
    _, decomp_meta = embedding_decomp(
        embeddings=embeddings,
        fill_embeddings=False,
        return_intermediate_curves=False,
        normalize_curve=False,
        min_interval=20,
        smooth_method="kernel",
        gamma=0.1,
    )
    milestone_indices = decomp_meta.milestone_indices
    milestone_rgbs = frames[milestone_indices]

    labels = [
        i
        for i, count in enumerate(milestone_indices)
        for _ in range(count - milestone_indices[i - 1] if i > 0 else count)
    ]
    labels = [labels[0]] + labels

    vis_2d_tsne(embeddings, labels)
    vis_3d_tsne(embeddings, labels)
