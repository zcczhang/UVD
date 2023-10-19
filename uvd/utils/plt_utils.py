import io
from typing import List, Union, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "round_metrics",
    "debug_texts_to_frame",
    "confirmHW3",
    "add_boundary_from_success",
    "plt_to_numpy",
]


def round_metrics(metrics: Any, n: int = 4) -> Any:
    if isinstance(metrics, (float, int)):
        return round(metrics, n)
    elif isinstance(metrics, dict):
        return {k: round_metrics(v, n) for k, v in metrics.items()}
    elif isinstance(metrics, (list, tuple)):
        return type(metrics)([round_metrics(m, n) for m in metrics])
    elif isinstance(metrics, np.ndarray):
        return round_metrics(metrics.tolist(), n)
    else:
        return metrics


def debug_texts_to_frame(
    frame: np.ndarray, debug_text: Union[List[str], dict], **kwargs
) -> np.ndarray:
    debug_text = round_metrics(debug_text)
    if isinstance(debug_text, dict):
        text_list = [f"{k}: {v}" for k, v in debug_text.items()]
    else:
        text_list = debug_text
    org_x_init = kwargs.pop("org_x_init", 10)
    org_x_increment = kwargs.pop("org_x_increment", 0)
    org_y_init = kwargs.pop("org_y_init", 30)
    org_y_increment = kwargs.pop("org_y_increment", 20)

    cv2_kwargs = dict(
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore
        fontScale=0.5,
        color=(0, 255, 0),
        thickness=1,
        lineType=cv2.LINE_AA,  # type: ignore
    )
    cv2_kwargs.update(kwargs)
    frame = frame.copy()
    for i, txt in enumerate(text_list):
        cv2.putText(  # type: ignore
            frame,
            txt,
            (org_x_init + i * org_x_increment, org_y_init + i * org_y_increment),
            **cv2_kwargs,
        )
    return frame


def confirmHW3(frame: np.ndarray):
    assert frame.ndim == 3 and (frame.shape[0] == 3 or frame.shape[2] == 3), frame.shape
    return frame if frame.shape[-1] == 3 else frame.transpose([1, 2, 0])


def add_boundary_from_success(
    frame: np.ndarray,
    success: bool,
    padding: int = 5,
    success_color: tuple = (0, 255, 0),
    fail_color: tuple = (255, 0, 0),
) -> np.ndarray:
    color = np.array(success_color) if success else np.array(fail_color)
    h, w, c = frame.shape
    new_h, new_w = h + 2 * padding, w + 2 * padding
    new_frame = np.full((new_h, new_w, c), color, dtype=np.uint8)
    new_frame[padding:-padding, padding:-padding] = frame
    return new_frame


def plt_to_numpy(fig: plt.Figure, close: bool = True):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw")
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()
    if close:
        plt.close(fig)
    return img_arr
