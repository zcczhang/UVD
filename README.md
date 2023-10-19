# Universal Visual Decomposer: <br>Long-Horizon Manipulation Made Easy

<div align="center">

[[Website]](https://cec-agent.github.io/)
[[arXiv]](https://zcczhang.github.io/UVD/)
[[PDF]](https://zcczhang.github.io/UVD/assets/pdf/full_paper.pdf)
[[Installation]](#Installation)
[[Usage]](#Usage)
[[BibTex]](#Citation)
______________________________________________________________________



https://github.com/zcczhang/UVD/assets/52727818/5555b99a-76eb-4d76-966f-787af763573a




</div>

# Installation

- Follow the [instruction](https://github.com/openai/mujoco-py#install-mujoco) for installing `mujuco-py` and install the following apt packages if using Ubuntu:
```commandline
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```
- create conda env with Python==3.9
```commandline
conda create -n uvd python==3.9 -y && conda activate uvd
```
- Install any/all standalone visual foundation models from their repos separately *before* setup UVD, in case dependency conflicts, e.g.:
<details><summary>
<a href="https://github.com/facebookresearch/vip">VIP</a>
</summary>
<p>

```commandline
git clone https://github.com/facebookresearch/vip.git
cd vip && pip install -e .
python -c "from vip import load_vip; vip = load_vip()"
```

</p>
</details>

<details><summary>
<a href="https://github.com/facebookresearch/r3m">R3M</a>
</summary>
<p>

```commandline
git clone https://github.com/facebookresearch/r3m.git
cd r3m && pip install -e .
python -c "from r3m import load_r3m; r3m = load_r3m('resnet50')"
```

</p>
</details>

<details><summary>
<a href="https://github.com/penn-pal-lab/LIV">LIV (& CLIP)</a>
</summary>
<p>

```commandline
git clone https://github.com/penn-pal-lab/LIV.git
cd LIV && pip install -e . && cd liv/models/clip && pip install -e .
python -c "from liv import load_liv; liv = load_liv()"
```

</p>
</details>


<details><summary>
<a href="https://github.com/facebookresearch/eai-vc">VC1</a>
</summary>
<p>

```commandline
git clone https://github.com/facebookresearch/eai-vc.git 
cd eai-vc && pip install -e vc_models
```

</p>
</details>

<details><summary>
<a href="https://github.com/facebookresearch/dinov2">DINOv2</a> and <a href="https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html">ResNet</a> pretrained with ImageNet-1k are directly loaded via <a href="https://pytorch.org/hub/">torch hub</a> and <a href="https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html">torchvision</a>.
</summary></details>

- Under *this* UVD repo directory, install other dependencies
```commandline
pip install -e .
```

# Usage

We provide a simple API for decompose RGB videos:

```python
import torch
import uvd

# (N sub-goals, *video frame shape)
subgoals = uvd.get_uvd_subgoals(
    "/PATH/TO/VIDEO.*",   # video filename or (L, *video frame shape) video numpy array
    preprocessor_name="vip",    # Literal["vip", "r3m", "liv", "clip", "vc1", "dinov2"]
    device="cuda" if torch.cuda.is_available() else "cpu",  # device for loading frozen preprocessor
    return_indices=False,   # True if only want the list of subgoal timesteps
)
```

or run
```commandline
python demo.py
```
to host a Gradio demo locally with different choices of visual representations. 

## Simulation Data

We post-processed the data released from original [Relay-Policy-Learning](https://github.com/google-research/relay-policy-learning/tree/master) that keeps the successful trajectories only and adapt the control and observations used in our paper by:
```commandline
python datasets/data_gen.py raw_data_path=/PATH/TO/RAW_DATA
```

Also consider to force set `Builder = LinuxCPUExtensionBuilder` to `Builder = LinuxGPUExtensionBuilder` in `PATH/TO/CONDA/envs/uvd/lib/python3.9/site-packages/mujoco_py/builder.py` to enable (multi-)GPU acceleration.


## Runtime Benchmark

Since UVD's goal is to be an off-the-shelf method applying to *any* existing policy learning frameworks and models, across BC and RL, we provide minimal scripts for benchmarking the runtime showing negligible runtime under `./scripts` directory:
```commandline
python scripts/benchmark_decomp.py /PATH/TO/VIDEO
```
and passing `--preprocessor_name` with other preprocessors (default `vip`) and `--n` for the number of repeated iterations (default `100`).

For inference or rollouts, we benchmark the runtime by 
```commandline
python scripts/benchmark_inference.py
```
and passing `--policy` for using MLP or causal GPT policy; `--preprocessor_name` with other preprocessors (default `vip`); `--use_uvd` as boolean arg for whether using UVD or no decomposition (i.e. final goal conditioned); and `--n` for the number of repeated iterations (default `100`). The default episode horizon is set to 300. We found that running in the terminal would be almost 2s slower every episode than directly running with python IDE (e.g. PyCharm, under the script directory and run as script instead of module), but the general trend that including UVD introduces negligible extra runtime still holds true. 

# Citation
If you find this project useful in your research, please consider citing:

```bibtex
@misc{zhang2023universal,
  title  = {Universal Visual Decomposer: Long-Horizon Manipulation Made Easy}, 
  author = {Zichen Zhang and Yunshuang Li and Osbert Bastani and Abhishek Gupta and Dinesh Jayaraman and Yecheng Jason Ma and Luca Weihs},
  title  = {Universal Visual Decomposer: Long-Horizon Manipulation Made Easy},
  year   = {2023},
  eprint = {arXiv:2310.08581},
}
```
