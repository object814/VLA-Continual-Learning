# VLA-Continual-Learning
Repo for project Vision-Language-Model Continual Learning

## Setup Instructions

Clone the repository from Github:

```bash
git clone https://github.com/object814/VLA-Continual-Learning.git
```

We are using conda to manage python environment:

```bash
conda create -n ENV_NAME python==3.8
conda activate ENV_NAME
```

Install Pytorch in your conda environment:

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

Install dependencies and add LIBERO dataset:

```bash
pip install -r requirements.txt
git submodule update --init --recursive
```

Install LIBERO:

```bash
cd external/LIBERO/
pip install -e .
```

Install dlimp:

```bash
cd external/dlimp/
pip install -e .
```

Install openvla:

```bash
cd external/openvla/
pip install -e .
```

**Note:**\
 We are using Python==3.8 even though lots of original dependencies in openvla and dlimp requires Python==3.11, our experience is that install the pip dependencies with conflicts to the latest version under Python==3.8 and it works

To check if LIBERO is installed successfully, run the following command:

```bash
python -c "import libero; print('LIBERO installed successfully')"
```

Download LIBERO full dataset:

```bash
cd external/LIBERO/
python benchmark_scripts/download_libero_datasets.py
```

For more information about LIBERO dataset, refer to [this document](https://lifelong-robot-learning.github.io/LIBERO/html/index.html)

We are using the HuggingFace ```transformers``` AutoClasses for OpenVLA inference and light-weight fine-tune, for more information about OpenVLA, refer to [this link](https://github.com/openvla/openvla?tab=readme-ov-file)

