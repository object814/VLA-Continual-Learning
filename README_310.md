# VLA-Continual-Learning
Repo for project Vision-Language-Model Continual Learning

## Setup Instructions

Clone the repository from Github:

```bash
git clone https://github.com/object814/VLA-Continual-Learning.git
```

We are using conda to manage python environment:

```bash
conda create -n ENV_NAME python==3.10
conda activate ENV_NAME
```

Install Pytorch in your conda environment:

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

Install dependencies and add LIBERO dataset:

```bash
pip install -r requirements_310.txt
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

Modify external/openvla/pyproject.toml [dependencies] to avoid error with Python=3.8:
- remove "dlimp @ git+https://github.com/moojink/dlimp_openvla"
- remove "torch==2.2.0", "torchvision>=0.16.0", "torchaudio"
- remove "tensorflow==2.15.0", "tensorflow_datasets==4.9.3", "tensorflow_graphics==2021.12.3"

```bash
cd external/openvla/
pip install -e .
```



You will encounter some error regarding to dependency conflicts, but as long as you see: **Successfully built openvla** and **Successfully installed openvla-0.0.3**, you are good to go

**Note:**\
 We are manually downgrading some of the dependency version to adapt Python=3.8 even though lots of original dependencies in openvla and dlimp requires Python=3.11\
 You might want to change to Python=3.11 but will have to compact with LIBERO

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

