# Remote sensing image object detection system

Based on ultralytics 8.3.65, the ultralytics project link is https://github.com/ultralytics/ultralytics/

The original project's README.md and README.zh-CN.md files have been removed, and the current README.md belongs to the rsiod-system project

## Environment Configuration

1.Create a virtual environment
```shell
conda create -p D:\CondaEnvironments\RSIOD-envs
```

2.Specify Python version 3.10
```shell
cd D:\Projects\rsiod-system
pip install python=3.10
```

3.Install PyTorch
```shell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

4.Install the ultralytics project from source code
```shell
pip install -e .
```

5.Install PyQt5
```shell
pip install PyQt5==5.15.9
```

## Usage

1.Configure relevant parameters in `src/config.yml`.

2.Run the `src/main.py` file.
