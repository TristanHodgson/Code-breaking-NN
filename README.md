# Code Breaking Neural Network

Please see the [write up](write-up/main.pdf) for more information. Please note that this is currently a work in progress due to limited time to train to produce formal results.


## Usage

1. Clone the repository

```bash
git clone https://github.com/TristanHodgson/Code-breaking-NN.git
cd Code-breaking-NN
```


2. Setup a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install PyTorch, go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and run the command for your system. For example

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

4. Install the requirements

```bash
pip install -r requirements.txt
```

If on linux then we may also install triton

```bash
pip install triton
```