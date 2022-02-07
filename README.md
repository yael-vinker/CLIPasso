# CLIPSketch: Semantically-Aware Abstract Object Sketching

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yael-vinker/CLIPSketch/blob/main/CLIPSketch.ipynb) [![arXiv](https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/)



[[Project Website](https://)]
<br>
<br>

![](repo_images/teaser.png?raw=true)
<!-- <p align='center'>  
  <img src='results/teaser.png' />
</p> -->

<br>

## Installation
### Installation via Docker [Recommended]
You can simply pull the docker image from docker hub, containing all the required libraries and packages:
```bash
docker pull yaelvinker/yv_base_clip
docker run --name clipsketch -it yaelvinker/yv_base_clip /bin/bash
```
Now you should have a running container.
Inside the container, clone the repository:

```bash
git clone https://github.com/yael-vinker/CLIPSketch.git
cd CLIPSketch/
```
Now you are all set and ready to move to the next stage (Run Demo).

### Installation via pip
Note that it is recommended to use the provided docker image, as we rely on diffvg which has specific requirements and does not compile smoothly on every environment.
1.  Clone the repo:
```bash
git clone https://github.com/yael-vinker/CLIPSketch.git
cd CLIPSketch
```
2. Create a new environment and install the libraries:
```bash
python3.6 -m venv clipsketch
source clipsketch/bin/activate
pip install -r requirements.txt
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/openai/CLIP.git
```
3. Install diffvg:
```bash
git clone https://github.com/BachiLi/diffvg
cd diffvg
git submodule update --init --recursive
python setup.py install
```

<br>

## Run Demo

<!-- #### Run a model on your own image -->

The input images to be drawn should be located under "target_images".
To sketch your own image, from CLIPSketch run:
```bash
python run_object_sketching.py --target_file <file_name>
```
A camel image is given for example:
```bash
python run_object_sketching.py --target_file "camel.png"
```
Optional arguments:
* --mask_object: It is recommended to use images without a background, however, if your image contains a background, you can mask it out by using this flag with "1" as an argument.
* --fix_scale: If your image is not squared, it might be cut off, it is recommended to use this flag with 1 as input to automatically fix the scale without cutting the image.
* --num_sketches: As stated in the paper, by default there will be three parallel running scripts to synthesize three sketches and automatically choose the best one. However, for some environments (for example when running on CPU) this might be slow, so you can specify --num_sketches 1 instead.

The resulting sketches will be saved to the "output_sketches" folder, in SVG format.
