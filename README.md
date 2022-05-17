# CLIPasso: Semantically-Aware Object Sketching

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yael-vinker/CLIPasso/blob/main/CLIPasso.ipynb) 
[![arXiv](https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/abs/2202.05822)



[[Project Website](https://clipasso.github.io/clipasso/)]
<br>
<br>
This is the official implementation of CLIPasso, a method for converting an image of an object to a sketch, allowing for varying levels of abstraction. <br>

<br>
<br>

![](repo_images/teaser2.png?raw=true)
At a high level, we define a sketch as a set of Bézier curves and use a differentiable rasterizer ([diffvg](https://github.com/BachiLi/diffvg)) to optimize the parameters of the curves directly with respect to a CLIP-based perceptual loss. <br>
We combine the final and intermediate activations of a pre-trained CLIP model to achieve both geometric and semantic simplifications.
<br> The abstraction degree is controlled by varying the number of strokes.
    
<br>

## Installation
### Installation via Docker [Recommended]
You can simply pull the docker image from docker hub, containing all the required libraries and packages:
```bash
docker pull yaelvinker/clipasso_docker
docker run --name clipsketch -it yaelvinker/clipasso_docker /bin/bash
```
Now you should have a running container.
Inside the container, clone the repository:

```bash
cd /home
git clone https://github.com/yael-vinker/CLIPasso.git
cd CLIPasso/
```
Now you are all set and ready to move to the next stage (Run Demo).

### Installation via pip
Note that it is recommended to use the provided docker image, as we rely on diffvg which has specific requirements and does not compile smoothly on every environment.
1.  Clone the repo:
```bash
git clone https://github.com/yael-vinker/CLIPasso.git
cd CLIPasso
```
2. Create a new environment and install the libraries:
```bash
python3.7 -m venv clipsketch
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
The resulting sketches will be saved to the "output_sketches" folder, in SVG format.

Optional arguments:
* ```--num_strokes``` Defines the number of strokes used to create the sketch, which determines the level of abstraction. The default value is set to 16, but for different images, different numbers might produce better results. 
* ```--mask_object``` It is recommended to use images without a background, however, if your image contains a background, you can mask it out by using this flag with "1" as an argument.
* ```--fix_scale``` If your image is not squared, it might be cut off, it is recommended to use this flag with 1 as input to automatically fix the scale without cutting the image.
* ```--num_sketches``` As stated in the paper, by default there will be three parallel running scripts to synthesize three sketches and automatically choose the best one. However, for some environments (for example when running on CPU) this might be slow, so you can specify --num_sketches 1 instead.
* ```-cpu``` If you want to run the code on the cpu (not recommended as it might be very slow).

<br>
<b>For example, below are optional running configurations:</b>
<br>

Sketching the camel with defauls parameters:
```bash
python run_object_sketching.py --target_file "camel.png"
```
Producing a single sketch of the camel at lower level of abstraction with 32 strokes:
```bash
python run_object_sketching.py --target_file "camel.png" --num_strokes 32 --num_sketches 1
```
Sketching the flamingo with higher level of abstraction, using 8 strokes:
```bash
python run_object_sketching.py --target_file "flamingo.png" --num_strokes 8
```

## Related Work
[CLIPDraw](https://arxiv.org/abs/2106.14843): Exploring Text-to-Drawing Synthesis through Language-Image Encoders, 2021 (Kevin Frans, L.B. Soros, Olaf Witkowski)

[Diffvg](https://github.com/BachiLi/diffvg): Differentiable vector graphics rasterization for editing and learning, ACM Transactions on Graphics 2020 (Tzu-Mao Li, Michal Lukáč, Michaël Gharbi, Jonathan Ragan-Kelley)

## Citation
If you make use of our work, please cite our paper:

```
@misc{vinker2022clipasso,
      title={CLIPasso: Semantically-Aware Object Sketching}, 
      author={Yael Vinker and Ehsan Pajouheshgar and Jessica Y. Bo and Roman Christian Bachmann and Amit Haim Bermano and Daniel Cohen-Or and Amir Zamir and Ariel Shamir},
      year={2022},
      eprint={2202.05822},
      archivePrefix={arXiv},
      primaryClass={cs.GR}
}
```
