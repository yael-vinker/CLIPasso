# CLIPSketch: Semantically-Aware Abstract Object Sketching

Official implementation.
### [Project]() | [Paper]()
<br>
<br>

![](readme_images/teaser.png?raw=true)
<!-- <p align='center'>  
  <img src='results/teaser.png' />
</p> -->
## Installation
#### Installation via Docker [Recommended]
You cam simplt pull the docker image from docker hub, containing all the required libraries and packages:
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


<!-- Using default tag: latest
latest: Pulling from yaelvinker/yv_base_clip
Digest: sha256:2b45345482a13e3b1c0eb095b93b30d552d8c6016ca5a178b27bcd6ba5071aed
Status: Image is up to date for yaelvinker/yv_base_clip:latest
docker.io/yaelvinker/yv_base_clip:latest

docker start sketch_update
docker run --name unpaired_tmo -it -p 8888:8888 unpaired_tmo /bin/bash
``` -->

#### Installation via Pip/Conda/Virtualenv
1.  Clone the repo:
```bash
git clone https://github.com/yael-vinker/unpaired_hdr_tmo.git
cd unpaired_hdr_tmo
```
2. Create a new environment and install the libraries:
```bash
python3.6 -m venv hdr_venv
source hdr_venv/bin/activate
pip install -r requirements.txt
```

<br>
<br>

## Quickstart (Run Demo Locally)

<!-- #### Run a model on your own image -->

The input images to be drawn should be located under "target_images".
To sketch your own image, run:
```bash
python run_object_sketching.py --target_file <file_name>
```
A cammel image is given for example:
```bash
python run_object_sketching.py --target_file "camel.png"
```
Optional arguments:
* --mask_object: It is recommended to use images without a background, however, if your image contains a background, you can mask it out by using this flag with "1" as argument.
* --fix_scale: If your image is not squred, it might be cut off, it is recommended to use this flag with 1 as input to automatically fix the scale without cutting the image.
* --num_sketches: As stated in the paper, by default there will be three parallel running scripts to synthesis three sketches and automatically choose the best one. However, for some environemnets (for example when running on CPU) this might be slow, so you can specify --num_sketches 1 instead.

The resulting sketches will be saved to "output_sketches" folder, in svg format.
