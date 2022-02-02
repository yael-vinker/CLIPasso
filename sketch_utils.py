import numpy as np
import skimage
import skimage.io
import os
import wandb
from PIL import Image
import matplotlib.pyplot as plt
from loss import Loss
import math
import pydiffvg
from torchvision import datasets, models, transforms
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import clip
import pandas as pd
import imageio
import PIL
from U2Net_.model import U2NET


def imwrite(img, filename, gamma = 2.2, normalize = False, use_wandb=False, wandb_name="", step=0, input_im=None):
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(img, np.ndarray):
        img = img.data.numpy()
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng
    img = np.clip(img, 0.0, 1.0)
    if img.ndim==2:
        #repeat along the third dimension
        img=np.expand_dims(img,2)
    img[:, :, :3] = np.power(img[:, :, :3], 1.0/gamma)
    img = (img * 255).astype(np.uint8)
    
    skimage.io.imsave(filename, img, check_contrast=False)
    images = [wandb.Image(Image.fromarray(img), caption="output")]
    if input_im is not None and step == 0:
        images.append(wandb.Image(input_im, caption="input"))
    if use_wandb:
        wandb.log({wandb_name + "_": images}, step=step)


def plot_batch(inputs, outputs, args, step, use_wandb, title):
    plt.figure()
    plt.subplot(2,1,1)
    grid = make_grid(inputs.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("inputs")

    plt.subplot(2,1,2)
    grid = make_grid(outputs, normalize=False, pad_value=2)
    npgrid = grid.detach().cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("outputs")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"output": wandb.Image(plt)}, step=step)
    plt.savefig("{}/{}".format(args.output_dir, title))
    plt.close()


def log_input(use_wandb, epoch, inputs, output_dir):
    grid = make_grid(inputs.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.tight_layout()
    if use_wandb:
        wandb.log({"input": wandb.Image(plt)}, step=epoch)
    plt.close()
    input_ = inputs[0].cpu().clone().detach().permute(1,2,0).numpy()
    input_ = (input_ - input_.min()) / (input_.max() - input_.min())
    imageio.imwrite("{}/{}.png".format(output_dir, "input"), input_)



def log_sketch_summary_final(path_svg, use_wandb, device, epoch, loss, title):
    canvas_width, canvas_height, shapes, shape_groups = load_svg(path_svg)
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width, # width
                    canvas_height, # height
                    2,   # num_samples_x
                    2,   # num_samples_y
                    0,   # seed
                    None,
                    *scene_args)

    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    plt.imshow(img.cpu().numpy())
    plt.axis("off")
    plt.title(f"{title} best res [{epoch}] [{loss}.]")
    if use_wandb:
        wandb.log({title: wandb.Image(plt)})
    plt.close()


def log_sketch_summary(sketch, title, use_wandb):
    plt.figure()
    grid = make_grid(sketch.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    if use_wandb:
        wandb.run.summary["best_loss_im"] = wandb.Image(plt)
    plt.close()


def plot_triplet(sketches, positive, negative, args, use_wandb):
    plt.figure()
    plt.subplot(1,3,1)
    grid = make_grid(positive, normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("target (positive)")

    plt.subplot(1,3,2)
    grid = make_grid(sketches, normalize=False, pad_value=2)
    npgrid = grid.detach().cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("output sketch")

    plt.subplot(1,3,3)
    grid = make_grid(negative, normalize=False, pad_value=2)
    npgrid = grid.detach().cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("negative")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"triplet_im": wandb.Image(plt)})
    plt.savefig("{}/{}".format(args.output_dir, "triplet"))
    plt.close()


def load_svg(path_svg):
    svg = os.path.join(path_svg)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg)
    return canvas_width, canvas_height, shapes, shape_groups


def plot_sorted(args, loss_func, canvas_width, canvas_height, shapes, original_target, t):
    with torch.no_grad():
        loss_per_stroke = []
        for i, s in enumerate(shapes):
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            shapes_=[]
            shape_groups =[]
            for j in range(len(shapes)):
                if j != i:
                    shapes_.append(shapes[j])
                    shape_groups.append(pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes_) - 1]), fill_color = None, stroke_color = stroke_color))
        # for i, s in enumerate(shapes):
        #     shapes_ = [s]
        #     path_group_ = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes_) - 1]), fill_color = None, stroke_color = stroke_color)
        #     shape_groups = [path_group_]
            img = render_warp(canvas_width, canvas_height, shapes_, shape_groups)
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
            img2 = img.clone()
            img = img[:, :, :3]
            # Convert img from HWC to NCHW
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2) # NHWC -> NCHW

            loss = loss_func(img, t, use_wandb=False, print_=False, compute_grad_ratio=False)
            loss_per_stroke.append(loss.item())
            utils.imwrite(img2.cpu(), '{}/{}_sorted_{}.jpg'.format(args.output_dir, i, loss.item()), gamma=gamma, use_wandb=args.use_wandb, wandb_name="sorted", step=t, input_im=original_target)

        blue = torch.tensor([0,1,0,1])
        red = torch.tensor([1,0,0,1])
        inds = np.argsort(loss_per_stroke)[::-1]
        ordered_shape = []
        ordered_shape_groups = []
        for c, i in enumerate(inds):
            shapes_ = [shapes[i]]
            ordered_shape.append(shapes[i])
            stroke_color = (1 - c / (len(inds) - 1)) * blue + (c / (len(inds) -1 )) * red
            ordered_shape_groups.append(pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(ordered_shape) - 1]), fill_color = None, stroke_color = stroke_color))
        img = render_warp(canvas_width, canvas_height, ordered_shape, ordered_shape_groups)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        utils.imwrite(img.cpu(), '{}/final_sorted.jpg'.format(args.output_dir), gamma=gamma, use_wandb=args.use_wandb, wandb_name="sorted", step=t, input_im=original_target)


def get_sorted_strokes_by_loss(args, loss_func, canvas_width, canvas_height, shapes, t):
    args.percep_loss="lpips"
    args.perceptual_weight=1
    args.train_with_clip=0
    args.clip_weight=0
    loss_func = Loss(args)
    with torch.no_grad():
        loss_per_stroke = []
        # for i, s in enumerate(shapes):
        #     shapes_ = [s]
        #     path_group_ = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes_) - 1]), fill_color = None, stroke_color = stroke_color)
        #     shape_groups = [path_group_]
        for i, s in enumerate(shapes):
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            shapes_=[]
            shape_groups =[]
            for j in range(len(shapes)):
                if j != i:
                    shapes_.append(shapes[j])
                    shape_groups.append(pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes_) - 1]), fill_color = None, stroke_color = stroke_color))
            img = render_warp(canvas_width, canvas_height, shapes_, shape_groups)
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
            img = img[:, :, :3]
            # Convert img from HWC to NCHW
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2) # NHWC -> NCHW

            loss = loss_func(img, t, use_wandb=False, print_=False)
            loss_per_stroke.append(loss.item())
    inds = np.argsort(loss_per_stroke)[::-1]
    args.percep_loss="none"
    args.perceptual_weight=0
    args.train_with_clip=1
    args.clip_weight=1
    return inds


def lr_func_cosine(cur_epoch, args):
    offset = args.WARMUP_EPOCHS if args.WARMUP_EPOCHS else 0.0
    lr = (
        args.COSINE_END_LR
        + (args.BASE_LR - args.COSINE_END_LR)
        * (
            math.cos(
                math.pi * (cur_epoch - offset) / (args.num_iter - offset)
            )
            + 1.0
        )
        * 0.5
    )
    return lr


def get_epoch_lr(cur_epoch, args):
    lr = lr_func_cosine(cur_epoch, args)
    # Perform warm up.
    if cur_epoch < args.WARMUP_EPOCHS:
        lr_start = args.WARMUP_START_LR
        lr_end = lr_func_cosine(args.WARMUP_EPOCHS, args)
        alpha = (lr_end - lr_start) / args.WARMUP_EPOCHS
        lr = cur_epoch * alpha + lr_start
    return lr


def load_dataset(data_dir, batch_size):
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    # expected range should be [0,1] after loading since each loss needs a different normalisation

    image_datasets = datasets.ImageFolder(os.path.join(data_dir), data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)
    print("[{}] images were loaded".format(len(image_datasets)))
    return dataloaders


def plot_attn_dino(attn, threshold_map, inputs, inds, use_wandb, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10,5))
   
    plt.subplot(2, attn.shape[0] + 2, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:,1], inds[:,0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, 2)
    plt.imshow(attn.sum(0).numpy(), interpolation='nearest')
    plt.title("atn map sum")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 3)
    plt.imshow(threshold_map[-1].numpy(), interpolation='nearest')
    plt.title("prob sum")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 4)
    plt.imshow(threshold_map[:-1].sum(0).numpy(), interpolation='nearest')
    plt.title("thresh sum")
    plt.axis("off")
    
    for i in range(attn.shape[0]):
        plt.subplot(2, attn.shape[0] + 2, i + 3)
        plt.imshow(attn[i].numpy())
        plt.axis("off")
        plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 1 + i + 4)
        plt.imshow(threshold_map[i].numpy())
        plt.axis("off")
    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()

def plot_attn_clip(attn, threshold_map, inputs, inds, use_wandb, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10,5))
   
    plt.subplot(1, 3, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:,1], inds[:,0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("atn map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    threshold_map_ = (threshold_map - threshold_map.min()) / (threshold_map.max() - threshold_map.min())
    plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
    plt.title("prob softmax")
    plt.scatter(inds[:,1], inds[:,0], s=10, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()

def plot_atten(attn, threshold_map, inputs, inds, use_wandb, output_path, saliency_model):
    if saliency_model == "dino":
        plot_attn_dino(attn, threshold_map, inputs, inds, use_wandb, output_path)
    elif saliency_model == "clip":
        plot_attn_clip(attn, threshold_map, inputs, inds, use_wandb, output_path)


classes_to_folders_sketchcoco = {
    "cat": "17",
    "dog": "18",
    "horse": "19",
    "sheep": "20",
    "cow": "21",
    "elephant": "22",
    "zebra": "24",
    "giraffe": "25"
}
classes = [
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "zebra",
    "giraffe"
]
num_classes = len(classes)

def get_files_path(parent_path, names_per_class, label, classes_, folder_names, num_images_per_class):
    category_ = []
    images_path_ = []
    labels_ = []
    for j, c in enumerate(classes_):
        images_names = names_per_class[c]
        cur_num_images_per_class = min(num_images_per_class, len(images_names))
        category_.extend([classes_[j] for i in range(cur_num_images_per_class)])
        images_path_.extend(["{}/{}/{}".format(parent_path, folder_names[j], name_) for name_ in images_names])
        labels_.extend([label + classes_[j] for i in range(cur_num_images_per_class)])
    return category_, images_path_, labels_


def get_images_names(parent_path, classes_, folder_names, num_images_per_class, corrupted_files=[]):
    # to verify pairs of images and sketches with same file name
    names_per_class_ = {}
    for j, c in enumerate(classes_):
        class_path = "{}/{}".format(parent_path, folder_names[j])
        images_names = os.listdir(class_path)
        images_names = [name for name in images_names if "{}-{}\n".format(os.path.splitext(name)[0], str(1)) not in corrupted_files][:num_images_per_class]
        names_per_class_[c] = images_names
    return names_per_class_


def fix_image_scale(im):
    im_np = np.array(im) / 255
    height, width = im_np.shape[0], im_np.shape[1]
    max_len = max(height, width) + 20
    new_background = np.ones((max_len, max_len, 3))
    y, x = max_len // 2 - height // 2, max_len // 2 - width // 2
    new_background[y: y + height, x: x + width] = im_np
    new_background = (new_background / new_background.max() * 255).astype(np.uint8)
    new_im = Image.fromarray(new_background)
    return new_im


def get_mask_u2net(args, pil_im):
    data_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
                ])

    input_im_trans = data_transforms(pil_im).unsqueeze(0).to(args.device)

    model_dir = os.path.join("/datasets/home/vinker/saliency/U-2-Net/saved_models/u2net/u2net.pth")
    net = U2NET(3,1)
    if torch.cuda.is_available() and args.use_gpu:
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    with torch.no_grad():
        d1,d2,d3,d4,d5,d6,d7= net(input_im_trans.detach())
    pred = d1[:,0,:,:]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    predict = pred
    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1
    mask = torch.cat([predict, predict, predict], axis=0).permute(1,2,0)
    mask = mask.cpu().numpy()
    # predict_np = predict.clone().cpu().data.numpy()
    im = Image.fromarray(mask[:,:,0]*255).convert('RGB')
    im.save(f"{args.output_dir}/mask.png")

    im_np = np.array(pil_im)
    im_np = im_np / im_np.max()
    im_np = mask * im_np
    im_np[mask == 0] = 1
    im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    im_final = Image.fromarray(im_final)

    return im_final, predict
