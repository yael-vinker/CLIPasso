import random
import CLIP_.clip as clip
import numpy as np
import pydiffvg
import sketch_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms


class Painter(torch.nn.Module):
    def __init__(self, args,
                num_strokes=4,
                num_segments=4,
                imsize=224,
                device=None,
                target_im=None,
                mask=None):
        super(Painter, self).__init__()

        self.args = args
        self.num_paths = num_strokes
        self.num_segments = num_segments
        self.width = args.width
        self.control_points_per_seg = args.control_points_per_seg
        self.opacity_optim = args.force_sparse
        self.num_stages = args.num_stages
        self.add_random_noise = "noise" in args.augemntations
        self.noise_thresh = args.noise_thresh
        self.softmax_temp = args.softmax_temp

        self.shapes = []
        self.shape_groups = []
        self.device = device
        self.canvas_width, self.canvas_height = imsize, imsize
        self.points_vars = []
        self.color_vars = []
        self.color_vars_threshold = args.color_vars_threshold

        self.path_svg = args.path_svg
        self.strokes_per_stage = self.num_paths
        self.optimize_flag = []

        # attention related for strokes initialisation
        self.attention_init = args.attention_init
        self.target_path = args.target
        self.saliency_model = args.saliency_model
        self.xdog_intersec = args.xdog_intersec
        self.mask_object = args.mask_object_attention
        
        self.text_target = args.text_target # for clip gradients
        self.saliency_clip_model = args.saliency_clip_model
        self.define_attention_input(target_im)
        self.mask = mask
        self.attention_map = self.set_attention_map() if self.attention_init else None
        
        self.thresh = self.set_attention_threshold_map() if self.attention_init else None
        self.strokes_counter = 0 # counts the number of calls to "get_path"        
        self.epoch = 0
        self.final_epoch = args.num_iter - 1
        

    def init_image(self, stage=0):
        if stage > 0:
            # if multi stages training than add new strokes on existing ones
            # don't optimize on previous strokes
            self.optimize_flag = [False for i in range(len(self.shapes))]
            for i in range(self.strokes_per_stage):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)
                self.optimize_flag.append(True)

        else:
            num_paths_exists = 0
            if self.path_svg != "none":
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = utils.load_svg(self.path_svg)
                # if you want to add more strokes to existing ones and optimize on all of them
                num_paths_exists = len(self.shapes)

            for i in range(num_paths_exists, self.num_paths):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)        
            self.optimize_flag = [True for i in range(len(self.shapes))]
        
        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
        return img
        # utils.imwrite(img.cpu(), '{}/init.png'.format(args.output_dir), gamma=args.gamma, use_wandb=args.use_wandb, wandb_name="init")

    def get_image(self):
        img = self.render_warp()
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
        return img

    def get_path(self):
        points = []
        self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
        p0 = self.inds_normalised[self.strokes_counter] if self.attention_init else (random.random(), random.random())
        points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        
        path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(self.width),
                                is_closed = False)
        self.strokes_counter += 1
        return path

    def render_warp(self):
        if self.opacity_optim:
            for group in self.shape_groups:
                group.stroke_color.data[:3].clamp_(0., 0.) # to force black stroke
                group.stroke_color.data[-1].clamp_(0., 1.) # opacity
                # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
        _render = pydiffvg.RenderFunction.apply
        # uncomment if you want to add random noise
        if self.add_random_noise:
            if random.random() > self.noise_thresh:
                eps = 0.01 * min(self.canvas_width, self.canvas_height)
                for path in self.shapes:
                    path.points.data.add_(eps * torch.randn_like(path.points))
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        img = _render(self.canvas_width, # width
                    self.canvas_height, # height
                    2,   # num_samples_x
                    2,   # num_samples_y
                    0,   # seed
                    None,
                    *scene_args)
        return img
    
    def parameters(self):
        self.points_vars = []
        # storkes' location optimization
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.points.requires_grad = True
                self.points_vars.append(path.points)
        return self.points_vars
    
    def get_points_parans(self):
        return self.points_vars
    
    def set_color_parameters(self):
        # for storkes' color optimization (opacity)
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            if self.optimize_flag[i]:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)
        return self.color_vars

    def get_color_parameters(self):
        return self.color_vars
        
    def save_svg(self, output_dir, name):
        pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)


    def dino_attn(self):
        patch_size=8 # dino hyperparameter
        threshold=0.6

        # for dino model
        mean_imagenet = torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None].to(self.device)
        std_imagenet = torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None].to(self.device)
        totens = transforms.Compose([
            transforms.Resize((self.canvas_height, self.canvas_width)),
            transforms.ToTensor()
            ])

        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').eval().to(self.device)
        
        self.main_im = Image.open(self.target_path).convert("RGB")
        main_im_tensor = totens(self.main_im).to(self.device)
        img = (main_im_tensor.unsqueeze(0) - mean_imagenet) / std_imagenet
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size
        
        with torch.no_grad():
            attn = dino_model.get_last_selfattention(img).detach().cpu()[0]

        nh = attn.shape[0]
        attn = attn[:,0,1:].reshape(nh,-1)
        val, idx = torch.sort(attn)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
        
        attn = attn.reshape(nh, w_featmap, h_featmap).float()
        attn = nn.functional.interpolate(attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
        
        return attn


    def define_attention_input(self, target_im):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([
                    preprocess.transforms[-1],
                ])
        self.image_input_attn_clip = data_transforms(target_im).to(self.device)
        

    def clip_attn(self):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        text_input = clip.tokenize([self.text_target]).to(self.device)

        if "RN" in self.saliency_clip_model:
            saliency_layer = "layer4"
            attn_map = gradCAM(
                model.visual,
                self.image_input_attn_clip,
                model.encode_text(text_input).float(),
                getattr(model.visual, saliency_layer)
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        else:
            # attn_map = interpret(self.image_input_attn_clip, text_input, model, device=self.device, index=0).astype(np.float32)
            attn_map = interpret(self.image_input_attn_clip, text_input, model, device=self.device)
            
        del model
        return attn_map

    def set_attention_map(self):
        assert self.saliency_model in ["dino", "clip"]
        if self.saliency_model == "dino":
            return self.dino_attn()
        elif self.saliency_model == "clip":
            return self.clip_attn()
        

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum() 

    def set_inds_clip(self):
        attn_map = (self.attention_map - self.attention_map.min()) / (self.attention_map.max() - self.attention_map.min())
        if self.xdog_intersec:
            xdog = XDoG_()
            im_xdog = xdog(self.image_input_attn_clip[0].permute(1,2,0).cpu().numpy(), k=10)
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map
            
        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)
        
        k = self.num_stages * self.num_paths
        self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
        self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
    
        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] =  self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] =  self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return attn_map_soft



    def set_inds_dino(self):
        k = max(3, (self.num_stages * self.num_paths) // 6 + 1) # sample top 3 three points from each attention head
        num_heads = self.attention_map.shape[0]
        self.inds = np.zeros((k * num_heads, 2))
        # "thresh" is used for visualisaiton purposes only
        thresh = torch.zeros(num_heads + 1, self.attention_map.shape[1], self.attention_map.shape[2])
        softmax = nn.Softmax(dim=1)
        for i in range(num_heads):
            # replace "self.attention_map[i]" with "self.attention_map" to get the highest values among
            # all heads. 
            topk, indices = np.unique(self.attention_map[i].numpy(), return_index=True)
            topk = topk[::-1][:k]
            cur_attn_map = self.attention_map[i].numpy()
            # prob function for uniform sampling
            prob = cur_attn_map.flatten()
            prob[prob > topk[-1]] = 1
            prob[prob <= topk[-1]] = 0
            prob = prob / prob.sum()
            thresh[i] = torch.Tensor(prob.reshape(cur_attn_map.shape))

            # choose k pixels from each head            
            inds = np.random.choice(range(cur_attn_map.flatten().shape[0]), size=k, replace=False, p=prob)
            inds = np.unravel_index(inds, cur_attn_map.shape)
            self.inds[i * k: i * k + k, 0] = inds[0]
            self.inds[i * k: i * k + k, 1] = inds[1]
        
        # for visualisaiton
        sum_attn = self.attention_map.sum(0).numpy()
        mask = np.zeros(sum_attn.shape)
        mask[thresh[:-1].sum(0) > 0] = 1
        sum_attn = sum_attn * mask
        sum_attn = sum_attn / sum_attn.sum()
        thresh[-1] = torch.Tensor(sum_attn)

        # sample num_paths from the chosen pixels.
        prob_sum = sum_attn[self.inds[:,0].astype(np.int), self.inds[:,1].astype(np.int)]
        prob_sum = prob_sum / prob_sum.sum()
        new_inds = []
        for i in range(self.num_stages):
            new_inds.extend(np.random.choice(range(self.inds.shape[0]), size=self.num_paths, replace=False, p=prob_sum))
        self.inds = self.inds[new_inds]
        print("self.inds",self.inds.shape)
    
        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] =  self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] =  self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return thresh

    def set_attention_threshold_map(self):
        assert self.saliency_model in ["dino", "clip"]
        if self.saliency_model == "dino":
            return self.set_inds_dino()
        elif self.saliency_model == "clip":
            return self.set_inds_clip()
        

    def get_attn(self):
        return self.attention_map
    
    def get_thresh(self):
        return self.thresh

    def get_inds(self):
        return self.inds
    
    def get_mask(self):
        return self.mask

    def set_random_noise(self, epoch):
        if epoch % self.args.save_interval == 0:
            self.add_random_noise = False
        else:
            self.add_random_noise = "noise" in self.args.augemntations

class PainterOptimizer:
    def __init__(self, args, renderer):
        self.renderer = renderer
        self.points_lr = args.lr
        self.color_lr = args.color_lr
        self.args = args
        self.optim_color = args.force_sparse

    def init_optimizers(self):
        self.points_optim = torch.optim.Adam(self.renderer.parameters(), lr=self.points_lr)
        if self.optim_color:
            self.color_optim = torch.optim.Adam(self.renderer.set_color_parameters(), lr=self.color_lr)

    def update_lr(self, counter):
        new_lr = utils.get_epoch_lr(counter, self.args)
        for param_group in self.points_optim.param_groups:
            param_group["lr"] = new_lr
    
    def zero_grad_(self):
        self.points_optim.zero_grad()
        if self.optim_color:
            self.color_optim.zero_grad()
    
    def step_(self):
        self.points_optim.step()
        if self.optim_color:
            self.color_optim.step()
    
    def get_lr(self):
        return self.points_optim.param_groups[0]['lr']


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad




def interpret(image, texts, model, device):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images)
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = [] # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_probs.detach() #attn_probs shape is 12, 50, 50
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1) # mean of the 12 something
        cams.append(cam)  
        R = R + torch.bmm(cam, R)
              
    cams_avg = torch.cat(cams) # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam    


class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma=0.98
        self.phi=200
        self.eps=-0.1
        self.sigma=0.8
        self.binarize=True
        
    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0  + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')
        return imdiff
