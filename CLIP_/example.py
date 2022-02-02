import torch
import clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def interpret(image, text, model, device, index=None):
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    if index is None:
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    for blk in image_attn_blocks:
        grad = blk.attn_grad
        cam = blk.attn_probs
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        R += torch.matmul(cam, R)
    R[0, 0] = 0
    image_relevance = R[0, 1:]

    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    plt.imshow(vis)
    plt.show()

    print("Label probs:", probs)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    image = preprocess(Image.open("catdog.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a dog", "a cat"]).to(device)
    interpret(model=model, image=image, text=text, device=device, index=0)
    interpret(model=model, image=image, text=text, device=device, index=1)

    image = preprocess(Image.open("el1.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["an elephant", "a zebra"]).to(device)
    interpret(model=model, image=image, text=text, device=device, index=0)
    interpret(model=model, image=image, text=text, device=device, index=1)

    image = preprocess(Image.open("el2.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["an elephant", "a zebra"]).to(device)
    interpret(model=model, image=image, text=text, device=device, index=0)
    interpret(model=model, image=image, text=text, device=device, index=1)

    image = preprocess(Image.open("el3.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["an elephant", "a zebra"]).to(device)
    interpret(model=model, image=image, text=text, device=device, index=0)
    interpret(model=model, image=image, text=text, device=device, index=1)

    image = preprocess(Image.open("el4.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["an elephant", "a zebra"]).to(device)
    interpret(model=model, image=image, text=text, device=device, index=0)
    interpret(model=model, image=image, text=text, device=device, index=1)

    image = preprocess(Image.open("dogbird.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a basset hound", "a parrot"]).to(device)
    interpret(model=model, image=image, text=text, device=device, index=0)
    interpret(model=model, image=image, text=text, device=device, index=1)


if __name__ == "__main__":
    main()

