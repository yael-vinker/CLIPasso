import re
import torch
import pydiffvg


parser = argparse.ArgumentParser()
parser.add_argument("--target_file", type=str, help="target image file, located in <target_images>")
args = parser.parse_args()


def read_svg(path_svg, multiply=False):
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(path_svg)
    if multiply:
        canvas_width *= 2
        canvas_height *= 2
        for path in shapes:
            path.points *= 2
            path.stroke_width *= 2
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width, # width
                canvas_height, # height
                2,   # num_samples_x
                2,   # num_samples_y
                0,   # seed
                None,
                *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    return img

abs_path = os.path.abspath(os.getcwd())
target_path = f"{abs_path}/target_images/{args.target_file}"
result_path = f"{abs_path}/output_sketches/{os.path.splitext(args.target_file)[0]}/"
svg_files = os.listdir(result_path)
svg_files = [f for f in svg_files if "best.svg" in f]
svg_output_path = f"{result_path}/{svg_files[0]}"
print(svg_output_path)
# torch
# read_svg(svg_output_path)

