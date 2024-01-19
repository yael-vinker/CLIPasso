import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function


def create_tensor_from_indices(indices, image_size_x, image_size_y):
    # Initialize the tensor with zeros
    tensor = torch.zeros((image_size_x, image_size_y))
    # Iterate over the indices and set the specified locations to 1
    for x, y, _ in indices:
        tensor[x, y] = 1
    return tensor.requires_grad_(True)


def floor_divide_positions(positions, desired_size):
    # Apply floor division to each element in each sub-list
    max_pos = np.array(positions).max()
    normalized = [[int((x /max_pos) * desired_size) for x in position] for position in positions]
    return normalized


def gumbel_softmax_static(logits: Tensor, args) -> Tensor:
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + args.gumbel_noise_scale * gumbel_noise
    y.requires_grad_(True)
    output = F.softmax(y / args.gumbel_temperature, dim=-1)
    return output


def flattened_topk(x: Tensor, args) -> Tensor:
    x_flattened = torch.flatten(x)
    _, indices = x_flattened.topk(args.num_dots)
    zeros = torch.zeros_like(x_flattened)
    zeros = zeros.scatter(0, indices, 1)
    return zeros.reshape(x.shape)


class PhosphenePlacementNetwork(nn.Module):
    def __init__(self, input_dim: np.array, output_dim: np.array) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = self.build_encoder()

    def build_encoder(self) -> nn.Module:
        return nn.Sequential(...)

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(self.encoder(x))


class GumbelSoftmaxTopK(Function):
    """A class that allows the differentiation of the discrete forward signal in the backward pass."""
    @staticmethod
    def forward(ctx, logits, args):
        # store for use in backward
        ctx.args = args
        ctx.save_for_backward(logits)

        soft_placement_selection = gumbel_softmax_static(logits, args)
        selected_indices = flattened_topk(soft_placement_selection, args)
        return selected_indices

    @staticmethod
    def backward(ctx, grad_output):
        logits, = ctx.saved_tensors
        args = ctx.args

        soft_grad = gumbel_softmax_static(logits, args)
        return soft_grad * grad_output, None  # (gradient w.r.t logits, gradient w.r.t. args)


class PhosphenePlacementAlgorithm(nn.Module):
    """i.e., the Bremen algorithm. TODO: descriptions, blah blah"""
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, logits: Tensor) -> Tensor:
        # get the backwards-differentiable top-K selection of logits for placement in the image
        hard_selection = GumbelSoftmaxTopK.apply(logits, self.args)
        output_image = self.render_output_image(hard_selection)
        return output_image

    def render_output_image(self, phosphene_locations: Tensor):
        phosphene = self.generate_phosphene()  # TODO: better here would be to have a function for selecting the type
                                               #  of phosphene depending on an argument. otherwise, just delete this
                                               #  and call self.generate_phosphene() in self.draw_elements() directly
        image = self.draw_differentiable_elements(phosphene, phosphene_locations)
        return image

    def generate_phosphene(self) -> Tensor:
        """Generates a phosphene with a given radius on a patch."""
        # Define grid of phosphene
        half_patch = int(self.args.patch_size // 2)
        x = torch.arange(start=-half_patch, end=half_patch + 1)
        x_grid, y_grid = torch.meshgrid([x, x])

        # Generate phosphene on the grid. Luminance values normalized between 0 and 1
        phosphene = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * self.args.phosphene_radius ** 2))
        phosphene /= phosphene.sum()
        return phosphene.unsqueeze(0)

    def draw_differentiable_elements(self, elements: torch.Tensor, elem_xy_locations: torch.Tensor) -> torch.Tensor:
        canvas = self.args.canvas.clone()
        output_img = torch.zeros_like(canvas)

        # Compute the scaling factor
        scale_x = canvas.shape[0] / elem_xy_locations.shape[0]
        scale_y = canvas.shape[1] / elem_xy_locations.shape[1]

        for x in range(elem_xy_locations.shape[0]):
            for y in range(elem_xy_locations.shape[1]):
                if elem_xy_locations[y, x] != 0:
                    # Calculate the scaled positions
                    x_pos, y_pos = int(x * scale_x), int(y * scale_y)
                    # pad the element to the canvas size
                    padding = self.calculate_padding(x_pos, y_pos, elements, canvas)
                    padded_element = F.pad(elements, padding)
                    # blend the element locations to the output
                    blending_weight = torch.sigmoid(elem_xy_locations[x, y])
                    # Apply weighted blending
                    output_img += blending_weight * padded_element[0]

        return output_img

    def gaussian_kernel(self, size, sigma):
        """Creates a 2D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2

        g = coords ** 2
        g = (-g / (2 * sigma ** 2)).exp()
        g = g / g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)

    def apply_gaussian_smoothing(self, elements, x_pos, y_pos, canvas_shape, kernel_size, sigma):
        """Applies Gaussian smoothing to the elements at a given position."""
        kernel = self.gaussian_kernel(kernel_size, sigma)
        pad_x = (kernel_size - 1) // 2
        pad_y = (kernel_size - 1) // 2

        # Adjust x_pos, y_pos to the top-left corner of the kernel
        x_start = int(max(0, x_pos - pad_x))
        y_start = int(max(0, y_pos - pad_y))

        # Crop the kernel if it goes beyond the canvas
        kernel = kernel[:min(canvas_shape[1] - x_start, kernel_size), :min(canvas_shape[0] - y_start, kernel_size)]

        # Apply the kernel to the elements
        smoothed_elements = F.conv2d(elements.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=(pad_x, pad_y))
        return smoothed_elements.squeeze(0)

    @staticmethod
    def calculate_padding(x_pos: int, y_pos: int, smoothed_element: Tensor, canvas: Tensor) -> Tuple[int, ...]:
        pad_top = max(y_pos - smoothed_element.shape[2] // 2, 0)
        pad_left = max(x_pos - smoothed_element.shape[1] // 2, 0)
        pad_bottom = max(canvas.shape[1] - pad_top - smoothed_element.shape[2], 0)
        pad_right = max(canvas.shape[0] - pad_left - smoothed_element.shape[1], 0)
        return (pad_left, pad_right, pad_top, pad_bottom)


def parse_arguments(num_phosphenes):
    parser = argparse.ArgumentParser(description='PhosphenePlacementAlgorithm arguments')

    parser.add_argument('--num_dots', type=int, help='How many dots to place in the image.', default=num_phosphenes)
    parser.add_argument('--gumbel_temperature', type=Tensor, help='Gumbel softmax temperature parameter.',
                        default=Tensor([0.1]))
    parser.add_argument('--gumbel_noise_scale', type=float,
                        help='How much randomness to add in the phosphene placement. More = better gradient flow, less'
                             ' = more accurate placement. Warning: setting to 0.0 effectively kills the gradient.',
                        default=0.02)
    parser.add_argument('--patch_size', type=int, help='Size of each of the patches of phosphenes.', default=10)
    parser.add_argument('--phosphene_radius', type=int, help='Radius of the phosphene.', default=1.5)
    parser.add_argument('--gaussian_kernel_sigma', type=float,
                        help='Gaussian Kernel size used for convolving the phosphene on the output image.',
                        default=0.5)
    parser.add_argument('--canvas', type=Tensor,
                        help='Output image size. TODO: need to match it to downstream model input size.',
                        default=torch.zeros([220, 220]))

    parser = parser.parse_args()
    assert parser.canvas.shape[0] % parser.patch_size == 0
    assert parser.canvas.shape[1] % parser.patch_size == 0

    return parser


def display_image_output(model, input):
    """
    Generate and display an output image from the given model and input.

    :param model: Instance of PhosphenePlacementAlgorithm
    :param input: Tensor input for the model
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        output_image = model(input)

    # Display the output image
    plt.imshow(output_image.detach().cpu().numpy(), cmap='gray')
    plt.title("Output Image")
    plt.show()


def display_differentiability(model, input):
    """
    Test if the forward function of the model is differentiable.

    :param model: Instance of PhosphenePlacementAlgorithm
    :param input: Tensor input for the model
    :return: Boolean indicating if the forward pass is differentiable
    """
    try:
        torch.autograd.set_detect_anomaly(True)
        output = model(input.requires_grad_(True))
        output.sum().backward()  # Try to compute gradients
        return True
    except RuntimeError as e:
       print(f"Error during differentiation: {e}")
       return False


if __name__ == '__main__':
    phosphene_positions = np.load('test_phosphene_arrangement.npz')['data']
    args = parse_arguments(num_phosphenes=len(phosphene_positions))

    model = PhosphenePlacementAlgorithm(args)
    target_logit_size = args.canvas.shape[0] // 10
    positions = floor_divide_positions(phosphene_positions, target_logit_size - 1)  # -1 for this case since counting from 0
    input_tensor = create_tensor_from_indices(positions, target_logit_size, target_logit_size)

    # Test and display the output image
    display_image_output(model, input_tensor)

    # Test differentiability
    is_differentiable = display_differentiability(model, input_tensor)
    print(f"Is the forward function differentiable? {is_differentiable}")
