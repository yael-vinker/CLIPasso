"""Collection of generative models."""

import torch as th
import ttools
import torch.nn as nn
import os
import sys
import inspect
from torchvision import datasets, models, transforms
import torch
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)

print(parent_dir)
sys.path.insert(0, parent_dir)

from network_training.models import rendering

LOG = ttools.get_logger(__name__)


class BezierVectorGenerator(th.nn.Module):
    NUM_SEGMENTS = 2
    def __init__(self, 
                num_strokes=4,
                num_segments=4,
                 stroke_width=None,
                 color_output=False,
                 imsize=224,
                 fix_radius=False,
                 optim_radius=False,
                 device=None):
        super(BezierVectorGenerator, self).__init__()


        self.num_strokes = num_strokes
        self.num_segments = num_segments
        self.radius = 0.5
        self.fix_radius = fix_radius
        self.optim_radius = optim_radius
        self.device = device

         # Torchvision's normalization: <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>
        self.register_buffer("shift", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("scale", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        

        if stroke_width is None:
            self.stroke_width = (0.5, 3.0)
            LOG.warning("Setting default stroke with %s", self.stroke_width)
        else:
            self.stroke_width = stroke_width
        self.imsize = imsize

        self.resnet18 = models.resnet18(pretrained=False)
        
        self.out_dim = 512
        # remove the last fully connected layer
        self.model = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))
        # 4 points bezier with n_segments -> 3*n_segments + 1 points
        self.point_predictor = nn.Sequential(
            th.nn.Linear(self.out_dim, 2 * self.num_strokes * (self.num_segments * 3 + 1)),
            th.nn.Tanh()  # bound spatial extent
        )

        if self.optim_radius:
            # predict radius for each point
            self.radius_predictor = nn.Sequential(
                th.nn.Linear(self.out_dim, self.num_strokes * (self.num_segments * 3 + 1)),
                th.nn.Sigmoid()
            )


    def set_fix_radius(self, value):
        self.fix_radius = value


    def forward(self, inputs):
        bs = inputs.shape[0]
        inputs = (inputs - self.shift) / self.scale
        feats = self.model(inputs)
        feats = feats.view(-1, self.out_dim)
        all_points = self.point_predictor(feats)

        # currently, we don't optimize these parameters
        all_widths = th.ones((bs, self.num_strokes)) * self.stroke_width[1]
        all_alphas = th.ones((bs, self.num_strokes))
        all_colors = th.zeros((bs, 3*self.num_strokes))
        all_colors = all_colors.view(bs, self.num_strokes, 3)

        all_points = all_points.view(bs, self.num_strokes, self.num_segments * 3 + 1, 2)

        if self.fix_radius:
            radius_pred = torch.ones((1, self.num_strokes * (self.num_segments * 3 + 1))).to(self.device) * self.radius
            if self.optim_radius:
                radius_pred = self.radius_predictor(feats)
            all_points_copy = all_points.clone()
            for j in range(self.num_strokes):
                p0 = all_points[:, j, 0]
                for i in range(1, all_points.shape[2], 3):
                    p1 = all_points[:, j, i].mul(radius_pred[:, j + i - 1]).add(p0)
                    p2 = all_points[:, j, i + 1].mul(radius_pred[:, j + i]).add(p1)
                    p3 = all_points[:, j, i + 2].mul(radius_pred[:, j + i + 1]).add(p2)
                    all_points_copy[:, j, i] = p1
                    all_points_copy[:, j, i + 1] = p2
                    all_points_copy[:, j, i + 2] = p3
                    p0 = p3
           
            output, scenes = rendering.bezier_render(all_points_copy, all_widths, all_alphas,        
                                         colors=all_colors,
                                         canvas_size=self.imsize)
            return output
        
        output, scenes = rendering.bezier_render(all_points, all_widths, all_alphas,
                                         colors=all_colors,
                                         canvas_size=self.imsize)


        return output
