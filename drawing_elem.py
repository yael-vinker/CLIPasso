# %%
import matplotlib.pyplot as plt
plt.plot()
import torch
import math


# %% Define functions

def prep_subsegments(
        patch_size: float,
        phi: float,
        subsegment_length: float,
        subsegment_width: float,
    ) -> torch.Tensor:
        
        """This function prepares single subsegments.
        Subsegments will then be combined to build curved segments"""

        pi = torch.tensor(
            math.pi
        )

        # extension of patch beyond origin
        n: int = int(patch_size // 2)

        # grid for the patch
        temp_grid: torch.Tensor = torch.arange(
            start=-n,
            end=n + 1,
        )

        x, y = torch.meshgrid(temp_grid, temp_grid, indexing="ij")

        r: torch.Tensor = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=2)

        # target orientation of basis segment
        phi90: torch.Tensor = phi + pi / 2

        # vector pointing to the ending point of subsegment (direction)
        #
        # when result is displayed with plt.imshow(segment),
        # phi=0 points to the right, and increasing phi rotates
        # the segment counterclockwise
        #
        e: torch.Tensor = torch.tensor(
            [torch.cos(phi90), torch.sin(phi90)]
        )

        # tangential vectors
        e_tang: torch.Tensor = e.flip(dims=[0]) * torch.tensor(
            [-1, 1]
        )

        # compute distances to segment: parallel/tangential
        d = torch.maximum(
            torch.zeros(
                (r.shape[0], r.shape[1])
            ),
            torch.abs(
                (r * e.unsqueeze(0).unsqueeze(0)).sum(dim=-1) - subsegment_length / 2
            )
            - subsegment_length / 2,
        )

        d_tang = (r * e_tang.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        # compute minimum distance to any of the two pointers
        dr = torch.sqrt(d**2 + d_tang**2)

        subsegment = torch.exp(-(dr**2) / 2 / subsegment_width**2)
        subsegment = subsegment / subsegment.sum()

        return subsegment

def segments_generator(
    n_dir: int = 8, # number of segment orientation
    n_open: int = 4, # number of opening angles between 2 subsegments
    n_filter: int = 4,
    patch_size: int = 41,
    segment_width: float = 2.5,
    segment_length: float = 15.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    """This functions generates curved segments. Curved segments are composed of 2 subsegments:
    1) a basis subsegment with a given orientation
    2) a second subsegment forming an angle with the basis segment
    Both subsegments are prepared individually with the function 'prep_subsegments'"""

    pi = torch.tensor(
            math.pi
        )
    
    # for n_dir directions, there are n_open_max opening angles possible...
    assert n_dir % 2 == 0, "n_dir must be multiple of 2"
    n_open_max: int = n_dir // 2

    # ...but this number can be reduced by integer multiples:
    assert (
        n_open_max % n_open == 0
    ), "n_open_max = n_dir // 2 must be multiple of n_open"
    mul_open: int = n_open_max // n_open

    # compute single segments
    segments: torch.Tensor = torch.zeros(
        (n_dir, patch_size, patch_size)
    )
    dirs: torch.Tensor = (
        2
        * pi
        * torch.arange(
            start=0, end=n_dir)
        / n_dir
    )

    for i_dir in range(n_dir):
        segments[i_dir] = prep_subsegments(
            patch_size=patch_size,
            phi=float(dirs[i_dir]),
            subsegment_length=segment_length,
            subsegment_width=segment_width,
        )

    # compute patches from segments
    clocks = torch.zeros(
        (patch_size, patch_size, n_open, n_dir, )
    )

    for i_dir in range(n_dir):
        for i_open in range(n_open):

            seg1 = segments[i_dir] # basis segment
            seg2 = segments[(i_dir + (i_open + 1) * mul_open) % n_dir] # second segment

            clocks[:,:,i_open, i_dir] = torch.where(seg1 > seg2, seg1, seg2)

    clocks = clocks.reshape((patch_size, patch_size, n_open * n_dir))
    clocks = clocks / clocks.sum(axis=(0, 1), keepdims=True)


    # clocks are already normalized!
    return clocks

def phosphene_generator(patch_size, radius_phosphene):

    """Generates a phosphene with a given radius on a patch"""

    # Define grid of phosphene
    half_patch = int(patch_size // 2)
    x = torch.arange(start= -half_patch, end= half_patch+1)
    x_grid, y_grid = torch.meshgrid(x, x, indexing="ij")

    # Generate phosphene on the grid. Luminance values normalized between 0 and 1
    phosphene = torch.exp(-(x_grid**2 + y_grid**2) / (2 * radius_phosphene**2))
    phosphene /= phosphene.sum()

    # Add ID of element as additional dimension.
    # As there is only one type of phosphene, the ID is set to one
    phosphene.unsqueeze_(-1)

    return phosphene

def draw_elements(elements, canvas, elem_xy_locations):
        
    """
    Draws elements on specified xy pixel coordinates of a canvas.
    
    "elem_xy_locations" is a 2D-Tensor with shape n x 3.
        n rows: number of elements
        3 cols: x_coor, ycoor, element ID
    """

    canvas_copy = canvas.clone().detach()
    output_img = torch.zeros_like(canvas_copy)

    # If there is more than 1 element ID in elem_xy_locations[:,2], 
    # there are different type of elements to put at the provided xy pixel coordinates
    multiple_elems = bool(len(torch.unique(elem_xy_locations[:, 2])) > 1)

    for element_i in range(0, elem_xy_locations.shape[0]):
        
        # Define where to draw the element on the canvas
        x_start_pos = int(elem_xy_locations[element_i, 0] - elements.shape[0]//2)
        x_stop_pos = int(elem_xy_locations[element_i, 0] + elements.shape[0]//2+1)
        y_start_pos = int(elem_xy_locations[element_i, 1] - elements.shape[1]//2)
        y_stop_pos = int(elem_xy_locations[element_i, 1] + elements.shape[1]//2+1)

        # Identify which element should be drawn at the current position
        if multiple_elems:
            element_idx = int(elem_xy_locations[element_i, 2])
        else:
            element_idx = 0

        # Draw elements on the canvas
        output_img[x_start_pos:x_stop_pos, y_start_pos:y_stop_pos] += elements[:,:,element_idx]

    return output_img


# %% Define parameters & arguments
# elem_patch_size = int(
#     1 + (int(args.size_elem * args.resolution) // 2) * 2
# )
elem_patch_size = int(
    1 + (int(1.0 * 80) // 2) * 2
)

blank_canvas = torch.zeros([978, 978])

position_selection_phos = torch.Tensor([[466, 155, 1],
         [840, 535, 1],
         [571, 136, 1],
         [511, 109, 1],
         [224, 232, 1],
         [803, 430, 1],
         [766, 393, 1],
         [348, 161, 1],
         [227, 627, 1],
         [140, 435, 1],
         [271, 190, 1],
         [459, 717, 1],
         [723, 359, 1],
         [186, 276, 1],
         [373, 554, 1],
         [669, 320, 1],
         [643, 830, 1],
         [607, 283, 1],
         [433, 486, 1],
         [193, 587, 1],
         [400, 159, 1],
         [567, 223, 1],
         [758, 758, 1],
         [831, 590, 1],
         [558, 772, 1],
         [425, 580, 1],
         [164, 329, 1]])

position_selection_seg = torch.Tensor([[466, 155, 25],
         [ 840, 535, 24],
         [571, 136, 27],
         [511, 109, 26],
         [224, 232, 25],
         [803, 430, 27],
         [766, 393, 27],
         [348, 161, 26],
         [227, 627, 27],
         [140, 435, 24],
         [271, 190, 25],
         [459, 717, 26],
         [723, 359, 27],
         [186, 276, 25],
         [373, 554, 6],
         [669, 320, 27],
         [643, 830, 27],
         [607, 283, 27],
         [433, 486, 24],
         [193, 587, 27],
         [400, 159, 26],
         [567, 223, 27],
         [758, 758, 24],
         [831, 590, 24],
         [558, 772, 27],
         [425, 580, 19]])


# %% Generate phosphene images

elem_param = {
    "phos_radius": 0.18,  # half-width of Gaussian. Unit: scaling factor of the elem_patch_size
}
phosphenes = phosphene_generator(  # later change name of function for "phos_generator"
    patch_size= elem_patch_size,
    radius_phosphene= elem_param["phos_radius"] * elem_patch_size
)
phos_img = draw_elements(
    elements= phosphenes,
    canvas= blank_canvas,
    elem_xy_locations= position_selection_phos # xy pixel coordinates on where to put elements
)
plt.imshow(phos_img)


# %% Generate Image Segments
elem_param = {
    "n_dir_subsegment": 8,  # number of directions for subsegments
    "n_angle_subsegment": 4,  # number of angles between two subsegments
    "width_subsegment": 0.07,  # relative width and size of tip extension of subsegments
    "len_subsegment": 0.18,  # relative length of subsegments
}

segments = segments_generator(
    patch_size= elem_patch_size,
    n_dir= elem_param["n_dir_subsegment"],
    segment_width= elem_param["width_subsegment"] * elem_patch_size,
    segment_length= elem_param["len_subsegment"] * elem_patch_size,
)
seg_img = draw_elements(
    elements= segments,
    canvas= blank_canvas,
    elem_xy_locations= position_selection_seg
)

plt.imshow(seg_img)
# %%
