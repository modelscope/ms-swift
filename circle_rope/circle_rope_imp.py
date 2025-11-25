import math

import torch


def get_circle_rope_index(w_index, h_index, t_index, config):
    # Load circle rope configurations
    circle_config = config.circle_rope
    move_to_origin = circle_config.get('move_to_origin', False)
    move_to_positive = circle_config.get('move_to_positive', False)
    dff_rate = circle_config.get('dff_rate', False)
    method = circle_config.get('method', 'circle')
    radius = circle_config.get('radius', -1)
    alpha = circle_config.get('alpha', -1)

    # Stack original coordinates
    ori_coords = torch.stack((w_index, h_index, t_index), dim=0)
    if move_to_origin:
        # Move coordinates to origin if specified
        ori_coords = move_to_origin_coords(ori_coords)

    # Determine radius: auto or fixed value
    if 'auto' in str(radius):
        if radius == 'auto':
            radius_scale = 1
        else:
            _, radius_scale = radius.split('-')
        # Calculate radius based on the maximum absolute coordinate value
        radius = ori_coords.max().abs() * float(radius_scale)
    else:
        radius = float(radius)

    # Perform circle projection
    convert_coords = circle_projection(ori_coords, text_vector=[1, 1, 1],
                                       radius=radius,
                                       alpha=alpha,
                                       method=method)

    # Apply differential rate if specified
    if dff_rate:
        no_circle_convert_coords = circle_projection(ori_coords, text_vector=[1, 1, 1],
                                                     radius=-1,
                                                     alpha=-1,
                                                     method='no_circle')
        # Linearly interpolate between circle projection and original coordinates
        convert_coords = (1 - dff_rate) * convert_coords + dff_rate * no_circle_convert_coords

    # Move coordinates to positive axis if specified
    if move_to_positive:
        if move_to_positive == 'auto':
            offset = 0
        else:
            offset = float(move_to_positive)
        convert_coords = move_to_positive_axis(convert_coords, offset=offset)

    # Flatten coordinate dimensions
    w_index = convert_coords[0].flatten()
    h_index = convert_coords[1].flatten()
    t_index = convert_coords[2].flatten()

    # Stack coordinates for language model position IDs
    llm_pos_ids = torch.stack([t_index, h_index, w_index])

    return llm_pos_ids


def move_to_origin_coords(coords):
    """
    Moves the center of the cube to the origin (stacked coordinates version).
    Parameters:
        coords: Tensor of shape (3, depth, height, width)
                Channel order corresponds to [x, y, z] axis coordinates.
    Returns:
        new_coords: Center-aligned coordinate tensor, maintaining the same shape.
    """
    # Calculate the center point for each axis [x_center, y_center, z_center]
    max_vals = torch.amax(coords, dim=(1, 2, 3))  # Get maximum value along spatial dimensions
    min_vals = torch.amin(coords, dim=(1, 2, 3))  # Get minimum value along spatial dimensions
    centers = (max_vals + min_vals) / 2.0
    # Adjust dimensions for broadcast subtraction (3, 1, 1, 1)
    centers = centers.view(-1, 1, 1, 1)
    # Perform translation
    new_coords = coords - centers

    return new_coords


def move_to_positive_axis(coords, offset=0):
    # Find the absolute minimum value across all coordinates
    min_vals = torch.abs(torch.min(coords))
    # Create a tensor of these minimum values for shifting
    centers = torch.tensor([min_vals, min_vals, min_vals]).view(-1, 1, 1, 1)

    # Shift coordinates to be positive and add an optional offset
    new_coords = coords + centers + offset

    return new_coords


def circle_projection(coords, text_vector=[1, 1, 1], radius=1.0, alpha=0.5, method='circle', rotate=True):
    """
    Maps a point cloud to the circumference of a circle on a plane perpendicular to the given text_vector.
    Parameters:
        coords: [3, N] or [3, D, H, W] point cloud or stacked coordinates.
        text_vector: [3] Normal vector of the target plane.
        radius: Target circle radius.
        alpha: Nonlinear coefficient (0-1, controls distribution density).
        method: 'circle' for mapping to circle, 'no_circle' for no mapping.
        rotate: Boolean, whether to rotate the plane.
    """

    # Original non-linear circular mapping
    if method == 'circle':
        coord_circle = map_to_circle(coords, radius, alpha)
    elif method == 'no_circle':
        # Pass through coordinates if no circle mapping is specified
        coord_circle = coords
    else:
        raise ValueError(f"Invalid circle projection method: {method}")

    if rotate:
        # Rotate the plane to be perpendicular to the text_vector
        coord_circle = rotate_plane_perpendicular_to_vector(coord_circle, text_vector)

    return coord_circle


def rotate_plane_perpendicular_to_vector(coord_circle, text_vector):
    data_device = coord_circle.device
    data_dtype = coord_circle.dtype

    # Construct the target plane coordinate system
    text = torch.tensor(text_vector, dtype=data_dtype, device=data_device).float()
    text_norm = torch.norm(text)
    if text_norm < 1e-6:
        raise ValueError("text_vector cannot be zero vector")
    text_unit = text / text_norm # Normalize the text vector

    # Construct an orthogonal basis
    if torch.abs(text_unit[0]) < 1e-6 and torch.abs(text_unit[1]) < 1e-6:
        # Handle the case where the vector is along the z-axis
        u = torch.tensor([1.0, 0.0, 0.0], device=data_device, dtype=data_dtype)
        v = torch.tensor([0.0, 1.0, 0.0], device=data_device, dtype=data_dtype)
    else:
        # Construct the first orthogonal vector u
        u = torch.stack([-text_unit[1], text_unit[0], torch.tensor(0.0, device=data_device)])
        u = u / torch.norm(u) # Normalize u
        # Construct the second orthogonal vector v using cross product
        v = torch.cross(text_unit, u, dim=0)
        v = v / torch.norm(v) # Normalize v

    # Project the circle points onto the new coordinate system
    x_components = coord_circle[0] * u[0] + coord_circle[1] * v[0] # Contribution to new X from original X and Y
    y_components = coord_circle[0] * u[1] + coord_circle[1] * v[1] # Contribution to new Y from original X and Y
    z_components = coord_circle[0] * u[2] + coord_circle[1] * v[2] # Contribution to new Z from original X and Y

    coord_componets = torch.stack([x_components, y_components, z_components])

    return coord_componets


def map_to_circle(tensor, radius=1.0, alpha=0.5):
    """
    Maps points on a plane (z coordinate is 0) to the edge of a circle centered at (0, 0, 0) with the given radius.

    Parameters:
        tensor: A tensor of shape (3, 1, H, W), where the three channels are x, y, z coordinates (here z coordinates are all 0).
        radius: The radius of the mapped circle, default 1.0.
        alpha: Value range [0,1], represents the weight of the normalized original angle; default 0.5.

    Returns:
        A tensor of the same shape as the input tensor, where each point on the plane is mapped to the edge of the circle, and the z coordinate remains unchanged.
    """
    # Extract x, y, z components; here x and y are tensors of shape (H, W)
    x = tensor[0, 0]
    y = tensor[1, 0]
    z = tensor[2, 0] # z coordinates are preserved
    H, W = x.shape

    def get_norm_theta():
        # Method 1: Calculate angle using original coordinates, then linearly normalize to [0, 2π]
        theta_orig = torch.atan2(y, x)
        theta_min = theta_orig.min()
        theta_max = theta_orig.max()
        theta_range = theta_max - theta_min
        if theta_range > 0:
            theta_uniform = (theta_orig - theta_min) / theta_range * (2 * math.pi)
        else:
            # Handle cases with a single point or all points collinear through origin
            theta_uniform = theta_orig
        return theta_uniform

    def get_index_theta():
        # Method 2: Generate uniformly distributed angles based on grid indices
        indices = torch.arange(H * W, dtype=torch.float32, device=tensor.device).reshape(H, W)
        theta_uniform = indices / (H * W) * (2 * math.pi)
        return theta_uniform

    # The larger alpha is, the closer it is to the normalized original angle.
    # When alpha=0, the grid index method is fully used.
    # When alpha=1, the original coordinate calculation angle is fully used.
    theta_norm = get_norm_theta()
    theta_index = get_index_theta()
    # Combine the two methods for calculating theta based on alpha
    theta_uniform = alpha * theta_norm + (1 - alpha) * theta_index

    # Generate mapped x, y coordinates based on the calculated uniform angle
    new_x = radius * torch.cos(theta_uniform)
    new_y = radius * torch.sin(theta_uniform)

    # Combine the three channels and maintain the shape as (3, 1, H, W)
    new_tensor = torch.stack([new_x, new_y, z], dim=0).unsqueeze(1)

    return new_tensor
