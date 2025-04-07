import numpy as np
import cv2

def create_stereo_frames(image, depth_map, shift=10):
    """Generate left & right perspective frames based on depth."""
    h, w = image.shape[:2]

    # Ensure depth map matches image size
    depth_map = cv2.resize(depth_map, (w, h))

    left_frame = np.zeros_like(image)
    right_frame = np.zeros_like(image)

    max_depth = np.max(depth_map) + 1e-6  # prevent divide-by-zero

    for y in range(h):
        for x in range(w):
            depth_value = depth_map[y, x]
            offset = int(shift * depth_value / max_depth)

            new_x_left = max(0, x - offset)
            new_x_right = min(w - 1, x + offset)

            left_frame[y, new_x_left] = image[y, x]
            right_frame[y, new_x_right] = image[y, x]

    return left_frame, right_frame


# import numpy as np
# import cv2

# def create_stereo_frames(image, depth_map, shift=10):
#     """Generate left & right perspective frames based on depth."""
#     h, w = image.shape[:2]
    
#     left_frame = np.zeros_like(image)
#     right_frame = np.zeros_like(image)
    
#     for y in range(h):
#         for x in range(w):
#             depth_value = depth_map[y, x]
#             offset = int(shift * depth_value / np.max(depth_map))
            
#             new_x_left = max(0, x - offset)
#             new_x_right = min(w - 1, x + offset)
            
#             left_frame[y, new_x_left] = image[y, x]
#             right_frame[y, new_x_right] = image[y, x]

#     return left_frame, right_frame