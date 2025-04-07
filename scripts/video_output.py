import cv2

def generate_3d_video(frames, output_path="output_3d_video.mp4", fps=30):
    """Compiles processed frames into a 3D video."""
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        video.write(frame)

    video.release()