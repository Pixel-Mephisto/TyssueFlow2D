import imageio
import numpy as np

class TissueExporter:
    def __init__(self) -> None:
        self.video_writer = None
        self.is_recording = False

    def start_recording(self, filepath: str, fps: int) -> bool:
        try:
            self.video_writer = imageio.get_writer(filepath, fps=fps, macro_block_size=2)
            self.is_recording = True
            return True
        except Exception:
            return False

    def capture_frame(self, fig) -> None:
        if not self.is_recording or self.video_writer is None: return
        width, height = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape((height, width, 4))[:, :, :3]
        
        if img.shape[0] % 2 != 0: img = img[:-1, :, :]
        if img.shape[1] % 2 != 0: img = img[:, :-1, :]
        self.video_writer.append_data(img)

    def close_stream(self) -> None:
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        self.is_recording = False