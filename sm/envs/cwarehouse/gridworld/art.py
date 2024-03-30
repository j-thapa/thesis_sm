from typing import List, Dict, Tuple
import numpy as np


def text_to_art(text: List[str]) -> np.ndarray:
    """Convert ascii art to 2D ordinal numpy array"""
    art = np.vstack(
        [np.frombuffer(line.encode("ascii"), dtype=np.uint8) for line in text]
    )
    return art


def art_mask(art: np.ndarray, transparent: str = ".") -> np.ndarray:
    """Returns the valid pixel mask for the given art."""
    return art != ord(transparent)


def art_pixels(art: np.ndarray, colors: dict) -> np.ndarray:
    """Returns colored pixels for the given art."""
    c = {ord(k): v for k, v in colors.items()}
    pixels = list(map(lambda a: c[int(a)], np.nditer(art)))
    return np.stack(pixels, 0).reshape(art.shape + (3,))


def create_image(shape: Tuple[int, int]) -> np.ndarray:
    H, W = shape
    return np.ones((H, W, 4, 4, 3), dtype=np.uint8) * 255


def finalize_image(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    return img.transpose(0, 2, 1, 3, 4).reshape((H * 4), (W * 4), 3)


class Sprite:
    def __init__(self, mask: np.ndarray, pixels: np.ndarray):
        self.mask = mask
        self.pixels = pixels

    @staticmethod
    def from_text(text: List[str], colors: Dict[int, Tuple[int, int, int]]) -> "Sprite":
        art = text_to_art(text)
        mask = art_mask(art)
        pixels = art_pixels(art, colors)
        return Sprite(mask, pixels)

    def draw(self, canvas: np.ndarray):
        """Renders sprite to image."""
        canvas[self.mask] = self.pixels[self.mask]


class ScaleIntensity(Sprite):
    def __init__(self, sprite: Sprite, factor: float = 0.7):
        self.factor = factor
        self.sprite = sprite

    def draw(self, canvas: np.ndarray):
        p = self.sprite.pixels[self.sprite.mask] * self.factor
        p = np.clip(p, 0, 255).astype(np.uint8)
        canvas[self.sprite.mask] = p
