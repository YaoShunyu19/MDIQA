import numpy as np
from PIL import Image
import matplotlib.cm as cm


def generate_heatmap(image_path, heatmap_tensor, output_path, alpha=0.3, norm=True):
    image = Image.open(image_path).convert("RGB")
    image.save(output_path)
    H, W, _ = np.array(image).shape
    if heatmap_tensor.ndim == 4:
        heatmap_tensor = heatmap_tensor.squeeze(0)
    
    heatmap = heatmap_tensor.squeeze(0).numpy()

    if norm:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    heatmap_resized = Image.fromarray(heatmap).resize((W, H), Image.BILINEAR)

    heatmap_colored = np.array(heatmap_resized)
    heatmap_colored = cm.jet(heatmap_colored)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_colored = Image.fromarray(heatmap_colored)

    heatmap_path = output_path[:-4] + '_map.png'
    heatmap_colored.save(heatmap_path)
    heatmap_colored = heatmap_colored.convert("RGBA")
    image = image.convert("RGBA")

    blended_image = Image.blend(image, heatmap_colored, alpha)

    blended_image = blended_image.convert("RGB")
    blended_image.save(output_path[:-4] + '_blend.png')

