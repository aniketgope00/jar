import os
import cv2
import numpy as np
from ultralytics import SAM
import matplotlib.pyplot as plt
import argparse
import logging

# Use the same log file as main.py
logging.basicConfig(
    filename="application.log",  # Same log file as main.py
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_sam_model(model_path="sam2_s.pt"):
    """Load the SAM model."""
    try:
        logging.info(f"Loading SAM model from path: {model_path}")
        model = SAM(model_path)
        logging.info("SAM model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error loading SAM model: {e}")
        return None

def segment_image(model, image_path, output_dir="segments"):
    """
    Segment an image using SAM model and save individual segments.
    Returns:
        masks (List[np.ndarray]): List of mask arrays (bool)
        scores (List[float]): Confidence scores for each mask
    """
    logging.info(f"Starting segmentation for image: {image_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        logging.error(f"Error: Could not read image at {image_path}")
        return [], []

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Create a grid of points across the image
    grid_size = 5
    x_points = np.linspace(0, w - 1, grid_size, dtype=int)
    y_points = np.linspace(0, h - 1, grid_size, dtype=int)
    xx, yy = np.meshgrid(x_points, y_points)
    point_coords = np.column_stack((xx.ravel(), yy.ravel()))
    point_labels = np.ones(len(point_coords))

    try:
        results = model(
            img_rgb,
            points=point_coords.tolist(),
            labels=point_labels.tolist()
        )

        if not results or not results[0] or not results[0].masks:
            logging.warning("No segments found in the image")
            return [], []

        masks = results[0].masks.data.cpu().numpy().astype(bool)
        scores = results[0].masks.data.cpu().numpy().max(axis=(1, 2))

        composite_mask = np.zeros((h, w, 4), dtype=np.float32)
        colors = [np.concatenate([np.random.random(3), [0.5]]) for _ in range(len(masks))]

        for i, (mask, score) in enumerate(zip(masks, scores)):
            result_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
            result_rgba[~mask] = [255, 255, 255, 0]
            output_path = os.path.join(output_dir, f'segment_{i + 1}.png')
            cv2.imwrite(output_path, result_rgba)

            mask_image = np.zeros((h, w, 4), dtype=np.float32)
            mask_image[mask] = colors[i]
            composite_mask = np.maximum(composite_mask, mask_image)

            logging.info(f"Saved segment {i + 1} with confidence score: {score:.3f}")

        composite_path = os.path.join(output_dir, 'composite.png')
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.imshow(composite_mask)
        plt.axis('off')
        plt.savefig(composite_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        logging.info(f"Composite image saved at: {composite_path}")

        logging.info(f"Processing complete! Found {len(masks)} segments.")
        logging.info(f"Results saved in: {output_dir}")
        return masks, scores

    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        return [], []

def main():
    parser = argparse.ArgumentParser(description="Segment an image using SAM model")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output", "-o", default="segments", help="Output directory (default: segments)")
    parser.add_argument("--model", "-m", default="sam2_s.pt", help="Path to SAM model (default: sam2_s.pt)")
    args = parser.parse_args()

    logging.info("Starting SAM segmentation script")
    model = load_sam_model(args.model)
    if model is None:
        logging.error("Failed to load SAM model. Exiting.")
        return

    masks, scores = segment_image(model, args.image_path, args.output)
    
    logging.info("Returned masks array shapes:")
    for i, mask in enumerate(masks):
        logging.info(f"Mask {i+1}: {mask.shape}, Confidence: {scores[i]:.3f}")

if __name__ == "__main__":
    image_path = "image_processing_module/truck.jpg"  # Replace with your image path
    os.makedirs("PROCESSED_IMAGE", exist_ok=True)
    processed_dir = "PROCESSED_IMAGE"
    logging.info("Running SAM API as standalone script")
    model = load_sam_model("sam2_s.pt")
    if model is None:
        logging.error("Failed to load SAM model.")
    else:
        masks, scores = segment_image(model, image_path, "segments")
