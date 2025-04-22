import numpy as np
import cv2
from PIL import Image
import os

def canny_edge_detector(image_path):
    """
    Applies Canny edge detection to the input image.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        PIL.Image.Image: Image with edges detected as a PIL Image object.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, 100, 200)

    # Convert edges to a PIL Image
    return Image.fromarray(edges)

def sobel_edge_detector(image_path):
    """
    Applies Sobel edge detection to the input image.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        PIL.Image.Image: Image with edges detected as a PIL Image object.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)

    # Apply Sobel operator in both x and y directions
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)

    # Calculate the magnitude of gradients
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))

    # Convert edges to a PIL Image
    return Image.fromarray(sobel_magnitude)


def laplacian_edge_detector(image_path):
    """
    Applies Laplacian edge detection to the input image.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        PIL.Image.Image: Image with edges detected as a PIL Image object.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    laplacian = np.uint8(np.clip(laplacian, 0, 255))

    # Convert edges to a PIL Image
    return Image.fromarray(laplacian)


if __name__ == "__main__":
    image_path = "truck.jpg"  # Replace with your image path
    processed_dir = "PROCESSED_IMAGE"
    os.makedirs(processed_dir, exist_ok=True)

    try:
        # Canny Edge Detection
        canny_image = canny_edge_detector(image_path)
        canny_image_path = os.path.join(processed_dir, "canny_edge.jpg")
        canny_image.save(canny_image_path)
        print(f"Canny edge-detected image saved at: {canny_image_path}")

        # Sobel Edge Detection
        sobel_image = sobel_edge_detector(image_path)
        sobel_image_path = os.path.join(processed_dir, "sobel_edge.jpg")
        sobel_image.save(sobel_image_path)
        print(f"Sobel edge-detected image saved at: {sobel_image_path}")

        # Laplacian Edge Detection
        laplacian_image = laplacian_edge_detector(image_path)
        laplacian_image_path = os.path.join(processed_dir, "laplacian_edge.jpg")
        laplacian_image.save(laplacian_image_path)
        print(f"Laplacian edge-detected image saved at: {laplacian_image_path}")
    except (FileNotFoundError, ValueError) as e:
        print(e)