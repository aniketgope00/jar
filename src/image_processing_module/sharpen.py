import numpy as np
import cv2
from PIL import Image

def sharpen_image(image):
    """
    Sharpens the input image using a kernel.

    Parameters:
        image (PIL.Image.Image): Input image as a PIL Image object.

    Returns:
        PIL.Image.Image: Sharpened image as a PIL Image object.
    """
    # Convert PIL Image to NumPy array
    image_array = np.array(image)

    # Apply sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_array = cv2.filter2D(image_array, -1, kernel)

    # Convert back to PIL Image
    return Image.fromarray(sharpened_array)

if __name__ == "__main__":
    path = "truck.jpg"
    image = Image.open(path)
    sharpened_image = sharpen_image(image)
    sharpened_image.show()