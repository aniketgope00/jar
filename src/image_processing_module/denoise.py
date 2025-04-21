import numpy as np
import cv2
from PIL import Image

def denoise_image(image):
    """
    Denoises the input image using Non-Local Means Denoising.

    Parameters:
        image (PIL.Image.Image): Input image as a PIL Image object.

    Returns:
        PIL.Image.Image: Denoised image as a PIL Image object.
    """
    # Convert PIL Image to NumPy array
    image_array = np.array(image)

    # Apply Non-Local Means Denoising
    denoised_array = cv2.fastNlMeansDenoisingColored(image_array, None, 10, 10, 7, 21)

    # Convert back to PIL Image
    return Image.fromarray(denoised_array)

if __name__ == "__main__":
    path = "truck.jpg"
    image = Image.open(path)
    denoised_image = denoise_image(image)
    denoised_image.show()