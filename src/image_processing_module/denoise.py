import numpy as np
import cv2
from PIL import Image

def denoise_image(image_path):
    """
    Denoises the image using Non-Local Means Denoising and saves the result.

    Parameters:
        image_path (str): Path to the input image.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Apply Non-Local Means Denoising
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return denoised

if __name__ == "__main__":
    path = "truck.jpg"
    denoised_image = denoise_image(path)
    cv2.imshow("Denoised Image", denoised_image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()