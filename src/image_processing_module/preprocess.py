import numpy as np
import cv2
from PIL import Image
from src.image_processing_module.denoise import denoise_image  # Corrected import path
from src.image_processing_module.sharpen import sharpen_image  # Corrected import path


def preprocess_image(image_path, responses):
    """
    Preprocesses an image based on user responses for denoising and sharpening.

    Args:
        image_path (str): Path to the input image.
        responses (dict): Dictionary containing user responses for denoising and sharpening.
    """
    # Load the image
    image = Image.open(image_path).convert('RGB')

    if responses['Denoise']:
        # Denoise the image
        image = denoise_image(image)  # Pass PIL Image object
        print("Image denoised.")

    if responses['Sharpen']:
        # Sharpen the image
        image = sharpen_image(image)  # Pass PIL Image object
        print("Image sharpened.")

    if not responses['Denoise'] and not responses['Sharpen']:
        print("No preprocessing applied.")
    return image.convert('RGB')  # Ensure the returned image is in RGB format


if __name__ == "__main__":
    responses = {'Denoising': True, 'Sharpening': True}
    image_path = "image_processing_module/test_images/output.png"  # Replace with your image path
    preprocessed_image = preprocess_image(image_path, responses)
    cv2.imshow("Preprocessed Image", np.array(preprocessed_image))
    cv2.waitKey(100000)
    cv2.destroyAllWindows()