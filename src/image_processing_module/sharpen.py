import numpy as np
import cv2
from PIL import Image

def sharpen_image(image_path):
    """
    Sharpens the image using the Laplacian filter and saves the result.

    Parameters:
        image_path (str): Path to the input image.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Apply Laplacian filter for sharpening on each channel
    def sharpen_channel(channel):
        laplacian = cv2.Laplacian(channel, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))  # Convert to uint8
        return cv2.addWeighted(channel, 1.5, laplacian, -0.5, 0)

    sharpened_b = sharpen_channel(b)
    sharpened_g = sharpen_channel(g)
    sharpened_r = sharpen_channel(r)

    # Merge the sharpened channels back into an RGB image
    sharpened = cv2.merge((sharpened_b, sharpened_g, sharpened_r))

    return sharpened

if __name__ == "__main__":
    path = "truck.jpg"
    sharpened_image = sharpen_image(path)
    cv2.imshow("Sharpened Image", sharpened_image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()