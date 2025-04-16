import os
import shutil
import cv2 as cv
import numpy as np
from tkinter import Tk, Label, Button, filedialog, Checkbutton, IntVar, Toplevel
from PIL import Image, ImageTk

UPLOAD_DIR = "UPLOADED_FILE"
PROCESSED_DIR = "PROCESSED_IMAGE"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.geometry("500x300")

        # Variables for checkboxes
        self.denoise_var = IntVar()
        self.sharpen_var = IntVar()

        # UI Elements
        Label(root, text="Image Processing Workflow", font=("Arial", 16)).pack(pady=10)
        Button(root, text="Upload Image", command=self.upload_image).pack(pady=5)
        Checkbutton(root, text="Denoise", variable=self.denoise_var).pack()
        Checkbutton(root, text="Sharpen", variable=self.sharpen_var).pack()
        Button(root, text="Process Image", command=self.process_image).pack(pady=5)
        self.status_label = Label(root, text="", font=("Arial", 10))
        self.status_label.pack(pady=10)

        self.uploaded_image_path = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
        if file_path:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(UPLOAD_DIR, filename)
            shutil.copy(file_path, dest_path)
            self.uploaded_image_path = dest_path
            self.status_label.config(text=f"Uploaded: {filename}")
            self.display_image(file_path, "Uploaded Image")

    def process_image(self):
        if not self.uploaded_image_path:
            self.status_label.config(text="No image uploaded!")
            return

        image = cv.imread(self.uploaded_image_path)
        if image is None:
            self.status_label.config(text="Error: Unable to load image.")
            return

        processed_image = image

        # Apply denoise
        if self.denoise_var.get():
            processed_image = cv.fastNlMeansDenoisingColored(processed_image, None, 10, 10, 7, 21)
            self.status_label.config(text="Denoising applied.")

        # Apply sharpen
        if self.sharpen_var.get():
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed_image = cv.filter2D(processed_image, -1, kernel)
            self.status_label.config(text="Sharpening applied.")

        # Save processed image
        processed_image_path = os.path.join(PROCESSED_DIR, "processed_image.jpg")
        cv.imwrite(processed_image_path, processed_image)
        self.status_label.config(text="Processed image saved.")
        self.display_image(processed_image_path, "Processed Image")

    def display_image(self, image_path, title):
        # Open a new window to display the image
        window = Toplevel(self.root)
        window.title(title)

        # Load and display the image
        image = Image.open(image_path)
        image = image.resize((400, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)

        label = Label(window, image=photo)
        label.image = photo  # Keep a reference to avoid garbage collection
        label.pack()

if __name__ == "__main__":
    root = Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
