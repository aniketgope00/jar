import os
import shutil
import cv2 as cv
import numpy as np
from dearpygui import dearpygui as dpg

UPLOAD_DIR = "UPLOADED_FILE"

# Home Window - UPLOAD FILE

dpg.create_context()
dpg.create_viewport(title="File Upload Example", width=500, height=200)

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def clear_uploads():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)
    print(f"Cleared uploads in {UPLOAD_DIR}")

def upload_callback(sender, app_data):
    selected_file = app_data['file_path_name']
    if selected_file:
        filename = os.path.basename(selected_file)
        dest_path = os.path.join(UPLOAD_DIR, filename)
        shutil.copy(selected_file, dest_path)
        dpg.set_value("status_text", f"Uploaded: {filename}")
        if filename.endswith(".jpg"):
            display_image(dest_path)  # Show the "Image Display" window

def display_image(image_path):
    width, height, channels, data = dpg.load_image(image_path)  # Load image as texture
    with dpg.texture_registry():
        texture_tag = f"texture_{os.path.basename(image_path)}"
        dpg.add_static_texture(width, height, data, tag=texture_tag)
    with dpg.window(label="Image Display", width=600, height=400):  # Set window size to 600x400
        dpg.add_text(f"Displaying: {os.path.basename(image_path)}")
        dpg.add_image(texture_tag, width=600, height=400)  # Resize image to fit the window

def process_image(image_path, denoise=False, sharpen=False):
    processed_dir = "PROCESSED_IMAGE"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    image = cv.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    processed_image = image

    if denoise:
        processed_image = cv.fastNlMeansDenoisingColored(processed_image, None, 10, 10, 7, 21)
        print("Denoising applied.")

    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_image = cv.filter2D(processed_image, -1, kernel)
        print("Sharpening applied.")

    # Save the processed image
    processed_image_path = os.path.join(processed_dir, "processed_image.jpg")
    cv.imwrite(processed_image_path, processed_image)
    print(f"Processed image saved at: {processed_image_path}")

    # Load the saved image for display
    processed_image_rgb = cv.cvtColor(cv.imread(processed_image_path), cv.COLOR_BGR2RGB)
    with dpg.texture_registry():
        texture_tag = "processed_texture"
        dpg.add_static_texture(processed_image_rgb.shape[1], processed_image_rgb.shape[0], processed_image_rgb.flatten() / 255.0, tag=texture_tag)

    # Create a new window to display the processed image
    with dpg.window(label="Processed Image", width=600, height=400):
        dpg.add_text("Processed Image")
        dpg.add_image(texture_tag)

    # Ensure the application continues running
    print("Image processing completed. Processed image displayed in a new window.")

def submit_callback(sender, app_data):
    denoise = dpg.get_value("denoise_checkbox")
    sharpen = dpg.get_value("sharpen_checkbox")
    if os.listdir(UPLOAD_DIR):
        image_path = os.path.join(UPLOAD_DIR, os.listdir(UPLOAD_DIR)[0])
        process_image(image_path, denoise=denoise, sharpen=sharpen)
    else:
        print("No image uploaded to process.")

with dpg.file_dialog(directory_selector=False, show=False, callback=upload_callback, tag="file_dialog_id", width=500, height=400):
    dpg.add_file_extension(".jpg", color=(150, 255, 150, 255))  # allow all files

with dpg.window(label="File Upload Window", width=400, height=200):
    dpg.add_text("Click below to upload a file.")
    dpg.add_button(label="Browse", callback=lambda: dpg.show_item("file_dialog_id"))
    dpg.add_spacer(height=10)
    dpg.add_text("", tag="status_text")  # status display

with dpg.window(label="Image Processing Options", width=400, height=200, pos=(50, 300)):
    dpg.add_checkbox(label="Denoise", tag="denoise_checkbox")
    dpg.add_checkbox(label="Sharpen", tag="sharpen_checkbox")
    dpg.add_button(label="Submit", callback=submit_callback)

if __name__ == "__main__":
    dpg.setup_dearpygui()
    screen_width = dpg.get_viewport_client_width()
    screen_height = dpg.get_viewport_client_height()
    dpg.set_viewport_width(screen_width)
    dpg.set_viewport_height(screen_height)
    dpg.show_viewport()
    dpg.start_dearpygui()
    clear_uploads()
