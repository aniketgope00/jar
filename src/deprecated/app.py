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

    # Show the "Image Processing Options" window after displaying the image
    dpg.show_item("image_processing_window")

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

    # Remove display logic
    print("Image processing completed. Processed image saved.")
    return processed_image_path

def submit_callback(sender, app_data):
    denoise = dpg.get_value("denoise_checkbox")
    sharpen = dpg.get_value("sharpen_checkbox")
    if os.listdir(UPLOAD_DIR):
        image_path = os.path.join(UPLOAD_DIR, os.listdir(UPLOAD_DIR)[0])
        processed_image_path = process_image(image_path, denoise=denoise, sharpen=sharpen)
        dpg.set_value("status_text", "Image processed and saved.")
        dpg.set_value("process_status_text", "Image processed and saved successfully.")  # Update process status

        # List to store marker positions
        markers = []

        # Create a new window to display the processed image
        with dpg.window(label="Processed Image Viewer", width=800, height=600, pos=(100, 100)):
            dpg.add_text("Click on the image to set markers.")
            width, height, channels, data = dpg.load_image(processed_image_path)  # Load processed image as texture
            with dpg.texture_registry():
                texture_tag = "processed_image_texture"
                dpg.add_static_texture(width, height, data, tag=texture_tag)

            # Add a drawlist for displaying the image
            with dpg.drawlist(width=width, height=height, tag="overlay_layer"):
                dpg.draw_image(texture_tag, pmin=(0, 0), pmax=(width, height))

            # Add a mouse click handler for storing marker positions
            def mouse_click_handler(sender, app_data):
                mouse_pos = dpg.get_mouse_pos(local=False)
                drawlist_pos = dpg.get_item_pos("overlay_layer")
                x, y = mouse_pos[0] - drawlist_pos[0], mouse_pos[1] - drawlist_pos[1]
                if 0 <= x <= width and 0 <= y <= height:  # Ensure click is within image bounds
                    markers.append((int(x), int(y)))  # Store marker position
                    dpg.add_text(f"Marker set at: ({int(x)}, {int(y)})", parent="marker_container")

            dpg.add_handler_registry(tag="mouse_handler_registry")
            dpg.add_mouse_click_handler(callback=mouse_click_handler, parent="mouse_handler_registry")

            dpg.add_spacer(height=10)
            with dpg.child_window(tag="marker_container", width=780, height=200):
                dpg.add_text("Markers:")
    else:
        print("No image uploaded to process.")

with dpg.file_dialog(directory_selector=False, show=False, callback=upload_callback, tag="file_dialog_id", width=500, height=400):
    dpg.add_file_extension(".jpg", color=(150, 255, 150, 255))  # allow all files

with dpg.window(label="File Upload Window", width=400, height=200):
    dpg.add_text("Click below to upload a file.")
    dpg.add_button(label="Browse", callback=lambda: dpg.show_item("file_dialog_id"))
    dpg.add_spacer(height=10)
    dpg.add_text("", tag="status_text")  # status display

with dpg.window(label="Image Processing Options", width=400, height=200, pos=(50, 300), tag="image_processing_window", show=False):
    dpg.add_checkbox(label="Denoise", tag="denoise_checkbox")
    dpg.add_checkbox(label="Sharpen", tag="sharpen_checkbox")
    dpg.add_button(label="Submit", callback=submit_callback)
    dpg.add_text("", tag="process_status_text")  # Text to display process status

if __name__ == "__main__":
    dpg.setup_dearpygui()
    screen_width = dpg.get_viewport_client_width()
    screen_height = dpg.get_viewport_client_height()
    dpg.set_viewport_width(screen_width)
    dpg.set_viewport_height(screen_height)
    dpg.show_viewport()
    dpg.start_dearpygui()
    clear_uploads()
