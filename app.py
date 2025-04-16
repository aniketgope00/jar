import os
import shutil
from dearpygui import dearpygui as dpg

UPLOAD_DIR = "UPLOADED_FILE"


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
        dpg.set_value("status_text", f"✅ Uploaded: {filename}")

with dpg.file_dialog(directory_selector=False, show=False, callback=upload_callback, tag="file_dialog_id", width=500, height=400):
    dpg.add_file_extension(".jpg", color=(150, 255, 150, 255))  # allow all files

with dpg.window(label="File Upload Window", width=400, height=200):
    dpg.add_text("Click below to upload a file.")
    dpg.add_button(label="Browse", callback=lambda: dpg.show_item("file_dialog_id"))
    dpg.add_spacer(height=10)
    dpg.add_text("", tag="status_text")  # status displayory_selector=False, show=False, callback=upload_callback, tag="file_dialog_id")


if __name__ == "__main__":
    dpg.setup_dearpygui()
    screen_width = dpg.get_viewport_client_width()
    screen_height = dpg.get_viewport_client_height()
    dpg.set_viewport_width(screen_width)
    dpg.set_viewport_height(screen_height)
    dpg.show_viewport()
    dpg.start_dearpygui()
    clear_uploads()
