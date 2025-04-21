import sys
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QGridLayout, QWidget,
    QFileDialog, QLabel, QTabWidget, QVBoxLayout, QHBoxLayout, QCheckBox
)
from PyQt5.QtGui import QPalette, QColor, QPixmap, QMovie, QMouseEvent, QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint
import os
import shutil
import cv2
import numpy as np
from src.image_processing_module import preprocess
#from src.segmentation_module.models import get_segmented_image  # Import the function

# Configure logging
logging.basicConfig(
    filename="application.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        logging.info("MainWindow initialized.")

        self.setWindowTitle("Two Options App")
        self.setGeometry(100, 100, 800, 800)

        # Set window background color to black
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("black"))
        self.setPalette(palette)

        # Create a label for the background GIF
        self.background_label = QLabel(self)
        self.background_label.setGeometry(0, 0, 800, 800)
        self.background_label.setScaledContents(True)

        # Load and play the GIF
        self.movie = QMovie("utils/lines-4497.gif")
        self.background_label.setMovie(self.movie)
        self.movie.start()
        self.background_label.lower()

        # Central tab widget
        self.tab_widget = QTabWidget()  # Make tab_widget an instance variable
        self.setCentralWidget(self.tab_widget)

        # Style tabs
        self.tab_widget.setStyleSheet("""
    QTabWidget::pane {
        background: black;
        border: none;
        padding: 0px;
    }

    QWidget {
        background: black;
    }

    QTabBar::tab {
        background: #1a1a1a;
        color: white;
        padding: 10px 20px;
        font: bold 14px "Segoe UI";
        border: none;
        border-radius: 8px;
        margin: 4px;
        min-width: 120px;  /* Ensures tabs are wide enough for text */
        min-height: 20px; /* Ensures tabs are tall enough for text */
    }

    QTabBar::tab:selected {
        background: #333333;
        color: white;
        margin-bottom: 0px;
    }

    QTabBar::tab:hover {
        background: #2a2a2a;
    }
""")


        # Main Tab
        main_tab = QWidget()
        self.tab_widget.addTab(main_tab, "Main")

        # Layout for main tab
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_tab.setLayout(main_layout)

        # Image Display
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_display)

        self.image_label = QLabel("No image uploaded")
        self.image_label.setStyleSheet("color: white;")
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)

        # Buttons layout
        button_layout = QVBoxLayout()
        button_layout.setSpacing(20)
        button_layout.setAlignment(Qt.AlignCenter)

        button_size = (240, 50)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setObjectName("actionButton")
        self.upload_button.setFixedSize(*button_size)
        self.upload_button.clicked.connect(self.upload_image)

        self.button1 = QPushButton("Create General Renders")
        self.button1.setObjectName("actionButton")
        self.button1.setFixedSize(*button_size)
        self.button1.clicked.connect(self.switch_to_settings_tab)  # Link to switch tab

        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.button1)
        main_layout.addLayout(button_layout)

        # Settings Tab
        settings_tab = QWidget()
        self.tab_widget.addTab(settings_tab, "Settings")
        settings_layout = QGridLayout()
        settings_tab.setLayout(settings_layout)

        # Apply qtcss styling to the settings tab
        settings_tab.setStyleSheet("""
            QWidget {
                background: black;
            }
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QCheckBox {
                color: white;
                font-size: 12px;
            }
            QPushButton#actionButton {
                background-color: #eb5e34;
                border-style: outset;
                border-width: 2px;
                border-radius: 10px;
                border-color: beige;
                font: bold 14px;
                color: white;
                padding: 6px;
            }
            QPushButton#actionButton:pressed {
                background-color: rgb(224, 0, 0);
                border-style: inset;
            }
        """)

        # Label for instructions in the settings tab
        instructions_label = QLabel("Add preprocessing options, set 3 markers for the segmentation models")
        instructions_label.setStyleSheet("color: white; font-weight: bold;")
        instructions_label.setAlignment(Qt.AlignCenter)
        settings_layout.addWidget(instructions_label, 0, 0, 1, 2)

        # Image Viewer at grid position (1, 0)
        self.original_image_label = QLabel("No image uploaded")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setStyleSheet("color: white; margin: 0px; padding: 0px;")  # Reduced padding/margin
        settings_layout.addWidget(self.original_image_label, 1, 0, 1, 2)

        # Enable mouse tracking for the original image label
        self.original_image_label.setMouseTracking(True)
        self.original_image_label.mousePressEvent = self.add_marker

        # List to store marker positions
        self.markers = []

        # Checkboxes for denoising and sharpening
        self.denoise_checkbox = QCheckBox("Denoise")
        self.denoise_checkbox.setStyleSheet("color: white;")
        settings_layout.addWidget(self.denoise_checkbox, 2, 0)

        self.sharpen_checkbox = QCheckBox("Sharpen")
        self.sharpen_checkbox.setStyleSheet("color: white;")
        settings_layout.addWidget(self.sharpen_checkbox, 2, 1)

        # Process Image Button
        self.process_button = QPushButton("Process Image")
        self.process_button.setObjectName("evilButton")
        self.process_button.setFixedSize(200, 40)  # Reduced button size
        self.process_button.clicked.connect(self.process_image)
        settings_layout.addWidget(self.process_button, 3, 0, 1, 2, alignment=Qt.AlignCenter)  # Center aligned

        # Logs Tab
        logs_tab = QWidget()
        self.tab_widget.addTab(logs_tab, "Logs")

        # Layout for logs tab
        logs_layout = QVBoxLayout()
        logs_tab.setLayout(logs_layout)

        # Logs display area
        self.logs_display = QLabel()
        self.logs_display.setStyleSheet("color: white; background-color: #1a1a1a; padding: 10px;")
        self.logs_display.setAlignment(Qt.AlignTop)
        self.logs_display.setWordWrap(True)
        logs_layout.addWidget(self.logs_display)

        # Refresh logs button
        refresh_button = QPushButton("Refresh Logs")
        refresh_button.setObjectName("actionButton")
        refresh_button.clicked.connect(self.refresh_logs)
        logs_layout.addWidget(refresh_button)

        # Load logs initially
        self.refresh_logs()

        # Global Stylesheet for the application
        self.setStyleSheet("""
            * {
                font-family: "Segoe UI";
            }
            QPushButton#actionButton {
                background-color: #eb5e34;
                border-style: outset;
                border-width: 2px;
                border-radius: 10px;
                border-color: beige;
                font: bold 14px;
                color: white;
                padding: 6px;
            }
            QPushButton#actionButton:pressed {
                background-color: rgb(224, 0, 0);
                border-style: inset;
            }
            QPushButton#evilButton {
                background-color: #ff5733;
                border-style: outset;
                border-width: 2px;
                border-radius: 8px;
                border-color: darkred;
                font: bold 12px;
                color: white;
                padding: 4px;
            }
            QPushButton#evilButton:pressed {
                background-color: #c70039;
                border-style: inset;
            }
        """)

    def closeEvent(self, event):
        """Clear logs when the application is closed."""
        try:
            with open("application.log", "w") as log_file:
                log_file.write("")  # Clear the log file
            self.logs_display.setText("")  # Clear the logs display in the UI
            logging.info("Application closed. Logs cleared.")
        except Exception as e:
            logging.error(f"Failed to clear logs: {e}")
        event.accept()

    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Upload Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            logging.info(f"Image uploaded: {file_path}")
            upload_dir = "UPLOADED_FILE"
            os.makedirs(upload_dir, exist_ok=True)
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(upload_dir, file_name)
            shutil.copy(file_path, destination_path)
            self.image_label.setText(f"Uploaded: {destination_path}")
            pixmap = QPixmap(destination_path)
            self.image_display.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            self.original_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))  # Update settings tab viewer
            self.uploaded_image_path = destination_path  # Store the uploaded image path
            logging.info(f"Uploaded image path stored: {self.uploaded_image_path}")

    def general_renders(self):
        logging.info("General renders option clicked.")
        print("Option 1 clicked!")

    def switch_to_settings_tab(self):
        logging.info("Switched to settings tab.")
        self.tab_widget.setCurrentIndex(1)  # Switch to the "Settings" tab (index 1)

    def print_options(self):
        """Log selected options instead of printing."""
        denoise = self.denoise_checkbox.isChecked()
        sharpen = self.sharpen_checkbox.isChecked()
        logging.info(f"Options selected - Denoise: {denoise}, Sharpen: {sharpen}")

    def add_marker(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Limit the number of markers to 3
            if len(self.markers) >= 3:
                print("Maximum of 3 markers allowed.")
                return

            # Get the position of the click relative to the image label
            x = event.pos().x()
            y = event.pos().y()

            # Adjust for the label's alignment and scaling
            pixmap = self.original_image_label.pixmap()
            if pixmap:
                label_width = self.original_image_label.width()
                label_height = self.original_image_label.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()

                # Calculate scaling factors
                scale_x = pixmap_width / label_width
                scale_y = pixmap_height / label_height

                # Center the pixmap within the label if necessary
                offset_x = max((label_width - pixmap_width / scale_x) // 2, 0)
                offset_y = max((label_height - pixmap_height / scale_y) // 2, 0)

                # Adjust the click position to match the pixmap's coordinate system
                x = int((x - offset_x) * scale_x)
                y = int((y - offset_y) * scale_y)

            self.markers.append((x, y))
            logging.info(f"Marker added at: ({x}, {y})")
            print(f"Marker added at: ({x}, {y})")

            # Reload the image with the new marker
            self.update_image_with_markers()

    def update_image_with_markers(self):
        # Reload the original image
        pixmap = self.original_image_label.pixmap()
        if pixmap:
            pixmap_copy = pixmap.copy()
            painter = QPainter(pixmap_copy)
            pen = QPen(Qt.red)
            pen.setWidth(3)  # Increase pen width for better visibility
            painter.setPen(pen)

            # Draw all markers as larger circles
            for marker in self.markers:
                painter.drawEllipse(QPoint(marker[0], marker[1]), 10, 10)  # Circle with radius 10

            painter.end()
            self.original_image_label.setPixmap(pixmap_copy)
            logging.info("Image updated with markers.")

    def process_image(self):
        """Process the uploaded image based on checkbox responses."""
        if not hasattr(self, "uploaded_image_path") or not self.uploaded_image_path:
            logging.warning("No image uploaded to process.")
            self.image_label.setText("No image uploaded to process.")
            return

        # Collect checkbox responses
        responses = {
            "Denoise": self.denoise_checkbox.isChecked(),
            "Sharpen": self.sharpen_checkbox.isChecked(),
        }
        logging.info(f"Processing image with responses: {responses}")

        # Process the image
        processed_image = preprocess.preprocess_image(self.uploaded_image_path, responses)

        # Save the processed image in RGB format
        processed_dir = "PROCESSED_IMAGE"
        os.makedirs(processed_dir, exist_ok=True)
        processed_image_path = os.path.join(processed_dir, "processed_image.jpg")
        processed_image.save(processed_image_path)  # Save as RGB
        logging.info(f"Processed image saved at: {processed_image_path}")

        # Display original and processed images
        self.display_images_side_by_side()

    def display_images_side_by_side(self):
        """Display the input image and the processed image side by side."""
        if not hasattr(self, "uploaded_image_path") or not self.uploaded_image_path:
            logging.warning("No image uploaded to display.")
            return

        processed_dir = "PROCESSED_IMAGE"
        processed_image_path = os.path.join(processed_dir, "processed_image.jpg")
        if not os.path.exists(processed_image_path):
            logging.warning("Processed image not found.")
            return

        # Load images
        input_pixmap = QPixmap(self.uploaded_image_path)
        processed_image = QPixmap(processed_image_path)  # Load directly as RGB
        processed_pixmap = processed_image

        # Create a new tab for comparison
        comparison_tab = QWidget()
        self.tab_widget.addTab(comparison_tab, "Comparison")

        # Layout for side-by-side display
        layout = QVBoxLayout()
        comparison_tab.setLayout(layout)

        # Horizontal layout for images
        image_layout = QHBoxLayout()

        # Input image display
        input_label = QLabel()
        input_label.setPixmap(input_pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        input_label.setAlignment(Qt.AlignCenter)
        input_label.setStyleSheet("border: 2px solid gray;")
        image_layout.addWidget(input_label)

        # Caption for input image
        input_caption = QLabel("Original Image")
        input_caption.setAlignment(Qt.AlignCenter)
        input_caption.setStyleSheet("color: white; font-size: 12px;")
        layout.addWidget(input_caption)

        # Processed image display
        processed_label = QLabel()
        processed_label.setPixmap(processed_pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        processed_label.setAlignment(Qt.AlignCenter)
        processed_label.setStyleSheet("border: 2px solid gray;")
        image_layout.addWidget(processed_label)

        # Caption for processed image
        processed_caption = QLabel("Processed Image")
        processed_caption.setAlignment(Qt.AlignCenter)
        processed_caption.setStyleSheet("color: white; font-size: 12px;")
        layout.addWidget(processed_caption)

        layout.addLayout(image_layout)

        # Add "Open 3D Model" button
        open_3d_model_button = QPushButton("Open 3D Model")
        open_3d_model_button.setObjectName("actionButton")
        open_3d_model_button.clicked.connect(self.open_3d_model_tab)
        layout.addWidget(open_3d_model_button)

        # Switch to the comparison tab
        self.tab_widget.setCurrentWidget(comparison_tab)

    def refresh_logs(self):
        """Load and display the contents of the log file."""
        try:
            with open("application.log", "r") as log_file:
                logs = log_file.read()
                self.logs_display.setText(logs)
                logging.info("Logs refreshed.")
        except FileNotFoundError:
            self.logs_display.setText("No logs available.")
            logging.warning("Log file not found.")

    def open_3d_model_tab(self):
        """Open a new tab and display the text 'See 3D model here'."""
        model_tab = QWidget()
        self.tab_widget.addTab(model_tab, "3D Model")

        # Layout for the 3D model tab
        model_layout = QVBoxLayout()
        model_tab.setLayout(model_layout)

        # Add a label with the text
        model_label = QLabel("See 3D model here")
        model_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        model_label.setAlignment(Qt.AlignCenter)
        model_layout.addWidget(model_label)

        # Switch to the new tab
        self.tab_widget.setCurrentWidget(model_tab)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
