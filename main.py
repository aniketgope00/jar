import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QGridLayout, QWidget,
    QFileDialog, QLabel, QTabWidget, QVBoxLayout, QHBoxLayout, QCheckBox
)
from PyQt5.QtGui import QPalette, QColor, QPixmap, QMovie, QMouseEvent, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import os
import shutil


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

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
                padding: 10px 20px;  /* Vertical: 10px, Horizontal: 20px */
                font: bold 14px "Segoe UI";
                border: none;
                border-radius: 8px;
                margin: 4px;
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

        # Label for instructions in the settings tab
        instructions_label = QLabel("Add preprocessing options, set 3 markers for the segmentation models")
        instructions_label.setStyleSheet("color: white; font-weight: bold;")
        instructions_label.setAlignment(Qt.AlignCenter)
        settings_layout.addWidget(instructions_label, 0, 0, 1, 2)

        # Image Viewer at grid position (1, 0)
        self.original_image_label = QLabel("No image uploaded")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setStyleSheet("color: white;")
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

        # Button to print selected options
        self.print_button = QPushButton("Print Options")
        self.print_button.setObjectName("actionButton")
        self.print_button.clicked.connect(self.print_options)
        settings_layout.addWidget(self.print_button, 3, 0, 1, 2)

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
        """)

    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Upload Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            upload_dir = "UPLOADED_FILE"
            os.makedirs(upload_dir, exist_ok=True)
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(upload_dir, file_name)
            shutil.copy(file_path, destination_path)
            self.image_label.setText(f"Uploaded: {destination_path}")
            pixmap = QPixmap(destination_path)
            self.image_display.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            self.original_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))  # Update settings tab viewer

    def general_renders(self):
        print("Option 1 clicked!")

    def switch_to_settings_tab(self):
        self.tab_widget.setCurrentIndex(1)  # Switch to the "Settings" tab (index 1)

    def print_options(self):
        denoise = self.denoise_checkbox.isChecked()
        sharpen = self.sharpen_checkbox.isChecked()
        print(f"Denoise: {denoise}, Sharpen: {sharpen}")

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
