import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QWidget, QFileDialog, QLabel
from PyQt5.QtGui import QPalette, QColor, QPixmap, QMovie
from PyQt5.QtCore import Qt
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
        self.background_label.setGeometry(0, 0, 800, 800)  # Set to cover the entire window
        self.background_label.setScaledContents(True)

        # Load and play the GIF
        self.movie = QMovie("utils/lines-4497.gif")  # Replace with the path to your GIF file
        self.background_label.setMovie(self.movie)
        self.movie.start()

        # Ensure the background is behind other widgets
        self.background_label.lower()

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create layout
        layout = QGridLayout()  # Grid layout for better positioning

        # Create a label for displaying the uploaded image
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_display, 0, 0, 1, 2)  # Span across 2 columns

        # Create a label for displaying the uploaded image path
        self.image_label = QLabel("No image uploaded")
        self.image_label.setStyleSheet("color: white;")
        layout.addWidget(self.image_label, 1, 0, 1, 2)  # Span across 2 columns

        # Create an upload image button
        upload_button = QPushButton("Upload Image")
        upload_button.setObjectName("uploadButton")
        upload_button.clicked.connect(self.upload_image)
        layout.addWidget(upload_button, 2, 0, 1, 2)  # Span across 2 columns

        # Create buttons
        button1 = QPushButton("Create General Renders")
        button1.setObjectName("evilButton")  # Assign object name
        button1.clicked.connect(self.general_renders)

        '''
        button2 = QPushButton("Create Fashion Renders")
        button2.setObjectName("evilButton")  # Assign object name
        button2.clicked.connect(self.fashion_renders)
        '''

        # Add buttons to layout
        layout.addWidget(button1, 3, 0)
        layout.addWidget(button2, 3, 1)

        # Set layout to central widget
        central_widget.setLayout(layout)

        # Align items at the center of the screen
        layout.setAlignment(Qt.AlignCenter)

        # Add spacing between buttons
        layout.setHorizontalSpacing(50)  # Horizontal spacing between columns
        layout.setVerticalSpacing(100)    # Vertical spacing between rows

        # Apply stylesheet
        self.setStyleSheet("""
            QPushButton#evilButton {
                background-color: #eb5e34;
                border-style: outset;
                border-width: 2px;
                border-radius: 10px;
                border-color: beige;
                font: bold 14px;
                color: white;
                min-width: 15em;  /* Increased width */
                min-height: 3em;  /* Increased height */
                padding: 6px;
            }
            QPushButton#evilButton:pressed {
                background-color: rgb(224, 0, 0);
                border-style: inset;
            }
            QPushButton#uploadButton {
                background-color: #34eb83;
                border-style: outset;
                border-width: 2px;
                border-radius: 10px;
                border-color: beige;
                font: bold 14px;
                color: black;
                min-width: 15em;  /* Increased width */
                min-height: 3em;  /* Increased height */
                padding: 6px;
            }
            QPushButton#uploadButton:pressed {
                background-color: rgb(0, 224, 0);
                border-style: inset;
            }
        """)

    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Upload Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # Ensure the "UPLOADED_FILE" directory exists
            upload_dir = "UPLOADED_FILE"
            os.makedirs(upload_dir, exist_ok=True)

            # Copy the uploaded file to the directory
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(upload_dir, file_name)
            shutil.copy(file_path, destination_path)

            # Update the label and display the image
            self.image_label.setText(f"Uploaded: {destination_path}")
            pixmap = QPixmap(destination_path)
            self.image_display.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))  # Scale image to fit

    def general_renders(self):
        print("Option 1 clicked!")

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())