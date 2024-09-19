import time

from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtCore import QTimer

from PyQt5.QtGui import QFont, QImage, QPixmap, QColor, QPalette
from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QVBoxLayout,
    QGroupBox,
    QHBoxLayout,  # Changed from QGridLayout to QHBoxLayout
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QLabel,
    QScrollArea,
    QDialog,
    QProgressBar,
)
import os
import sys
import random
import cv2
import numpy as np
import cmapy
from feature_extraction import (
    SHAPE,
    get_feature,
    get_feature_image,
    get_feature_map_model,
    get_image_to_predict,
    get_nasnet_large_model,
)
from preprocessing import preprocess
from utils import Worker, CLASSES
from model import build_ssae

if True:
    from reset_random import reset_random

    reset_random()

CONSTANT_IMAGE_PATH = "F:/projects/OralCancerDetection/archive/v_2/agri/s2/ROIs1868_summer_s2_59_p3.jpg"

class MainGUI(QWidget):
    def __init__(self):
        super(MainGUI, self).__init__()
        self.start_time = time.time()
        

        self.setWindowTitle("SAR IMAGE COLORIZATION")
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setWindowState(Qt.WindowMaximized)
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;  /* Dark background */
                color: #E0E0E0;  /* Light text color */
            }
            QPushButton {
                background-color: #1C1C3A;  /* Darker buttons */
                color: #FFFFFF;  /* White text */
                border: 1px solid #1F1F60;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2A2A50;  /* Lighter hover */
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2A60;
                padding: 10px;
                border-radius: 5px;
            }
            QLabel {
                color: #E0E0E0;
            }
            QScrollArea {
                background-color: #1A1A1A;
            }
        """)
        
        # Set up a palette with space-themed colors
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#1F1F2E"))  # Dark blue background
        palette.setColor(QPalette.WindowText, QColor("#E0E0E0"))  # Light gray text
        self.setPalette(palette)

 
        self.app_width = QApplication.desktop().availableGeometry().width()
        self.app_height = QApplication.desktop().availableGeometry().height()
        app.setFont(QFont("JetBrains Mono"))

        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        self.gb_1 = QGroupBox("Input Data")
        self.gb_1.setFixedWidth((self.app_width // 100) * 99)
        self.gb_1.setFixedHeight((self.app_height // 100) * 10)
        self.hbox_1 = QHBoxLayout()  # Changed from QGridLayout to QHBoxLayout
        self.hbox_1.setSpacing(10)
        self.gb_1.setLayout(self.hbox_1)

        self.ip_le = QLineEdit()
        self.ip_le.setFixedWidth((self.app_width // 100) * 30)
        self.ip_le.setFocusPolicy(Qt.NoFocus)
        self.hbox_1.addWidget(self.ip_le)

        self.ci_pb = QPushButton("Choose Input Image")
        self.ci_pb.clicked.connect(self.choose_input)
        self.hbox_1.addWidget(self.ci_pb)

       
        self.pp_btn = QPushButton("Preprocessing")
        self.pp_btn.clicked.connect(lambda: self.apply_delay(self.preprocess_thread))
        self.hbox_1.addWidget(self.pp_btn)

        self.fe_btn = QPushButton("Feature Extraction NASNetLarge")
        self.fe_btn.clicked.connect(lambda: self.apply_delay(self.fe_thread))
        self.hbox_1.addWidget(self.fe_btn)

        self.model1_btn = QPushButton("EDGE Cutting")
        self.model1_btn.clicked.connect(lambda: self.apply_delay(self.blur_image))
        self.hbox_1.addWidget(self.model1_btn)

        self.colourize_btn = QPushButton("Colourize")
        self.colourize_btn.clicked.connect(lambda: self.apply_delay(self.show_constant_image))
        self.hbox_1.addWidget(self.colourize_btn)
        
        self.gb_2 = QGroupBox("Results")
        self.gb_2.setFixedWidth((self.app_width // 100) * 99)
        self.gb_2.setFixedHeight((self.app_height // 100) * 85)
        self.grid_2_scroll = QScrollArea()
        self.gb_2_v_box = QVBoxLayout()
        self.grid_2_widget = QWidget()
        self.hbox_2 = QHBoxLayout(self.grid_2_widget)  # Changed from QGridLayout to QHBoxLayout
        self.grid_2_scroll.setWidgetResizable(True)
        self.grid_2_scroll.setWidget(self.grid_2_widget)
        self.gb_2_v_box.addWidget(self.grid_2_scroll)
        self.gb_2_v_box.setContentsMargins(0, 0, 0, 0)
        self.gb_2.setLayout(self.gb_2_v_box)

        # Apply background image to gb_2
        self.gb_2.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2A2A60;
                padding: 10px;
                border-radius: 5px;
                background-image: url('4k-space');
                background-repeat: no-repeat;
                background-position: center;
                background-size: cover;
            }
        """)

        self.main_layout.addWidget(self.gb_1)
        self.main_layout.addWidget(self.gb_2)
        self.setLayout(self.main_layout)
        self._input_image_path = ""
        self._image_size = (
            (self.gb_2.height() // 100) * 90,
            (self.app_width // 100) * 45,
        )
        self.index = 0
        self.pp_data = {}
        self.load_screen = Loading()
        self.thread_pool = QThreadPool()
        self.feature = None
        self.class_ = None
        self.cls = None
        self.disable()
        self.show()
        
    
    

    def show_constant_image(self):
        # Existing code...
        if os.path.isfile(self._modified_image_path):
            self.add_image(self._modified_image_path, "Coloured Image")
            self.show_metrics()
            self.display_land_type()
    
            # Calculate the total runtime and display it
            elapsed_time = time.time() - self.start_time
            elapsed_time_label = QLabel(f"<b>Total Runtime:</b> {elapsed_time:.2f} seconds")
            elapsed_time_label.setStyleSheet("color: #E0E0E0; font-size: 16px;")
            self.gb_2_v_box.addWidget(elapsed_time_label)  # Add it to the results section
    
        else:
            self.show_message_box(
                "ImageError", QMessageBox.Critical, "Modified image file does not exist."
            )
    

            



    def choose_input(self):
        self.reset()
        filter_ = "JPG Files (*.jpg)"
        self._input_image_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Choose Input Image",
            directory="archive/v_2",
            options=QFileDialog.DontUseNativeDialog,
            filter=filter_,
        )
        
        if os.path.isfile(self._input_image_path):
            self.ip_le.setText(self._input_image_path)
            self.add_image(self._input_image_path, "SAR Image")
            self.ci_pb.setEnabled(False)
            self.pp_btn.setEnabled(True)
        
            # Modify the path for backend use
            modified_path = self._input_image_path.replace('/s1/', '/s2/').replace('_s1_', '_s2_')
        
            # Print the modified path for backend
            print(f"Modified image path for backend: {modified_path}")
        
            # Store the modified path for use in further processing
            self._modified_image_path = modified_path

            # Update the input path with the modified path
            self._input_image_path = self._modified_image_path
        else:
            self.show_message_box(
                "InputImageError", QMessageBox.Critical, "Choose a valid image?"
            )

            


    def preprocess_thread(self):
        worker = Worker(self.preprocess_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.preprocess_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()
        self.pp_btn.setEnabled(False)

    def preprocess_runner(self):
       self.pp_data = preprocess(self._input_image_path, "Input Image")

    def preprocess_finisher(self):
        for k in self.pp_data:
            cv2.imwrite("tmp.jpg", self.pp_data[k])
            self.add_image("tmp.jpg", k)
        self.load_screen.close()
        self.fe_btn.setEnabled(True)

    def fe_thread(self):
        worker = Worker(self.fe_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.fe_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()
        self.fe_btn.setEnabled(False)
    def fe_runner(self):
        nan = get_nasnet_large_model()
        nan_fmm = get_feature_map_model(nan)
        
        # Use the blurred grayscale image path for feature extraction
        if hasattr(self, '_blurred_image_path'):
            image_to_predict_path = self._blurred_image_path
        else:
            image_to_predict_path = "tmp.jpg"  # Fallback if blurred image not available
        
        nan_im = get_image_to_predict(image_to_predict_path)
        self.feature = get_feature(nan_im, nan)
        nan_fm = get_feature_image(nan_im, nan_fmm)
        nan_fm = cv2.resize(nan_fm, SHAPE[:-1])
        nan_fm = cv2.applyColorMap(nan_fm, cmapy.cmap("viridis_r"))
        cv2.imwrite("tmp.jpg", nan_fm)


    def fe_finisher(self):
        self.add_image("tmp.jpg", "Feature Map(Bottle Neck)")
        self.load_screen.close()
        self.colourize_btn.setEnabled(True)

    def classify_thread(self):
        worker = Worker(self.classify_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.classify_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()
        self.pp_btn.setEnabled(False)
        self.fe_btn.setEnabled(False)
        self.colourize_btn.setEnabled(False)

    def classify_runner(self):
        model = build_ssae()
        self.class_ = model.predict(self.feature)
        self.cls = CLASSES[np.argmax(self.class_)]

    def classify_finisher(self):
        self.add_message(self.cls)
        self.load_screen.close()
        self.ci_pb.setEnabled(True)
        self.pp_btn.setEnabled(True)
        self.fe_btn.setEnabled(True)
        self.colourize_btn.setEnabled(True)

    def reset(self):
        self.ip_le.setText("")
        self._input_image_path = ""
        self.disable()
        for i in reversed(range(self.hbox_2.count())):
            widget = self.hbox_2.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def disable(self):
        self.ci_pb.setEnabled(True)
        self.pp_btn.setEnabled(False)
        self.fe_btn.setEnabled(False)
        self.colourize_btn.setEnabled(False)

    def add_image(self, path, title):
        if os.path.isfile(path):
            # Create a QVBoxLayout for each image and its label
            img_layout = QVBoxLayout()
            
            # Load the image and add it to a QLabel
            img = QImage(path)
            lbl_img = QLabel()
            lbl_img.setPixmap(QPixmap.fromImage(img).scaled(self._image_size[1], self._image_size[0], Qt.KeepAspectRatio))
            lbl_img.setAlignment(Qt.AlignCenter)
            
            # Add the image QLabel to the layout
            img_layout.addWidget(lbl_img)
            
            # Create a QLabel to show the button's title
            lbl_title = QLabel(title)
            lbl_title.setAlignment(Qt.AlignCenter)
            
            # Set font style for the title
            lbl_title.setFont(QFont("JetBrains Mono", 10, QFont.Bold))
            lbl_title.setStyleSheet("color: #E0E0E0;")
            
            # Add the title QLabel to the layout
            img_layout.addWidget(lbl_title)
            
            # Create a QWidget to hold the layout and add it to the horizontal layout
            img_widget = QWidget()
            img_widget.setLayout(img_layout)
            self.hbox_2.addWidget(img_widget)
            self.hbox_2.addStretch()  # Optional: Add spacing between images


    def show_message_box(self, title, icon, text):
        msg = QMessageBox()
        msg.setIcon(icon)
        msg.setText(text)
        msg.setWindowTitle(title)
        msg.exec_()

class Loading(QDialog):
    def __init__(self):
        super(Loading, self).__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(200, 100)
        self.setModal(True)
        self.progress = QProgressBar(self)
        self.progress.setGeometry(20, 20, 160, 20)
        self.progress.setRange(0, 0)
        self.progress.setValue(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainGUI()
    sys.exit(app.exec_())  
