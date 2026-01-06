import os
from pathlib import Path

class Config:
    DATASET_DIR = 'data'
    TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, 'train/images')
    TRAIN_ANNOTATIONS = os.path.join(DATASET_DIR, 'train/annotations.csv')
    TEST_IMAGES_DIR = os.path.join(DATASET_DIR, 'test/images')
    TEST_ANNOTATIONS = os.path.join(DATASET_DIR, 'test/annotations.csv')
    
    YOLO_DATASET_DIR = 'faces_dataset'
    YOLO_CONFIG_FILE = 'face_dataset.yaml'
    
    TRAINING_DIR = 'yolo_training'
    MODEL_NAME = 'face_detector'
    BEST_MODEL_PATH = os.path.join(TRAINING_DIR, MODEL_NAME, 'weights/best.pt')
    LAST_MODEL_PATH = os.path.join(TRAINING_DIR, MODEL_NAME, 'weights/last.pt')
    
    OUTPUT_DIR = 'outputs'
    
    DEFAULT_EPOCHS = 30
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_IMAGE_SIZE = 640
    DEFAULT_PATIENCE = 5
    YOLO_MODEL = 'yolov8n.pt'  
    DEVICE = 'cpu'  
    
    DEFAULT_CONF_THRESHOLD = 0.5
    DEFAULT_IOU_THRESHOLD = 0.5
    
    IMAGE_FORMATS = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    @staticmethod
    def create_directories():
        directories = [
            Config.OUTPUT_DIR,
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def get_model_path():
        return Config.BEST_MODEL_PATH
    
    @staticmethod
    def model_exists():
        return os.path.exists(Config.BEST_MODEL_PATH)
    
    @staticmethod
    def dataset_preprocessed():
        return os.path.exists(Config.YOLO_CONFIG_FILE)
    
    @staticmethod
    def validate_dataset():
        required_paths = [
            Config.TRAIN_IMAGES_DIR,
            Config.TRAIN_ANNOTATIONS,
            Config.TEST_IMAGES_DIR,
            Config.TEST_ANNOTATIONS
        ]
        
        missing = []
        for path in required_paths:
            if not os.path.exists(path):
                missing.append(path)
        
        return len(missing) == 0, missing


Config.create_directories()