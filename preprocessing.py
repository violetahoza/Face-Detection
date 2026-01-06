import os
import yaml
import shutil
from pathlib import Path
from csv_parser import CSVParser
from config import Config


class DatasetPreprocessor:
    def __init__(self):
        pass
    
    def validate_dataset(self):
        valid, missing = Config.validate_dataset()
        if not valid:
            raise FileNotFoundError(
                f"Missing required files:\n" + "\n".join(f"  - {p}" for p in missing)
            )
        return True
    
    def convert_to_yolo(self, csv_path, img_dir, split_name):
        print(f"\n📁 Converting {split_name} data from: {csv_path}")
        
        parser = CSVParser(csv_path, img_dir)
        annotations = parser.parse()
        
        images_dir = os.path.join(Config.YOLO_DATASET_DIR, split_name, 'images')
        labels_dir = os.path.join(Config.YOLO_DATASET_DIR, split_name, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        converted = 0
        skipped = 0
        
        for ann in annotations:
            img_name = ann['image_name']
            img_path = ann['image_path']
            
            if not os.path.exists(img_path):
                print(f"  ⚠️ Image not found: {img_name}")
                skipped += 1
                continue
            
            dest_img = os.path.join(images_dir, img_name)
            shutil.copy(img_path, dest_img)
            
            label_path = os.path.join(labels_dir, f"{Path(img_name).stem}.txt")
            
            with open(label_path, 'w') as f:
                for bbox in ann['bboxes']:
                    x0, y0, x1, y1 = bbox
                    img_w, img_h = ann['width'], ann['height']
                    
                    # Convert to YOLO format (normalized center coordinates + width/height)
                    x_center = ((x0 + x1) / 2) / img_w
                    y_center = ((y0 + y1) / 2) / img_h
                    width = (x1 - x0) / img_w
                    height = (y1 - y0) / img_h
                    
                    # class_id=0 for "face"
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            converted += 1
        
        print(f"✓ Converted {converted} images ({skipped} skipped)")
        return converted, skipped
    
    def create_yaml_config(self):
        dataset_config = {
            'path': os.path.abspath(Config.YOLO_DATASET_DIR),
            'train': 'train/images',
            'val': 'val/images',
            'names': {0: 'face'},
            'nc': 1  # number of classes
        }
        
        with open(Config.YOLO_CONFIG_FILE, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✓ Created configuration: {Config.YOLO_CONFIG_FILE}")
        return Config.YOLO_CONFIG_FILE
    
    def prepare_dataset(self):
        print("📦 DATASET PREPROCESSING\n")
        
        try:
            print("\n1️⃣ Validating dataset structure...")
            self.validate_dataset()
            print("✓ Dataset structure valid")
            
            print("\n2️⃣ Converting training data...")
            train_count, train_skipped = self.convert_to_yolo(
                Config.TRAIN_ANNOTATIONS,
                Config.TRAIN_IMAGES_DIR,
                'train'
            )
            
            print("\n3️⃣ Converting validation/test data...")
            val_count, val_skipped = self.convert_to_yolo(
                Config.TEST_ANNOTATIONS,
                Config.TEST_IMAGES_DIR,
                'val'
            )
            
            print("\n4️⃣ Creating dataset configuration...")
            config_path = self.create_yaml_config()
            
            print("✅ PREPROCESSING COMPLETE!\n")
            print(f"  Training images: {train_count}")
            print(f"  Validation images: {val_count}")
            print(f"  Total: {train_count + val_count}")
            print(f"  Configuration: {config_path}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            return False


if __name__ == "__main__":
    preprocessor = DatasetPreprocessor()
    preprocessor.prepare_dataset()