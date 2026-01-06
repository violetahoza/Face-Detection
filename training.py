import os
from config import Config

class YOLOTrainer:
    def __init__(self):
        self.model = None
        self.results = None
    
    def validate_config(self):
        if not os.path.exists(Config.YOLO_CONFIG_FILE):
            raise FileNotFoundError(
                f"Dataset configuration not found: {Config.YOLO_CONFIG_FILE}\n"
                "Run preprocessing first!"
            )
        return True
    
    def train(self, epochs=None, batch_size=None, img_size=None, patience=None, verbose=True):
        from ultralytics import YOLO
        
        epochs = epochs or Config.DEFAULT_EPOCHS
        batch_size = batch_size or Config.DEFAULT_BATCH_SIZE
        img_size = img_size or Config.DEFAULT_IMAGE_SIZE
        patience = patience or Config.DEFAULT_PATIENCE
        
        if verbose:
            print("🚀 MODEL TRAINING\n")
        
        try:
            self.validate_config()
        except FileNotFoundError as e:
            if verbose:
                print(f"\n❌ {str(e)}")
            return False
        
        if verbose:
            print(f"\n⚙️ Training Configuration:")
            print(f"  Model: {Config.YOLO_MODEL}")
            print(f"  Epochs: {epochs}")
            print(f"  Image size: {img_size}")
            print(f"  Batch size: {batch_size}")
            print(f"  Patience: {patience} epochs")
            print(f"  Device: {Config.DEVICE}")
            print(f"  Dataset: {Config.YOLO_CONFIG_FILE}")
            
            
            print("🏋️ TRAINING STARTED...\n")
            print("💡 Press Ctrl+C to stop (progress will be saved)\n")
        
        try:
            self.model = YOLO(Config.YOLO_MODEL)
            
            self.results = self.model.train(
                data=Config.YOLO_CONFIG_FILE,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                patience=patience,
                save=True,
                plots=True,
                device=Config.DEVICE,
                project=Config.TRAINING_DIR,
                name=Config.MODEL_NAME,
                exist_ok=True,
                verbose=verbose
            )
            
            if verbose:
                print("✅ TRAINING COMPLETE!\n")
                print(f"\n📊 Results:")
                print(f"  Best model: {Config.BEST_MODEL_PATH}")
                print(f"  Metrics: {Config.TRAINING_DIR}/{Config.MODEL_NAME}/results.csv")
                print(f"  Plots: {Config.TRAINING_DIR}/{Config.MODEL_NAME}/*.png")
            
            return True
            
        except KeyboardInterrupt:
            if verbose:
                print("\n\n⚠️ Training interrupted by user\n")
                print("Progress saved. You can resume or use the last checkpoint.\n")
            return False
        except Exception as e:
            if verbose:
                print(f"\n❌ Training error: {str(e)}")
            return False
    
    def load_model(self, model_path=None):
        from ultralytics import YOLO
        
        model_path = model_path or Config.BEST_MODEL_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"📦 Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("✓ Model loaded successfully")
        return self.model


if __name__ == "__main__":
    trainer = YOLOTrainer()
    trainer.train()