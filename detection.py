import os
import cv2
from pathlib import Path
from csv_parser import CSVParser
from config import Config


class FaceDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or Config.BEST_MODEL_PATH
        self.model = None
        
    def load_model(self, model_path=None):
        from ultralytics import YOLO
        
        if model_path:
            self.model_path = model_path
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"📦 Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("✓ Model loaded successfully")
        return self.model
    
    def detect_faces(self, image_path, conf_threshold=None, save_result=True):
        conf_threshold = conf_threshold or Config.DEFAULT_CONF_THRESHOLD
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        results = self.model(image_path, conf=conf_threshold, verbose=False)
        
        boxes = results[0].boxes
        detections = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf)
            })
        
        if save_result:
            img = cv2.imread(image_path)
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Face {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
            
            img_name = Path(image_path).name
            output_path = os.path.join(Config.OUTPUT_DIR, f"detected_{img_name}")
            cv2.imwrite(output_path, img)
        
        return detections
    
    def detect_batch(self, image_folder, conf_threshold=None, pattern='*.jpg'):
        conf_threshold = conf_threshold or Config.DEFAULT_CONF_THRESHOLD
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        image_paths = list(Path(image_folder).glob(pattern))
        
        if not image_paths:
            print(f"⚠️ No images found in {image_folder} with pattern {pattern}")
            return {}
        
        print(f"\n🔍 Processing {len(image_paths)} images...")
        
        all_detections = {}
        
        for img_path in image_paths:
            img_name = img_path.name
            try:
                detections = self.detect_faces(str(img_path), conf_threshold, save_result=True)
                all_detections[img_name] = detections
                print(f"  ✓ {img_name}: {len(detections)} faces detected")
            except Exception as e:
                print(f"  ❌ {img_name}: Error - {str(e)}")
                all_detections[img_name] = []
        
        return all_detections


class ModelEvaluator:
    def __init__(self, detector):
        self.detector = detector
        self.ground_truth = None
    
    def load_ground_truth(self):
        print(f"\n📊 Loading ground truth from: {Config.TEST_ANNOTATIONS}")
        parser = CSVParser(Config.TEST_ANNOTATIONS, Config.TEST_IMAGES_DIR)
        annotations = parser.parse()
        
        self.ground_truth = {}
        for ann in annotations:
            img_name = ann['image_name']
            self.ground_truth[img_name] = {
                'bboxes': ann['bboxes'],
                'count': len(ann['bboxes'])
            }
        
        print(f"✓ Loaded {len(self.ground_truth)} ground truth images")
        return self.ground_truth
    
    def calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_min_x = max(x1_min, x2_min)
        inter_min_y = max(y1_min, y2_min)
        inter_max_x = min(x1_max, x2_max)
        inter_max_y = min(y1_max, y2_max)
        
        if inter_max_x < inter_min_x or inter_max_y < inter_min_y:
            return 0.0
        
        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def evaluate(self, conf_threshold=None, iou_threshold=None):
        conf_threshold = conf_threshold or Config.DEFAULT_CONF_THRESHOLD
        iou_threshold = iou_threshold or Config.DEFAULT_IOU_THRESHOLD
        
        print("📈 MODEL EVALUATION\n")
        
        if self.ground_truth is None:
            self.load_ground_truth()
        
        print(f"\n🔍 Running detection (conf={conf_threshold}, IoU={iou_threshold})...")
        detections = self.detector.detect_batch(
            Config.TEST_IMAGES_DIR,
            conf_threshold=conf_threshold
        )
        
        print(f"\n📊 Calculating metrics...")
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_gt_faces = 0
        total_det_faces = 0
        
        for img_name in self.ground_truth:
            gt_boxes = self.ground_truth[img_name]['bboxes']
            det_boxes = [d['bbox'] for d in detections.get(img_name, [])]
            
            total_gt_faces += len(gt_boxes)
            total_det_faces += len(det_boxes)
            
            matched_gt = set()
            matched_det = set()
            
            # Match detections to ground truth
            for i, det_box in enumerate(det_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt_box in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    
                    iou = self.calculate_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_threshold:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                    matched_det.add(i)
                else:
                    false_positives += 1
            
            # Unmatched ground truth boxes are false negatives
            false_negatives += len(gt_boxes) - len(matched_gt)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("✅ EVALUATION RESULTS\n")
        print(f"\n📊 Detection Statistics:")
        print(f"  Total ground truth faces: {total_gt_faces}")
        print(f"  Total detected faces: {total_det_faces}")
        print(f"  True Positives: {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  False Negatives: {false_negatives}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"  Precision: {precision:.4f} ({precision * 100:.2f}%)")
        print(f"  Recall: {recall:.4f} ({recall * 100:.2f}%)")
        print(f"  F1-Score: {f1_score:.4f} ({f1_score * 100:.2f}%)")
        
        print(f"\n💾 Annotated images saved in: {Config.OUTPUT_DIR}/")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_gt_faces': total_gt_faces,
            'total_det_faces': total_det_faces
        }


if __name__ == "__main__":
    detector = FaceDetector()
    detector.load_model()
    
    evaluator = ModelEvaluator(detector)
    evaluator.evaluate()