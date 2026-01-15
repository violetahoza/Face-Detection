import os
import cv2
from pathlib import Path
from csv_parser import CSVParser
from config import Config
import json
from datetime import datetime

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
                print(f"  ✗ {img_name}: Error - {str(e)}")
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
    
    def evaluate(self, conf_threshold=None, iou_threshold=None, max_images=None, save_results=True):
        conf_threshold = conf_threshold or Config.DEFAULT_CONF_THRESHOLD
        iou_threshold = iou_threshold or Config.DEFAULT_IOU_THRESHOLD
        
        print("📈 MODEL EVALUATION\n")
        
        if self.ground_truth is None:
            self.load_ground_truth()
        
        image_names = list(self.ground_truth.keys())
        image_names.sort()
        if max_images and max_images < len(image_names):
            image_names = image_names[:max_images]
            print(f"ℹ️  Evaluating first {max_images} images out of {len(self.ground_truth)}")
        
        print(f"\n🔍 Running detection (conf={conf_threshold}, IoU={iou_threshold})...")
        
        detections = {}
        for img_name in image_names:
            img_path = os.path.join(Config.TEST_IMAGES_DIR, img_name)
            if os.path.exists(img_path):
                try:
                    det = self.detector.detect_faces(img_path, conf_threshold, save_result=True)
                    detections[img_name] = det
                    print(f"  ✓ {img_name}: {len(det)} faces detected")
                except Exception as e:
                    print(f"  ✗ {img_name}: Error - {str(e)}")
                    detections[img_name] = []
        
        print(f"\n📊 Calculating metrics...")
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_gt_faces = 0
        total_det_faces = 0
        
        per_image_results = []
        
        for img_name in image_names:
            if img_name not in self.ground_truth:
                continue
                
            gt_boxes = self.ground_truth[img_name]['bboxes']
            det_boxes = [d['bbox'] for d in detections.get(img_name, [])]
            
            total_gt_faces += len(gt_boxes)
            total_det_faces += len(det_boxes)
            
            matched_gt = set()
            matched_det = set()
            
            img_tp = 0
            img_fp = 0
            
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
                    img_tp += 1
                    matched_gt.add(best_gt_idx)
                    matched_det.add(i)
                else:
                    false_positives += 1
                    img_fp += 1
            
            img_fn = len(gt_boxes) - len(matched_gt)
            false_negatives += img_fn
            
            per_image_results.append({
                'image_name': img_name,
                'ground_truth_faces': len(gt_boxes),
                'detected_faces': len(det_boxes),
                'true_positives': img_tp,
                'false_positives': img_fp,
                'false_negatives': img_fn
            })
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'confidence_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'max_images': max_images,
                'images_evaluated': len(image_names)
            },
            'metrics': {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives),
                'total_gt_faces': int(total_gt_faces),
                'total_det_faces': int(total_det_faces)
            },
            'per_image_results': per_image_results
        }
        
        if save_results:
            eval_dir = os.path.join(Config.OUTPUT_DIR, 'evaluation')
            os.makedirs(eval_dir, exist_ok=True)
            
            json_path = os.path.join(eval_dir, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            report_path = os.path.join(eval_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(report_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("FACE DETECTION MODEL EVALUATION REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Timestamp: {results['timestamp']}\n")
                f.write(f"Images Evaluated: {len(image_names)}\n")
                f.write(f"Confidence Threshold: {conf_threshold}\n")
                f.write(f"IoU Threshold: {iou_threshold}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write("OVERALL METRICS\n")
                f.write("-" * 60 + "\n")
                f.write(f"Precision:         {precision:.4f} ({precision * 100:.2f}%)\n")
                f.write(f"Recall:            {recall:.4f} ({recall * 100:.2f}%)\n")
                f.write(f"F1-Score:          {f1_score:.4f} ({f1_score * 100:.2f}%)\n\n")
                
                f.write("-" * 60 + "\n")
                f.write("DETECTION STATISTICS\n")
                f.write("-" * 60 + "\n")
                f.write(f"Ground Truth Faces:    {total_gt_faces}\n")
                f.write(f"Detected Faces:        {total_det_faces}\n")
                f.write(f"True Positives:        {true_positives}\n")
                f.write(f"False Positives:       {false_positives}\n")
                f.write(f"False Negatives:       {false_negatives}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write("PER-IMAGE RESULTS (First 20)\n")
                f.write("-" * 60 + "\n")
                for result in per_image_results[:20]:
                    f.write(f"\n{result['image_name']}:\n")
                    f.write(f"  GT Faces: {result['ground_truth_faces']}, ")
                    f.write(f"Detected: {result['detected_faces']}, ")
                    f.write(f"TP: {result['true_positives']}, ")
                    f.write(f"FP: {result['false_positives']}, ")
                    f.write(f"FN: {result['false_negatives']}\n")
                
                if len(per_image_results) > 20:
                    f.write(f"\n... and {len(per_image_results) - 20} more images\n")
            
            print(f"\n💾 Results saved:")
            print(f"  JSON: {json_path}")
            print(f"  Report: {report_path}")
        
        print("\n" + "=" * 60)
        print("✅ EVALUATION RESULTS")
        print("=" * 60)
        print(f"\n📊 Detection Statistics:")
        print(f"  Images evaluated: {len(image_names)}")
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
        print("=" * 60 + "\n")
        
        return results


if __name__ == "__main__":
    detector = FaceDetector()
    detector.load_model()
    
    evaluator = ModelEvaluator(detector)
    evaluator.evaluate(max_images=100)