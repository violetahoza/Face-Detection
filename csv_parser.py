import csv
import os

class CSVParser:
    def __init__(self, csv_path, img_dir):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.annotations = []
    
    def parse(self):
        image_dict = {}
        
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['image_name']
                img_path = os.path.join(self.img_dir, img_name)
                
                bbox = (
                    int(row['x0']),
                    int(row['y0']),
                    int(row['x1']),
                    int(row['y1'])
                )
                
                if img_name not in image_dict:
                    image_dict[img_name] = {
                        'image_path': img_path,
                        'image_name': img_name,
                        'width': int(row['width']),
                        'height': int(row['height']),
                        'bboxes': []
                    }
                
                image_dict[img_name]['bboxes'].append(bbox)
        
        self.annotations = list(image_dict.values())
        return self.annotations
    
    def get_image_count(self):
        return len(self.annotations)
    
    def get_bbox_count(self):
        return sum(len(ann['bboxes']) for ann in self.annotations)


if __name__ == "__main__":
    parser = CSVParser(
        'data/faces_dataset/train/annotations.csv',
        'data/faces_dataset/train/images'
    )
    annotations = parser.parse()
    
    print(f"Parsed {parser.get_image_count()} images")
    print(f"Total bounding boxes: {parser.get_bbox_count()}")
    
    if annotations:
        print(f"\nFirst image: {annotations[0]['image_name']}")
        print(f"  Dimensions: {annotations[0]['width']}x{annotations[0]['height']}")
        print(f"  Faces: {len(annotations[0]['bboxes'])}")