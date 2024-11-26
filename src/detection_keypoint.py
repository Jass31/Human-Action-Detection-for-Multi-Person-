import sys
import cv2
import numpy as np
from pydantic import BaseModel
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16

class DetectPerson:
    def __init__(self, detection_model='/Users/jashtandel/Downloads/yolo11n.pt'):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.detection_model = detection_model
        self.__load_model()

    def __load_model(self):
        self.model = YOLO(model=self.detection_model).to(self.device)

    def __call__(self, image: np.ndarray) -> Results:
        results = self.model.predict(image, conf=0.25, classes=[0])[0]  # class 0 is person
        return results

class DetectKeypoint:
    def __init__(self, pose_model='/Users/jashtandel/SEM6/HAAD/yolo11x-pose.pt'):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.pose_model = pose_model
        self.get_keypoint = GetKeypoint()
        self.__load_model()

    def __load_model(self):
        self.model = YOLO(model=self.pose_model).to(self.device)

    def extract_keypoint(self, keypoint: np.ndarray) -> dict:
        extracted = {}
        for name, idx in self.get_keypoint.__fields__.items():
            idx_value = idx.default
            if idx_value < len(keypoint):
                extracted[name.lower()] = {
                    'xy': keypoint[idx_value][:2],
                    'confidence': keypoint[idx_value][2] if len(keypoint[idx_value]) > 2 else None
                }
        return extracted

    def get_all_keypoints(self, results: Results) -> list:
        all_keypoints = []
        if results.keypoints is not None:
            keypoints_data = results.keypoints.data.cpu().numpy() if torch.is_tensor(results.keypoints.data) else results.keypoints.data
            
            for person_keypoints in keypoints_data:
                keypoint_data = self.extract_keypoint(person_keypoints)
                all_keypoints.append(keypoint_data)
        return all_keypoints

    def get_xy_keypoint(self, results: Results) -> np.ndarray:
        all_keypoints = self.get_all_keypoints(results)
        if not all_keypoints:
            return np.array([])
        
        xy_points = []
        for person in all_keypoints:
            person_points = []
            for _, point_data in person.items():
                person_points.extend(point_data['xy'])
            xy_points.append(person_points)
        
        return np.array(xy_points)

    def __call__(self, image: np.ndarray) -> Results:
        results = self.model.predict(image, save=False)[0]
        return results

class AngleFeatureGenerator:
    def __init__(self):
        self.get_keypoint = GetKeypoint()
    
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        if any(point is None or not isinstance(point, np.ndarray) or len(point) != 2 for point in [p1, p2, p3]):
            return None
        
        ba = p1 - p2
        bc = p3 - p2
        
        if np.all(ba == 0) or np.all(bc == 0):
            return None
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return angle

    def calculate_vector_angle(self, vector: np.ndarray) -> float:
        if vector is None or not isinstance(vector, np.ndarray) or len(vector) != 2:
            return None
        
        if np.all(vector == 0):
            return None
            
        angle = np.degrees(np.arctan2(vector[1], vector[0]))
        return angle

    def generate_features(self, keypoints: dict) -> dict:
        features = {}
        
        def get_coords(key):
            point_data = keypoints.get(key.lower(), {})
            coords = point_data.get('xy')
            return np.array(coords) if coords is not None else None

        # Calculate all angles as in the original code
        features['right_arm_angle'] = self.calculate_angle(
            get_coords('RIGHT_SHOULDER'),
            get_coords('RIGHT_ELBOW'),
            get_coords('RIGHT_WRIST')
        )
        
        # ... [same angle calculations as before]
        
        return features

    def process_keypoints(self, keypoints_list: list) -> list:
        processed_features = []
        for keypoints in keypoints_list:
            features = self.generate_features(keypoints)
            features = {k: v if v is not None else 0.0 for k, v in features.items()}
            processed_features.append(features)
        return processed_features

class CombinedDetector:
    def __init__(self, detection_model_path='/Users/jashtandel/Downloads/yolo11n.pt',
                 pose_model_path='/Users/jashtandel/SEM6/HAAD/yolo11x-pose.pt'):
        self.person_detector = DetectPerson(detection_model_path)
        self.keypoint_detector = DetectKeypoint(pose_model_path)
        self.feature_generator = AngleFeatureGenerator()

    def process_frame(self, frame: np.ndarray) -> tuple:
        # Detect persons
        person_results = self.person_detector(frame)
        person_boxes = person_results.boxes
        
        # Detect poses
        pose_results = self.keypoint_detector(frame)
        keypoints_list = self.keypoint_detector.get_all_keypoints(pose_results)
        
        # Generate angle features
        angle_features = self.feature_generator.process_keypoints(keypoints_list)
        
        return person_boxes, keypoints_list, angle_features

def visualize_results(image: np.ndarray, person_boxes, keypoints_list: list, angle_features: list) -> np.ndarray:
    """Visualize detected persons, poses, and angles on the image"""
    img_copy = image.copy()
    
    # Draw person detection boxes
    for box in person_boxes.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, f'Person: {conf:.2f}', 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw keypoints and connections
    for keypoints in keypoints_list:
        for name, point_data in keypoints.items():
            point = tuple(map(int, point_data['xy']))
            cv2.circle(img_copy, point, 3, (255, 0, 0), -1)
            
            # Draw connections between keypoints
            connections = [
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_elbow'),
                ('right_shoulder', 'right_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_hip', 'left_knee'),
                ('right_hip', 'right_knee'),
                ('left_knee', 'left_ankle'),
                ('right_knee', 'right_ankle')
            ]
            
            for start_point, end_point in connections:
                if start_point in keypoints and end_point in keypoints:
                    pt1 = tuple(map(int, keypoints[start_point]['xy']))
                    pt2 = tuple(map(int, keypoints[end_point]['xy']))
                    cv2.line(img_copy, pt1, pt2, (0, 255, 255), 2)

    # Draw angle features
    y_offset = 30
    for i, features in enumerate(angle_features):
        for name, value in features.items():
            if value is not None:
                text = f"Person {i+1} {name}: {value:.1f}°"
                cv2.putText(img_copy, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                y_offset += 20

    return img_copy

if __name__ == "__main__":
    # Initialize detector
    detector = CombinedDetector()
    
    # Process video or image
    video_path = '/Users/jashtandel/Downloads/PETS09-S2L1-raw.mp4'
    cap = cv2.VideoCapture(video_path)
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Process frame
            person_boxes, keypoints_list, angle_features = detector.process_frame(frame)
            
            # Visualize results
            annotated_frame = visualize_results(frame, person_boxes, keypoints_list, angle_features)
            
            # Display frame
            cv2.imshow('Combined Detection and Pose Estimation', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()