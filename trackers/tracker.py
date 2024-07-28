from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __ini__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size=20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1) 
            detections = self.model.predict(frames)
            detections += detections_batch
            break
        return detections
    
    def get_object_tracks(self, frames):

        detections = self.detect_frames(frames)

        for frames_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Convert to Supervision
            detection_supervision = sv.Detections.from_ultralytics(detection)

            print(detection_supervision)