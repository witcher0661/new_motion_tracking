from ultralytics import YOLO

class Tracker:
    def __ini__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        batch_size=20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1) 
            detections = self.model.predict(frames)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames):

        detections = self.detect_frames(frames)