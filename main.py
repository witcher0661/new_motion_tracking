from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read Video
    video_frames = read_video(r"C:\Users\44780\Documents\new_motion_tracking\input_videos\counter_attack.mp4")
    
    #Initialize Tracker
    tracker = Tracker(r"C:\Users\44780\Documents\new_motion_tracking\models\best.pt", conf_threshold=0.25, iou_threshold=0.45)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    save_video(output_video_frames, r"C:\Users\44780\Documents\new_motion_tracking\output_video\output_video.avi")

if __name__ == "__main__":
    main()