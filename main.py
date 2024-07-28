from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read Video
    video_frames = read_video(r"C:\Users\44780\Documents\new_motion_tracking\input_videos\counter_attack.mp4")
    
    #Initialize Tracker
    tracker = Tracker(r"C:\Users\44780\Documents\new_motion_tracking\models\best.pt")

    tracks = tracker.get_object_tracks(video_frames)

    # Save Video
    save_video(video_frames, r"C:\Users\44780\Documents\new_motion_tracking\output_video\output_video.avi")

if __name__ == "__main__":
    main()