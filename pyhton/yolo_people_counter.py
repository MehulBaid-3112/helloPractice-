import cv2
import numpy as np
from ultralytics import YOLO
import argparse


def count_people_in_video(
    video_path: str,
    model_name: str = "yolov8n.pt",
    conf_thresh: float = 0.35,
    process_fps: int = 1,
    show: bool = True,
    resize_width: int = 640,
):
    """
    Counts number of people in a video using YOLOv8.

    - Reads video
    - Runs detection on (process_fps) frames per second
    - Counts class 'person' (COCO class 0)
    - Uses median count across frames for stable result
    """

    # Load YOLO model
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not open video: {video.mp4
}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30  # fallback

    video_fps = int(round(video_fps))
    frame_step = max(1, video_fps // process_fps)

    print(f"📽️ Video FPS: {video_fps}")
    print(f"⚙️ Processing at ~{process_fps} FPS (frame_step={frame_step})")
    print(f"🤖 Model: {model_name}, conf_thresh={conf_thresh}")

    frame_idx = 0
    processed = 0
    counts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        processed += 1

        # Resize for speed
        if resize_width is not None:
            h, w = frame.shape[:2]
            scale = resize_width / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (resize_width, new_h))

        # Run YOLO
        results = model(frame, verbose=False)[0]

        person_count = 0
        for box in results.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls == 0 and conf >= conf_thresh:
                person_count += 1

        counts.append(person_count)
        print(f"Frame {processed:03d}: People detected = {person_count}")

        # Optional display
        if show:
            annotated = frame.copy()
            for box in results.boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls == 0 and conf >= conf_thresh:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"person {conf:.2f}",
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            cv2.putText(
                annotated,
                f"People: {person_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            cv2.imshow("YOLO People Counter", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("🛑 Stopped by user (q pressed).")
                break

        frame_idx += 1

    cap.release()
    if show:
        cv2.destroyAllWindows()

    if len(counts) == 0:
        print("⚠️ No frames processed.")
        return 0, 0, 0

    median_count = int(np.median(counts))
    max_count = int(np.max(counts))
    avg_count = float(np.mean(counts))

    print("\n========== FINAL RESULT ==========")
    print(f"Processed frames: {processed}")
    print(f"Median people count: {median_count}")
    print(f"Max people count: {max_count}")
    print(f"Avg people count: {avg_count:.2f}")
    print("==================================\n")

    return median_count, max_count, avg_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 People Counter from Video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to process")
    parser.add_argument("--show", action="store_true", help="Show video window")
    parser.add_argument("--no-show", action="store_true", help="Disable video window")
    parser.add_argument("--width", type=int, default=640, help="Resize width (0 = no resize)")

    args = parser.parse_args()

    show_window = not args.no_show
    resize_width = None if args.width == 0 else args.width

    count_people_in_video(
        video_path=args.video,
        model_name=args.model,
        conf_thresh=args.conf,
        process_fps=args.fps,
        show=show_window,
        resize_width=resize_width,
    )
