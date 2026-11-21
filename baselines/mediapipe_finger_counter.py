"""Rule-based MediaPipe Hands baseline for real-time finger counting (1 or 2 hands).

This baseline contrasts a zero-training landmark heuristic approach against the
fine-tuned YOLO11s object detector served by the main FastAPI application.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

# MediaPipe Hands landmark indices for fingertips (thumb, index, middle, ring, pinky).
FINGER_TIP_IDS = (4, 8, 12, 16, 20)
THUMB_IP_ID = 3


def count_hand_fingers(
    landmarks: Sequence[tuple[int, int]],
    handedness: str = "Right",
) -> int:
    """Count raised fingers for a single hand from 21 (x, y) pixel landmarks.

    Args:
        landmarks: Sequence of 21 ``(x, y)`` pixel coordinates in MediaPipe Hands order.
        handedness: ``"Right"`` or ``"Left"`` as reported by MediaPipe (assuming a
            mirrored webcam view where a right hand's thumb extends toward smaller x).

    Returns:
        Integer count of extended fingers in ``[0, 5]``.
    """
    if len(landmarks) < 21:
        raise ValueError(f"Expected 21 hand landmarks, got {len(landmarks)}.")

    fingers_up = 0

    # Thumb check depends on handedness along the horizontal axis (x).
    thumb_tip_x = landmarks[FINGER_TIP_IDS[0]][0]
    thumb_ip_x = landmarks[THUMB_IP_ID][0]
    if handedness.lower() == "left":
        if thumb_tip_x > thumb_ip_x:
            fingers_up += 1
    else:
        if thumb_tip_x < thumb_ip_x:
            fingers_up += 1

    # Index, middle, ring, pinky: fingertip is above its PIP joint (smaller y in image coords).
    for tip_id in FINGER_TIP_IDS[1:]:
        if landmarks[tip_id][1] < landmarks[tip_id - 2][1]:
            fingers_up += 1

    return fingers_up


def count_all_hands(
    hands_landmarks: Sequence[Sequence[tuple[int, int]]],
    handedness_labels: Sequence[str] | None = None,
) -> int:
    """Sum extended fingers across all detected hands."""
    total = 0
    for idx, hand_lms in enumerate(hands_landmarks):
        label = handedness_labels[idx] if handedness_labels and idx < len(handedness_labels) else "Right"
        total += count_hand_fingers(hand_lms, handedness=label)
    return total


def run_webcam(
    camera_index: int = 0,
    max_hands: int = 2,
    min_detection_conf: float = 0.5,
    min_tracking_conf: float = 0.5,
    mirror: bool = True,
) -> None:
    """Run live webcam finger counting using OpenCV and MediaPipe Hands."""
    import cv2
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}.")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                hands_coords: list[list[tuple[int, int]]] = []
                handedness_labels: list[str] = []

                if results.multi_hand_landmarks:
                    multi_handedness = results.multi_handedness or []
                    for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                        coords = [(int(lm.x * width), int(lm.y * height)) for lm in hand_lm.landmark]
                        hands_coords.append(coords)

                        if idx < len(multi_handedness):
                            label = multi_handedness[idx].classification[0].label
                        else:
                            label = "Right"
                        handedness_labels.append(label)

                total_fingers = count_all_hands(hands_coords, handedness_labels)

                # Overlay counter badge
                cv2.rectangle(frame, (16, 16), (210, 82), (13, 19, 28), cv2.FILLED)
                cv2.putText(
                    frame,
                    f"Fingers: {total_fingers}",
                    (28, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("MediaPipe Finger Counter Baseline (press 'q' to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe Hands rule-based finger counter baseline.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0).")
    parser.add_argument("--max-hands", type=int, default=2, choices=[1, 2], help="Maximum hands to track.")
    parser.add_argument("--min-detection-conf", type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument("--min-tracking-conf", type=float, default=0.5, help="Tracking confidence threshold.")
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal webcam mirroring.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_webcam(
        camera_index=args.camera,
        max_hands=args.max_hands,
        min_detection_conf=args.min_detection_conf,
        min_tracking_conf=args.min_tracking_conf,
        mirror=not args.no_mirror,
    )
