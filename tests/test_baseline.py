"""Unit tests for the MediaPipe rule-based finger counting baseline."""

from __future__ import annotations

import pytest

from baselines.mediapipe_finger_counter import count_all_hands, count_hand_fingers


def _build_hand_landmarks(
    thumb_up: bool,
    other_fingers_up: tuple[bool, bool, bool, bool],
    handedness: str = "Right",
) -> list[tuple[int, int]]:
    """Construct synthetic 21 MediaPipe landmarks matching the requested finger state."""
    lms = [(200, 300) for _ in range(21)]

    # Thumb: tip=4, ip=3 (mirrored frame: right hand thumb extends left -> smaller x)
    lms[3] = (200, 260)
    if handedness == "Right":
        lms[4] = (170, 250) if thumb_up else (230, 250)
    else:
        lms[4] = (230, 250) if thumb_up else (170, 250)

    # Index(8/6), Middle(12/10), Ring(16/14), Pinky(20/18)
    for idx, is_up in enumerate(other_fingers_up):
        tip_id = (8, 12, 16, 20)[idx]
        pip_id = tip_id - 2
        x_pos = 180 + idx * 20
        lms[pip_id] = (x_pos, 220)
        lms[tip_id] = (x_pos, 150) if is_up else (x_pos, 260)

    return lms


def test_right_hand_counts() -> None:
    closed = _build_hand_landmarks(False, (False, False, False, False), "Right")
    assert count_hand_fingers(closed, "Right") == 0

    peace = _build_hand_landmarks(False, (True, True, False, False), "Right")
    assert count_hand_fingers(peace, "Right") == 2

    open_hand = _build_hand_landmarks(True, (True, True, True, True), "Right")
    assert count_hand_fingers(open_hand, "Right") == 5


def test_left_hand_thumb_handedness() -> None:
    thumb_only_left = _build_hand_landmarks(True, (False, False, False, False), "Left")
    assert count_hand_fingers(thumb_only_left, "Left") == 1

    folded_thumb_left = _build_hand_landmarks(False, (False, False, False, False), "Left")
    assert count_hand_fingers(folded_thumb_left, "Left") == 0


def test_two_hands_sum() -> None:
    right = _build_hand_landmarks(True, (True, True, True, True), "Right")
    left = _build_hand_landmarks(True, (True, True, False, False), "Left")
    assert count_all_hands([right, left], ["Right", "Left"]) == 8


def test_invalid_landmark_count_raises() -> None:
    with pytest.raises(ValueError):
        count_hand_fingers([(0, 0)] * 10, "Right")
