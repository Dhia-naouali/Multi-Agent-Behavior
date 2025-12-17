from pathlib import Path
from dataclasses import dataclass

@dataclass
class Conf:
    INDEX_COLS = [
        "video_id",
        "agent_mouse_id",
        "target_mouse_id",
        "video_frame",
    ]

    WORKING_DIR = Path("./")
    INPUT_DIR = Path("MABe-mouse-behavior-detection/")
    TRAIN_TRACKING_DIR = INPUT_DIR / "train_tracking"
    TRAIN_ANNOTATION_DIR = INPUT_DIR / "train_annotation"
    TEST_TRACKING_DIR = INPUT_DIR / "test_tracking"

    SELF_FEATURES_DIR = WORKING_DIR / "self_features"
    PAIR_FEATURES_DIR = WORKING_DIR / "pair_features"

    (WORKING_DIR / "self_features").mkdir(exist_ok=True, parents=True)
    (WORKING_DIR / "pair_features").mkdir(exist_ok=True, parents=True)