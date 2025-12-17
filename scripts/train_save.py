import gc
import re
import time
import datetime
import polars as pl
from tqdm import tqdm

from data import load_data, process_videos
from utils import train_validate, Conf

conf = Conf()

train_df, x_self_train, x_pair_train = load_data(conf.INPUT_DIR / "train.csv")
process_videos(train_df, conf.TRAIN_TRACKING_DIR, conf.SELF_FEATURES_DIR, conf.PAIR_FEATURES_DIR)

for action_cat, x in zip(["self", "pair"], [x_self_train, x_pair_train]):
    groups = x.group_by("lab_id", "behavior", maintain_order=True)
    total_groups = len(list(groups))
    start_time = time.perf_counter()

    for idx, ((lab_id, behavior), group) in tqdm(enumerate(groups), total=total_groups):
        if idx == 0:
            tqdm.write(
                f"|{'LAB':^25}|{'BEHAVIOR':^15}|{'SAMPLES':^10}|{'POSITIVE':^10}|{'FEATURES':^10}|{'F1':^10}|{'ELAPSED TIME':^15}|",
                end="\n",
            )

        tqdm.write(f"|{lab_id:^25}|{behavior:^15}|", end="")
        index_list = []
        feature_list = []
        label_list = []

        for row in group.rows(named=True):
            video_id = row["video_id"]
            agent = row["agent"]

            agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))

            data = pl.scan_parquet(conf.WORKING_DIR / f"{action_cat}_features" / f"{video_id}.parquet").filter(
                (pl.col("agent_mouse_id") == agent_mouse_id)
            )
            index = data.select(conf.INDEX_COLS).collect(engine="streaming")
            feature = data.select(pl.exclude(conf.INDEX_COLS)).collect(engine="streaming")

            annotation_path = conf.TRAIN_ANNOTATION_DIR / lab_id / f"{video_id}.parquet"
            if annotation_path.exists():
                annotation = (
                    pl.scan_parquet(annotation_path)
                    .filter((pl.col("action") == behavior) & (pl.col("agent_id") == agent_mouse_id))
                    .collect()
                )
            else:
                annotation = pl.DataFrame(
                    schema={
                        "agent_id": pl.Int8,
                        "target_id": pl.Int8,
                        "action": str,
                        "start_frame": pl.Int16,
                        "stop_frame": pl.Int16,
                    }
                )

            label_frames = set()
            for annotation_row in annotation.rows(named=True):
                label_frames.update(range(annotation_row["start_frame"], annotation_row["stop_frame"]))
            label = index.select(pl.col("video_frame").is_in(label_frames).cast(pl.Int8).alias("label"))

            if label.get_column("label").sum() == 0:
                continue

            index_list.append(index)
            feature_list.append(feature)
            label_list.append(label.get_column("label"))

        if not index_list:
            elapsed_time = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
            tqdm.write(f"{0:>10,}|{0:>10,}|{0:>10,}|{'-':>10}|{str(elapsed_time):>15}|", end="\n")
            continue

        indices = pl.concat(index_list, how="vertical")
        features = pl.concat(feature_list, how="vertical")
        labels = pl.concat(label_list, how="vertical")

        del index_list, feature_list, label_list
        gc.collect()

        tqdm.write(f"{len(indices):>10,}|{labels.sum():>10,}|{len(features.columns):>10,}|", end="")

        f1 = train_validate(lab_id, behavior, indices, features, labels)
        tqdm.write(f"{f1:>10.2f}|", end="")

        elapsed_time = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
        tqdm.write(f"{str(elapsed_time):>15}|", end="\n")

        gc.collect()