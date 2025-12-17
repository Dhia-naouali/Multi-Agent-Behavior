import gc, joblib
import polars as pl
from .single_features import make_self_features
from .pair_features import make_pair_features
from .skeleton import SELF_BEHAVIORS, PAIR_BEHAVIORS, BODY_PARTS, out_pairs


def load_data(csv_path, self_behaviors=SELF_BEHAVIORS, pair_behaviors=PAIR_BEHAVIORS):
    train_dataframe = pl.read_csv(csv_path)

    train_behavior_dataframe = (
        train_dataframe.filter(pl.col("behaviors_labeled").is_not_null())
        .select(
            pl.col("lab_id"),
            pl.col("video_id"),
            pl.col("behaviors_labeled").map_elements(eval, return_dtype=pl.List(pl.Utf8)).alias("behaviors_labeled_list"),
        )
        .explode("behaviors_labeled_list")
        .rename({"behaviors_labeled_list": "behaviors_labeled_element"})
        .select(
            pl.col("lab_id"),
            pl.col("video_id"),
            pl.col("behaviors_labeled_element").str.split(",").list[0].str.replace_all("'", "").alias("agent"),
            pl.col("behaviors_labeled_element").str.split(",").list[1].str.replace_all("'", "").alias("target"),
            pl.col("behaviors_labeled_element").str.split(",").list[2].str.replace_all("'", "").alias("behavior"),
        )
    )

    x_single_train = train_behavior_dataframe.filter(pl.col("behavior").is_in(self_behaviors))
    x_pair_train = train_behavior_dataframe.filter(pl.col("behavior").is_in(pair_behaviors))
    return x_single_train, x_pair_train



def process_one_video(row, TRAIN_TRACKING_DIR, SELF_FEATURES_DIR, PAIR_FEATURES_DIR):
    """Process a single video to extract self and pair features."""
    lab_id = row["lab_id"]
    video_id = row["video_id"]

    tracking_path = TRAIN_TRACKING_DIR / f"{lab_id}/{video_id}.parquet"
    tracking = pl.read_parquet(tracking_path)

    self_features = make_self_features(metadata=row, tracking=tracking)
    pair_features = make_pair_features(metadata=row, tracking=tracking)

    self_features.write_parquet(SELF_FEATURES_DIR / f"{video_id}.parquet")
    pair_features.write_parquet(PAIR_FEATURES_DIR / f"{video_id}.parquet")

    return video_id



def process_videos(train_df, TRAIN_TRACKING_DIR, SELF_FEATURES_DIR, PAIR_FEATURES_DIR):
    rows = list(train_df.filter(pl.col("behaviors_labeled").is_not_null()).rows(named=True))
    results = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(process_one_video)(row, TRAIN_TRACKING_DIR, SELF_FEATURES_DIR, PAIR_FEATURES_DIR) for row in rows)
    del rows, results
    gc.collect()