import polars as pl

PAIR_BEHAVIORS = ...
SELF_BEHAVIORS = ...



SINGLE_BEHAVIORS = [
    "biteobject", "climb", "dig", "exploreobject", "freeze",
    "genitalgroom", "huddle", "rear", "rest", "run", "selfgroom",
]

PAIR_BEHAVIORS = [
    "allogroom", "approach", "attack", "attemptmount", "avoid", "chase",
    "chaseattack", "defend", "disengage", "dominance", "dominancegroom",
    "dominancemount", "ejaculate", "escape", "flinch", "follow", "intromit",
    "mount", "reciprocalsniff", "shepherd", "sniff", "sniffbody", "sniffface",
    "sniffgenital", "submit", "tussle",
]

def load_data(csv_path, single_behaviors, pair_behaviors=PAIR_BEHAVIORS):
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

    x_single_train = train_behavior_dataframe.filter(pl.col("behavior").is_in(single_behaviors))
    x_pair_train = train_behavior_dataframe.filter(pl.col("behavior").is_in(pair_behaviors))
    return x_single_train, x_pair_train