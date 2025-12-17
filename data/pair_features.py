import polars as pl
import itertools

from data.utils import _scale
from data.skeleton import BODY_PARTS

def add_interaction_features(x, mouse_pair, avail_A, avail_B, fps):
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return x
    
    if hasattr(mouse_pair['A'], 'columns') and hasattr(mouse_pair['A'].columns, 'get_level_values'):
        A_x = mouse_pair['A']['body_center']['x'].to_numpy()
        A_y = mouse_pair['A']['body_center']['y'].to_numpy()
        B_x = mouse_pair['B']['body_center']['x'].to_numpy()
        B_y = mouse_pair['B']['body_center']['y'].to_numpy()
        
        x = x.with_columns([
            pl.Series("_A_x", A_x),
            pl.Series("_A_y", A_y),
            pl.Series("_B_x", B_x),
            pl.Series("_B_y", B_y),
        ])
    else:
        x = x.with_columns([
            pl.col("A_body_center_x").alias("_A_x"),
            pl.col("A_body_center_y").alias("_A_y"),
            pl.col("B_body_center_x").alias("_B_x"),
            pl.col("B_body_center_y").alias("_B_y"),
        ])
    
    rel_x = pl.col("_A_x") - pl.col("_B_x")
    rel_y = pl.col("_A_y") - pl.col("_B_y")
    rel_dist = (rel_x.pow(2) + rel_y.pow(2)).sqrt()
    
    A_vx = pl.col("_A_x").diff()
    A_vy = pl.col("_A_y").diff()
    B_vx = pl.col("_B_x").diff()
    B_vy = pl.col("_B_y").diff()
    
    x = x.with_columns([
        rel_dist.alias("_rel_dist"),
        A_vx.alias("_A_vx"),
        A_vy.alias("_A_vy"),
        B_vx.alias("_B_vx"),
        B_vy.alias("_B_vy"),
    ])
    
    A_lead = (
        (pl.col("_A_vx") * (pl.col("_A_x") - pl.col("_B_x")) + 
         pl.col("_A_vy") * (pl.col("_A_y") - pl.col("_B_y"))) /
        ((pl.col("_A_vx").pow(2) + pl.col("_A_vy").pow(2)).sqrt() * pl.col("_rel_dist") + 1e-6)
    )
    
    B_lead = (
        (pl.col("_B_vx") * (pl.col("_B_x") - pl.col("_A_x")) + 
         pl.col("_B_vy") * (pl.col("_B_y") - pl.col("_A_y"))) /
        ((pl.col("_B_vx").pow(2) + pl.col("_B_vy").pow(2)).sqrt() * pl.col("_rel_dist") + 1e-6)
    )
    
    x = x.with_columns([
        A_lead.alias("_A_lead"),
        B_lead.alias("_B_lead"),
    ])
    
    exprs = []
    for window in [30, 60]:
        ws = _scale(window, fps)
        exprs.append(
            pl.col("_A_lead").rolling_mean(ws, min_samples=max(1, ws // 6)).alias(f'A_ld{window}')
        )
        exprs.append(
            pl.col("_B_lead").rolling_mean(ws, min_samples=max(1, ws // 6)).alias(f'B_ld{window}')
        )
    
    x = x.with_columns(exprs)
    
    approach = -pl.col("_rel_dist").diff()
    chase = approach * pl.col("_B_lead")
    
    w = 30
    ws = _scale(w, fps)
    x = x.with_columns([
        chase.rolling_mean(ws, min_samples=max(1, ws // 6)).alias(f'chase_{w}')
    ])
    
    A_sp = (pl.col("_A_vx").pow(2) + pl.col("_A_vy").pow(2)).sqrt()
    B_sp = (pl.col("_B_vx").pow(2) + pl.col("_B_vy").pow(2)).sqrt()
    
    x = x.with_columns([
        A_sp.alias("_A_sp"),
        B_sp.alias("_B_sp"),
    ])
    
    corr_exprs = []
    for window in [60, 120]:
        ws = _scale(window, fps)
        mean_A = pl.col("_A_sp").rolling_mean(ws, min_samples=max(1, ws // 6))
        mean_B = pl.col("_B_sp").rolling_mean(ws, min_samples=max(1, ws // 6))
        std_A = pl.col("_A_sp").rolling_std(ws, min_samples=max(1, ws // 6))
        std_B = pl.col("_B_sp").rolling_std(ws, min_samples=max(1, ws // 6))
        
        cov = (
            ((pl.col("_A_sp") - mean_A) * (pl.col("_B_sp") - mean_B))
            .rolling_mean(ws, min_samples=max(1, ws // 6))
        )
        
        corr = (cov / (std_A * std_B + 1e-6)).alias(f'sp_cor{window}')
        corr_exprs.append(corr)
    
    x = x.with_columns(corr_exprs)
    
    return x.drop([
        "_A_x", "_A_y", "_B_x", "_B_y", "_rel_dist",
        "_A_vx", "_A_vy", "_B_vx", "_B_vy",
        "_A_lead", "_B_lead", "_A_sp", "_B_sp"
    ])


def make_pair_features(
    metadata: dict,
    tracking: pl.DataFrame,
) -> pl.DataFrame:
    def body_parts_distance(agent_or_target_1, body_part_1, agent_or_target_2, body_part_2):
        assert agent_or_target_1 == "agent" or agent_or_target_1 == "target"
        assert agent_or_target_2 == "agent" or agent_or_target_2 == "target"
        assert body_part_1 in BODY_PARTS
        assert body_part_2 in BODY_PARTS
        return (
            (pl.col(f"{agent_or_target_1}_x_{body_part_1}") - pl.col(f"{agent_or_target_2}_x_{body_part_2}")).pow(2)
            + (pl.col(f"{agent_or_target_1}_y_{body_part_1}") - pl.col(f"{agent_or_target_2}_y_{body_part_2}")).pow(2)
        ).sqrt() / metadata["pix_per_cm_approx"]

    def body_part_speed(agent_or_target, body_part, period_ms):
        assert agent_or_target == "agent" or agent_or_target == "target"
        assert body_part in BODY_PARTS
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
        return (
            (
                (pl.col(f"{agent_or_target}_x_{body_part}").diff()).pow(2)
                + (pl.col(f"{agent_or_target}_y_{body_part}").diff()).pow(2)
            ).sqrt()
            / metadata["pix_per_cm_approx"]
            * metadata["frames_per_second"]
        ).rolling_mean(window_size=window_frames, center=True, min_samples=1)

    def elongation(agent_or_target):
        assert agent_or_target == "agent" or agent_or_target == "target"
        d1 = body_parts_distance(agent_or_target, "nose", agent_or_target, "tail_base")
        d2 = body_parts_distance(agent_or_target, "ear_left", agent_or_target, "ear_right")
        return d1 / (d2 + 1e-06)

    def body_angle(agent_or_target):
        assert agent_or_target == "agent" or agent_or_target == "target"
        v1x = pl.col(f"{agent_or_target}_x_nose") - pl.col(f"{agent_or_target}_x_body_center")
        v1y = pl.col(f"{agent_or_target}_y_nose") - pl.col(f"{agent_or_target}_y_body_center")
        v2x = pl.col(f"{agent_or_target}_x_tail_base") - pl.col(f"{agent_or_target}_x_body_center")
        v2y = pl.col(f"{agent_or_target}_y_tail_base") - pl.col(f"{agent_or_target}_y_body_center")
        return (v1x * v2x + v1y * v2y) / ((v1x.pow(2) + v1y.pow(2)).sqrt() * (v2x.pow(2) + v2y.pow(2)).sqrt() + 1e-06)


    n_mice = (
        (metadata["mouse1_strain"] is not None)
        + (metadata["mouse2_strain"] is not None)
        + (metadata["mouse3_strain"] is not None)
        + (metadata["mouse4_strain"] is not None)
    )
    start_frame = tracking.select(pl.col("video_frame").min()).item()
    end_frame = tracking.select(pl.col("video_frame").max()).item()

    result = []

    pivot = tracking.pivot(
        on=["bodypart"],
        index=["video_frame", "mouse_id"],
        values=["x", "y"],
    ).sort(["mouse_id", "video_frame"])
    pivot_trackings = {mouse_id: pivot.filter(pl.col("mouse_id") == mouse_id) for mouse_id in range(1, n_mice + 1)}

    for agent_mouse_id, target_mouse_id in itertools.permutations(range(1, n_mice + 1), 2):
        result_element = pl.DataFrame(
            {
                "video_id": metadata["video_id"],
                "agent_mouse_id": agent_mouse_id,
                "target_mouse_id": target_mouse_id,
                "video_frame": pl.arange(start_frame, end_frame + 1, eager=True),
            },
            schema={
                "video_id": pl.Int32,
                "agent_mouse_id": pl.Int8,
                "target_mouse_id": pl.Int8,
                "video_frame": pl.Int32,
            },
        )

        merged_pivot = (
            pivot_trackings[agent_mouse_id]
            .select(
                pl.col("video_frame"),
                pl.exclude("video_frame").name.prefix("agent_"),
            )
            .join(
                pivot_trackings[target_mouse_id].select(
                    pl.col("video_frame"),
                    pl.exclude("video_frame").name.prefix("target_"),
                ),
                on="video_frame",
                how="inner",
            )
        )
        columns = merged_pivot.columns
        merged_pivot = merged_pivot.with_columns(
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_x_{bp}") for bp in BODY_PARTS if f"agent_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_y_{bp}") for bp in BODY_PARTS if f"agent_y_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"target_x_{bp}") for bp in BODY_PARTS if f"target_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"target_y_{bp}") for bp in BODY_PARTS if f"target_y_{bp}" not in columns],
        )


        features = merged_pivot.with_columns(
            pl.lit(agent_mouse_id).alias("agent_mouse_id"),
            pl.lit(target_mouse_id).alias("target_mouse_id"),
        ).select(
            pl.col("video_frame"),
            pl.col("agent_mouse_id"),
            pl.col("target_mouse_id"),
            *[
                body_parts_distance("agent", agent_body_part, "target", target_body_part).alias(
                    f"at__{agent_body_part}__{target_body_part}__distance"
                )
                for agent_body_part, target_body_part in itertools.product(BODY_PARTS, repeat=2)
            ],
            *[
                body_part_speed("agent", body_part, period_ms).alias(f"agent__{body_part}__speed_{period_ms}ms")
                for body_part, period_ms in itertools.product(["ear_left", "ear_right", "tail_base"], [500, 1000, 2000, 3000])
            ],
            *[
                body_part_speed("target", body_part, period_ms).alias(f"target__{body_part}__speed_{period_ms}ms")
                for body_part, period_ms in itertools.product(["ear_left", "ear_right", "tail_base"], [500, 1000, 2000, 3000])
            ],
            elongation("agent").alias("agent__elongation"),
            elongation("target").alias("target__elongation"),
            body_angle("agent").alias("agent__body_angle"),
            body_angle("target").alias("target__body_angle"),
        )

        interaction_df = merged_pivot.rename({
            col: col.replace("agent_x_", "A_x_").replace("agent_y_", "A_y_")
            for col in merged_pivot.columns if col.startswith("agent_")
        }).rename({
            col: col.replace("target_x_", "B_x_").replace("target_y_", "B_y_")
            for col in merged_pivot.columns if col.startswith("target_")
        })
        

        interaction_features = add_interaction_features(
            interaction_df,
            fps=metadata["frames_per_second"],
            A_x='A_x_body_center',
            A_y='A_y_body_center',
            B_x='B_x_body_center',
            B_y='B_y_body_center',
            A_nose_x='A_x_nose',
            A_nose_y='A_y_nose',
            B_nose_x='B_x_nose',
            B_nose_y='B_y_nose',
            A_tail_x='A_x_tail_base',
            A_tail_y='A_y_tail_base',
            B_tail_x='B_x_tail_base',
            B_tail_y='B_y_tail_base'
        )
        

        tracking_cols = [col for col in interaction_features.columns 
                        if col.startswith(('A_x_', 'A_y_', 'B_x_', 'B_y_'))]
        interaction_features_only = interaction_features.drop(tracking_cols)
        

        if 'video_frame' in interaction_features_only.columns:
            interaction_features_only = interaction_features_only.with_columns(
                pl.col('video_frame').cast(pl.Int32)
            )
        

        interaction_features_only = interaction_features_only.rename({
            col: f"pair__{col}" for col in interaction_features_only.columns
            if col != "video_frame"
        })
        

        features = features.join(
            interaction_features_only,
            on="video_frame",
            how="left"
        )


        result_element = result_element.join(
            features,
            on=["video_frame", "agent_mouse_id", "target_mouse_id"],
            how="left",
        )
        result.append(result_element)

    return pl.concat(result, how="vertical")




def add_interaction_features(x, mouse_pair, avail_A, avail_B, fps):

    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return x
    

    if hasattr(mouse_pair['A'], 'columns') and hasattr(mouse_pair['A'].columns, 'get_level_values'):

        A_x = mouse_pair['A']['body_center']['x'].to_numpy()
        A_y = mouse_pair['A']['body_center']['y'].to_numpy()
        B_x = mouse_pair['B']['body_center']['x'].to_numpy()
        B_y = mouse_pair['B']['body_center']['y'].to_numpy()
        
        x = x.with_columns([
            pl.Series("_A_x", A_x),
            pl.Series("_A_y", A_y),
            pl.Series("_B_x", B_x),
            pl.Series("_B_y", B_y),
        ])
    else:
        x = x.with_columns([
            pl.col("A_x_body_center").alias("_A_x"),
            pl.col("A_y_body_center").alias("_A_y"),
            pl.col("B_x_body_center").alias("_B_x"),
            pl.col("B_y_body_center").alias("_B_y"),
        ])
    

    def _scale(window, fps):
        return max(1, int(window * fps / 30))
    

    rel_x = pl.col("_A_x") - pl.col("_B_x")
    rel_y = pl.col("_A_y") - pl.col("_B_y")
    rel_dist = (rel_x.pow(2) + rel_y.pow(2)).sqrt()
    

    A_vx = pl.col("_A_x").diff()
    A_vy = pl.col("_A_y").diff()
    B_vx = pl.col("_B_x").diff()
    B_vy = pl.col("_B_y").diff()
    
    x = x.with_columns([
        rel_dist.alias("_rel_dist"),
        A_vx.alias("_A_vx"),
        A_vy.alias("_A_vy"),
        B_vx.alias("_B_vx"),
        B_vy.alias("_B_vy"),
    ])
    

    A_lead = (
        (pl.col("_A_vx") * (pl.col("_A_x") - pl.col("_B_x")) + 
         pl.col("_A_vy") * (pl.col("_A_y") - pl.col("_B_y"))) /
        ((pl.col("_A_vx").pow(2) + pl.col("_A_vy").pow(2)).sqrt() * pl.col("_rel_dist") + 1e-6)
    )
    
    B_lead = (
        (pl.col("_B_vx") * (pl.col("_B_x") - pl.col("_A_x")) + 
         pl.col("_B_vy") * (pl.col("_B_y") - pl.col("_A_y"))) /
        ((pl.col("_B_vx").pow(2) + pl.col("_B_vy").pow(2)).sqrt() * pl.col("_rel_dist") + 1e-6)
    )
    
    x = x.with_columns([
        A_lead.alias("_A_lead"),
        B_lead.alias("_B_lead"),
    ])
    

    exprs = []
    for window in [30, 60]:
        ws = _scale(window, fps)
        exprs.append(
            pl.col("_A_lead").rolling_mean(ws, min_samples=max(1, ws // 6)).alias(f'A_ld{window}')
        )
        exprs.append(
            pl.col("_B_lead").rolling_mean(ws, min_samples=max(1, ws // 6)).alias(f'B_ld{window}')
        )
    
    x = x.with_columns(exprs)
    

    approach = -pl.col("_rel_dist").diff()
    chase = approach * pl.col("_B_lead")
    
    w = 30
    ws = _scale(w, fps)
    x = x.with_columns([
        chase.rolling_mean(ws, min_samples=max(1, ws // 6)).alias(f'chase_{w}')
    ])
    

    A_sp = (pl.col("_A_vx").pow(2) + pl.col("_A_vy").pow(2)).sqrt()
    B_sp = (pl.col("_B_vx").pow(2) + pl.col("_B_vy").pow(2)).sqrt()
    
    x = x.with_columns([
        A_sp.alias("_A_sp"),
        B_sp.alias("_B_sp"),
    ])
    
    corr_exprs = []
    for window in [60, 120]:
        ws = _scale(window, fps)
        mean_A = pl.col("_A_sp").rolling_mean(ws, min_samples=max(1, ws // 6))
        mean_B = pl.col("_B_sp").rolling_mean(ws, min_samples=max(1, ws // 6))
        std_A = pl.col("_A_sp").rolling_std(ws, min_samples=max(1, ws // 6))
        std_B = pl.col("_B_sp").rolling_std(ws, min_samples=max(1, ws // 6))
        
        cov = (
            ((pl.col("_A_sp") - mean_A) * (pl.col("_B_sp") - mean_B))
            .rolling_mean(ws, min_samples=max(1, ws // 6))
        )
        
        corr = (cov / (std_A * std_B + 1e-6)).alias(f'sp_cor{window}')
        corr_exprs.append(corr)
    
    x = x.with_columns(corr_exprs)
    
    return x.drop([
        "_A_x", "_A_y", "_B_x", "_B_y", "_rel_dist",
        "_A_vx", "_A_vy", "_B_vx", "_B_vy",
        "_A_lead", "_B_lead", "_A_sp", "_B_sp"
    ])


def make_pair_features(
    metadata: dict,
    tracking: pl.DataFrame,
) -> pl.DataFrame:
    def body_parts_distance(agent_or_target_1, body_part_1, agent_or_target_2, body_part_2):
        assert agent_or_target_1 == "agent" or agent_or_target_1 == "target"
        assert agent_or_target_2 == "agent" or agent_or_target_2 == "target"
        assert body_part_1 in BODY_PARTS
        assert body_part_2 in BODY_PARTS
        return (
            (pl.col(f"{agent_or_target_1}_x_{body_part_1}") - pl.col(f"{agent_or_target_2}_x_{body_part_2}")).pow(2)
            + (pl.col(f"{agent_or_target_1}_y_{body_part_1}") - pl.col(f"{agent_or_target_2}_y_{body_part_2}")).pow(2)
        ).sqrt() / metadata["pix_per_cm_approx"]

    def body_part_speed(agent_or_target, body_part, period_ms):
        assert agent_or_target == "agent" or agent_or_target == "target"
        assert body_part in BODY_PARTS
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
        return (
            (
                (pl.col(f"{agent_or_target}_x_{body_part}").diff()).pow(2)
                + (pl.col(f"{agent_or_target}_y_{body_part}").diff()).pow(2)
            ).sqrt()
            / metadata["pix_per_cm_approx"]
            * metadata["frames_per_second"]
        ).rolling_mean(window_size=window_frames, center=True, min_samples=1)

    def elongation(agent_or_target):
        assert agent_or_target == "agent" or agent_or_target == "target"
        d1 = body_parts_distance(agent_or_target, "nose", agent_or_target, "tail_base")
        d2 = body_parts_distance(agent_or_target, "ear_left", agent_or_target, "ear_right")
        return d1 / (d2 + 1e-06)

    def body_angle(agent_or_target):
        assert agent_or_target == "agent" or agent_or_target == "target"
        v1x = pl.col(f"{agent_or_target}_x_nose") - pl.col(f"{agent_or_target}_x_body_center")
        v1y = pl.col(f"{agent_or_target}_y_nose") - pl.col(f"{agent_or_target}_y_body_center")
        v2x = pl.col(f"{agent_or_target}_x_tail_base") - pl.col(f"{agent_or_target}_x_body_center")
        v2y = pl.col(f"{agent_or_target}_y_tail_base") - pl.col(f"{agent_or_target}_y_body_center")
        return (v1x * v2x + v1y * v2y) / ((v1x.pow(2) + v1y.pow(2)).sqrt() * (v2x.pow(2) + v2y.pow(2)).sqrt() + 1e-06)

    n_mice = (
        (metadata["mouse1_strain"] is not None)
        + (metadata["mouse2_strain"] is not None)
        + (metadata["mouse3_strain"] is not None)
        + (metadata["mouse4_strain"] is not None)
    )
    start_frame = tracking.select(pl.col("video_frame").min()).item()
    end_frame = tracking.select(pl.col("video_frame").max()).item()

    result = []

    pivot = tracking.pivot(
        on=["bodypart"],
        index=["video_frame", "mouse_id"],
        values=["x", "y"],
    ).sort(["mouse_id", "video_frame"])
    pivot_trackings = {mouse_id: pivot.filter(pl.col("mouse_id") == mouse_id) for mouse_id in range(1, n_mice + 1)}

    for agent_mouse_id, target_mouse_id in itertools.permutations(range(1, n_mice + 1), 2):
        result_element = pl.DataFrame(
            {
                "video_id": metadata["video_id"],
                "agent_mouse_id": agent_mouse_id,
                "target_mouse_id": target_mouse_id,
                "video_frame": pl.arange(start_frame, end_frame + 1, eager=True),
            },
            schema={
                "video_id": pl.Int32,
                "agent_mouse_id": pl.Int8,
                "target_mouse_id": pl.Int8,
                "video_frame": pl.Int32,
            },
        )

        merged_pivot = (
            pivot_trackings[agent_mouse_id]
            .select(
                pl.col("video_frame"),
                pl.exclude("video_frame").name.prefix("agent_"),
            )
            .join(
                pivot_trackings[target_mouse_id].select(
                    pl.col("video_frame"),
                    pl.exclude("video_frame").name.prefix("target_"),
                ),
                on="video_frame",
                how="inner",
            )
        )
        columns = merged_pivot.columns
        merged_pivot = merged_pivot.with_columns(
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_x_{bp}") for bp in BODY_PARTS if f"agent_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_y_{bp}") for bp in BODY_PARTS if f"agent_y_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"target_x_{bp}") for bp in BODY_PARTS if f"target_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"target_y_{bp}") for bp in BODY_PARTS if f"target_y_{bp}" not in columns],
        )

        features = merged_pivot.with_columns(
            pl.lit(agent_mouse_id).alias("agent_mouse_id"),
            pl.lit(target_mouse_id).alias("target_mouse_id"),
        ).select(
            pl.col("video_frame"),
            pl.col("agent_mouse_id"),
            pl.col("target_mouse_id"),
            *[
                body_parts_distance("agent", agent_body_part, "target", target_body_part).alias(
                    f"at__{agent_body_part}__{target_body_part}__distance"
                )
                for agent_body_part, target_body_part in itertools.product(BODY_PARTS, repeat=2)
            ],
            *[
                body_part_speed("agent", body_part, period_ms).alias(f"agent__{body_part}__speed_{period_ms}ms")
                for body_part, period_ms in itertools.product(["ear_left", "ear_right", "tail_base"], [500, 1000, 2000, 3000])
            ],
            *[
                body_part_speed("target", body_part, period_ms).alias(f"target__{body_part}__speed_{period_ms}ms")
                for body_part, period_ms in itertools.product(["ear_left", "ear_right", "tail_base"], [500, 1000, 2000, 3000])
            ],
            elongation("agent").alias("agent__elongation"),
            elongation("target").alias("target__elongation"),
            body_angle("agent").alias("agent__body_angle"),
            body_angle("target").alias("target__body_angle"),
        )


        interaction_df = merged_pivot.rename({
            col: col.replace("agent_x_", "A_x_").replace("agent_y_", "A_y_")
            for col in merged_pivot.columns if col.startswith("agent_")
        }).rename({
            col: col.replace("target_x_", "B_x_").replace("target_y_", "B_y_")
            for col in merged_pivot.columns if col.startswith("target_")
        })
        
        avail_A = [bp for bp in BODY_PARTS if f"A_x_{bp}" in interaction_df.columns]
        avail_B = [bp for bp in BODY_PARTS if f"B_x_{bp}" in interaction_df.columns]
        
        mouse_pair = {'A': None, 'B': None}
        
        interaction_features = add_interaction_features(
            x=interaction_df,
            mouse_pair=mouse_pair,
            avail_A=avail_A,
            avail_B=avail_B,
            fps=metadata["frames_per_second"]
        )
        
        tracking_cols = [col for col in interaction_features.columns 
                        if col.startswith(('A_x_', 'A_y_', 'B_x_', 'B_y_'))]
        interaction_features_only = interaction_features.drop(tracking_cols)
        
        if 'video_frame' in interaction_features_only.columns:
            interaction_features_only = interaction_features_only.with_columns(
                pl.col('video_frame').cast(pl.Int32)
            )
        
        interaction_features_only = interaction_features_only.rename({
            col: f"pair__{col}" for col in interaction_features_only.columns
            if col != "video_frame"
        })
        
        features = features.join(
            interaction_features_only,
            on="video_frame",
            how="left"
        )


        result_element = result_element.join(
            features,
            on=["video_frame", "agent_mouse_id", "target_mouse_id"],
            how="left",
        )
        result.append(result_element)

    return pl.concat(result, how="vertical")