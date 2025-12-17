import itertools
import polars as pl

from .utils import (
    _scale,
    add_curvature_features, 
    add_multiscale_features, 
    add_state_features, 
    add_longrange_features, 
    add_cumulative_distance_single, 
    add_groom_microfeatures, 
    add_mouth_window_features, 
    _scale_signed,
    BODY_PARTS,
    out_pairs
)


def transform_single(single_mouse, body_parts_tracked, fps):
    available_body_parts = set()
    
    for part in body_parts_tracked:
        if f'x_{part}' in single_mouse.columns and f'y_{part}' in single_mouse.columns:
            available_body_parts.add(part)
    

    X = single_mouse.clone()
    

    distance_exprs = []
    for p1, p2 in itertools.combinations(body_parts_tracked, 2):
        for out_pair in out_pairs:
            out_part1, out_part2 = out_pair
            if (p1 == out_part1 and p2 == out_part2) or (p2 == out_part1 and p1 == out_part2):
                continue
        if p1 in available_body_parts and p2 in available_body_parts:

            dist = (
                (pl.col(f'x_{p1}') - pl.col(f'x_{p2}')).pow(2) +
                (pl.col(f'y_{p1}') - pl.col(f'y_{p2}')).pow(2)
            ).alias(f'{p1}+{p2}')
            distance_exprs.append(dist)
    
    if distance_exprs:
        X = X.with_columns(distance_exprs)
    

    if all(p in available_body_parts for p in ['ear_left', 'ear_right', 'tail_base']):
        lag = _scale(10, fps)
        
        speed_exprs = [
            (
                (pl.col('x_ear_left') - pl.col('x_ear_left').shift(lag)).pow(2) +
                (pl.col('y_ear_left') - pl.col('y_ear_left').shift(lag)).pow(2)
            ).alias('sp_lf'),
            (
                (pl.col('x_ear_right') - pl.col('x_ear_right').shift(lag)).pow(2) +
                (pl.col('y_ear_right') - pl.col('y_ear_right').shift(lag)).pow(2)
            ).alias('sp_rt'),
            (
                (pl.col('x_ear_left') - pl.col('x_tail_base').shift(lag)).pow(2) +
                (pl.col('y_ear_left') - pl.col('y_tail_base').shift(lag)).pow(2)
            ).alias('sp_lf2'),
            (
                (pl.col('x_ear_right') - pl.col('x_tail_base').shift(lag)).pow(2) +
                (pl.col('y_ear_right') - pl.col('y_tail_base').shift(lag)).pow(2)
            ).alias('sp_rt2'),
        ]
        
        X = X.with_columns(speed_exprs)
    

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X = X.with_columns(
            (pl.col('nose+tail_base') / (pl.col('ear_left+ear_right') + 1e-6)).alias('elong')
        )
    

    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1x = pl.col('x_nose') - pl.col('x_body_center')
        v1y = pl.col('y_nose') - pl.col('y_body_center')

        v2x = pl.col('x_tail_base') - pl.col('x_body_center')
        v2y = pl.col('y_tail_base') - pl.col('y_body_center')
        
        body_ang = (
            (v1x * v2x + v1y * v2y) /
            ((v1x.pow(2) + v1y.pow(2)).sqrt() * (v2x.pow(2) + v2y.pow(2)).sqrt() + 1e-6)
        ).alias('body_ang')
        
        X = X.with_columns(body_ang)
    
    if 'body_center' in available_body_parts:
        body_exprs = []
        
        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            
            body_exprs.extend([
                pl.col('x_body_center').rolling_mean(ws, min_samples=1, center=True).alias(f'cx_m{w}'),
                pl.col('y_body_center').rolling_mean(ws, min_samples=1, center=True).alias(f'cy_m{w}'),
                pl.col('x_body_center').rolling_std(ws, min_samples=1, center=True).alias(f'cx_s{w}'),
                pl.col('y_body_center').rolling_std(ws, min_samples=1, center=True).alias(f'cy_s{w}'),
            ])
            
            body_exprs.extend([
                (
                    pl.col('x_body_center').rolling_max(ws, min_samples=1, center=True) -
                    pl.col('x_body_center').rolling_min(ws, min_samples=1, center=True)
                ).alias(f'x_rng{w}'),
                (
                    pl.col('y_body_center').rolling_max(ws, min_samples=1, center=True) -
                    pl.col('y_body_center').rolling_min(ws, min_samples=1, center=True)
                ).alias(f'y_rng{w}'),
            ])
        
        X = X.with_columns(body_exprs)
        
        dx = pl.col('x_body_center').diff()
        dy = pl.col('y_body_center').diff()
        
        X = X.with_columns([dx.alias('_dx'), dy.alias('_dy')])
        
        disp_act_exprs = []
        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            
            disp_act_exprs.append(
                (
                    pl.col('_dx').rolling_sum(ws, min_samples=1).pow(2) +
                    pl.col('_dy').rolling_sum(ws, min_samples=1).pow(2)
                ).sqrt().alias(f'disp{w}')
            )
            
            disp_act_exprs.append(
                (
                    pl.col('_dx').rolling_var(ws, min_samples=1) +
                    pl.col('_dy').rolling_var(ws, min_samples=1)
                ).sqrt().alias(f'act{w}')
            )
        
        X = X.with_columns(disp_act_exprs).drop(['_dx', '_dy'])
        
        X = add_curvature_features(X, 'x_body_center', 'y_body_center', fps)
        X = add_multiscale_features(X, 'x_body_center', 'y_body_center', fps)
        X = add_state_features(X, 'x_body_center', 'y_body_center', fps)
        X = add_longrange_features(X, 'x_body_center', 'y_body_center', fps)
        X = add_cumulative_distance_single(X, 'x_body_center', 'y_body_center', fps)
    
    if all(p in available_body_parts for p in ['nose', 'body_center']):
        X = add_groom_microfeatures(X, fps, 
                                    body_center_x='x_body_center',
                                    body_center_y='y_body_center',
                                    nose_x='x_nose',
                                    nose_y='y_nose',
                                    tail_base_x='x_tail_base' if 'tail_base' in available_body_parts else None,
                                    tail_base_y='y_tail_base' if 'tail_base' in available_body_parts else None)
    
    if 'mouth' in available_body_parts:
        pix_per_cm = 10.0
        X = add_mouth_window_features(X, pix_per_cm, fps, 
                                     mouth_x='x_mouth', mouth_y='y_mouth')
    
    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = (
            (pl.col('x_nose') - pl.col('x_tail_base')).pow(2) +
            (pl.col('y_nose') - pl.col('y_tail_base')).pow(2)
        ).sqrt().alias('_nt_dist')
        
        X = X.with_columns(nt_dist)
        
        nt_exprs = []
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            nt_exprs.extend([
                pl.col('_nt_dist').shift(l).alias(f'nt_lg{lag}'),
                (pl.col('_nt_dist') - pl.col('_nt_dist').shift(l)).alias(f'nt_df{lag}'),
            ])
        
        X = X.with_columns(nt_exprs).drop('_nt_dist')
    
    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_dist = (
            (pl.col('x_ear_left') - pl.col('x_ear_right')).pow(2) +
            (pl.col('y_ear_left') - pl.col('y_ear_right')).pow(2)
        ).sqrt().alias('_ear_d')
        
        X = X.with_columns(ear_dist)
        
        ear_exprs = []
        for off in [-30, -20, -10, 10, 20, 30]:
            o = _scale_signed(off, fps)
            ear_exprs.append(
                pl.col('_ear_d').shift(-o).alias(f'ear_o{off}')
            )
        
        w = _scale(30, fps)
        ear_exprs.append(
            (
                pl.col('_ear_d').rolling_std(w, min_samples=1, center=True) /
                (pl.col('_ear_d').rolling_mean(w, min_samples=1, center=True) + 1e-6)
            ).alias('ear_con')
        )
        
        X = X.with_columns(ear_exprs).drop('_ear_d')
    
    return X.cast(pl.Float32)




def make_self_features(
    metadata: dict,
    tracking: pl.DataFrame,
) -> pl.DataFrame:
    def body_parts_distance(body_part_1, body_part_2):
        assert body_part_1 in BODY_PARTS
        assert body_part_2 in BODY_PARTS
        return (
            (pl.col(f"agent_x_{body_part_1}") - pl.col(f"agent_x_{body_part_2}")).pow(2)
            + (pl.col(f"agent_y_{body_part_1}") - pl.col(f"agent_y_{body_part_2}")).pow(2)
        ).sqrt() / metadata["pix_per_cm_approx"]

    def body_part_speed(body_part, period_ms):
        assert body_part in BODY_PARTS
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
        return (
            ((pl.col(f"agent_x_{body_part}").diff()).pow(2) + (pl.col(f"agent_y_{body_part}").diff()).pow(2)).sqrt()
            / metadata["pix_per_cm_approx"]
            * metadata["frames_per_second"]
        ).rolling_mean(window_size=window_frames, center=True, min_samples=1)

    def elongation():
        d1 = body_parts_distance("nose", "tail_base")
        d2 = body_parts_distance("ear_left", "ear_right")
        return d1 / (d2 + 1e-06)

    def body_angle():
        v1x = pl.col("agent_x_nose") - pl.col("agent_x_body_center")
        v1y = pl.col("agent_y_nose") - pl.col("agent_y_body_center")
        v2x = pl.col("agent_x_tail_base") - pl.col("agent_x_body_center")
        v2y = pl.col("agent_y_tail_base") - pl.col("agent_y_body_center")
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

    for agent_mouse_id in range(1, n_mice + 1):
        result_element = pl.DataFrame(
            {
                "video_id": metadata["video_id"],
                "agent_mouse_id": agent_mouse_id,
                "target_mouse_id": -1,
                "video_frame": pl.arange(start_frame, end_frame + 1, eager=True),
            },
            schema={
                "video_id": pl.Int32,
                "agent_mouse_id": pl.Int8,
                "target_mouse_id": pl.Int8,
                "video_frame": pl.Int32,
            },
        )

        pivot = pivot_trackings[agent_mouse_id].select(
            pl.col("video_frame"),
            pl.exclude("video_frame").name.prefix("agent_"),
        )
        columns = pivot.columns
        pivot = pivot.with_columns(
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_x_{bp}") for bp in BODY_PARTS if f"agent_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_y_{bp}") for bp in BODY_PARTS if f"agent_y_{bp}" not in columns],
        )

        features = pivot.with_columns(
            pl.lit(agent_mouse_id).alias("agent_mouse_id"),
            pl.lit(-1).alias("target_mouse_id"),
        ).select(
            pl.col("video_frame"),
            pl.col("agent_mouse_id"),
            pl.col("target_mouse_id"),
            *[
                body_parts_distance(body_part_1, body_part_2).alias(f"aa__{body_part_1}__{body_part_2}__distance")
                for body_part_1, body_part_2 in itertools.combinations(BODY_PARTS, 2)
            ],
            *[
                body_part_speed(body_part, period_ms).alias(f"agent__{body_part}__speed_{period_ms}ms")
                for body_part, period_ms in itertools.product(["tail_base", "nose"], [500, 1000, 2000, 3000])
            ],
            elongation().alias("agent__elongation"),
            body_angle().alias("agent__body_angle"),
        )

        pivot_renamed = pivot.rename({
            col: col.replace("agent_x_", "x_").replace("agent_y_", "y_")
            for col in pivot.columns if col.startswith("agent_")
        })
        
        engineered = transform_single(
            pivot_renamed, 
            body_parts_tracked=BODY_PARTS,
            fps=metadata["frames_per_second"]
        )

        tracking_cols = [col for col in engineered.columns if col.startswith(('x_', 'y_'))]
        engineered_only = engineered.drop(tracking_cols)


        if 'video_frame' in engineered_only.columns:
            engineered_only = engineered_only.with_columns(
                pl.col('video_frame').cast(pl.Int32)
            )


        engineered_only = engineered_only.rename({
            col: f"agent__{col}" for col in engineered_only.columns
            if col != "video_frame"
        })

        features = features.join(
            engineered_only,
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