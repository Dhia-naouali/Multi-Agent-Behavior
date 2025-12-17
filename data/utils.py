import polars as pl


def _scale(n_frames_at_30fps, fps, ref=30.0):
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def add_curvature_features(x, center_x, center_y, fps):
    vel_x = pl.col(center_x).diff().alias("_vel_x")
    vel_y = pl.col(center_y).diff().alias("_vel_y")
    acc_x = pl.col(center_x).diff().diff().alias("_acc_x")
    acc_y = pl.col(center_y).diff().diff().alias("_acc_y")
    
    cross = (vel_x * acc_y) - (vel_y * acc_x)
    vel_mag = (vel_x**2 + vel_y**2).sqrt()
    curvature = ((cross.abs()) / (vel_mag**3 + 1e-6)).alias("_curvature")
    angle = pl.arctan2(vel_y, vel_x).alias("_angle")
    
    x = x.with_columns([vel_x, vel_y, acc_x, acc_y, curvature, angle])
    
    rolls = []
    for w in [25, 50, 75, 150]:
        ws = _scale(w, fps)
        rolls.append(
            pl.col("_curvature").rolling_mean(ws, min_samples=max(1, ws // 5)).alias(f'curv_mean_{w}')
        )
    
    angle_change = pl.col("_angle").diff().abs().alias("_angle_change")
    x = x.with_columns([angle_change, *rolls])
    
    w = 30
    ws = _scale(w, fps)
    turn_expr = pl.col("_angle_change").rolling_sum(window_size=ws, min_samples=max(1, ws // 5)).alias(f"turn_rate_{w}")
    
    return x.with_columns(turn_expr).drop(["_vel_x","_vel_y","_acc_x","_acc_y","_curvature","_angle","_angle_change"])


def add_multiscale_features(x, center_x, center_y, fps):
    vel_x = pl.col(center_x).diff()
    vel_y = pl.col(center_y).diff()
    speed = (vel_x.pow(2) + vel_y.pow(2)).sqrt() * float(fps)
    
    x = x.with_columns(speed.alias("_speed"))
    
    scales = [20, 40, 60, 80]
    exprs = []
    
    for scale in scales:
        ws = _scale(scale, fps)
        exprs.append(
            pl.col("_speed").rolling_mean(ws, min_samples=max(1, ws // 4)).alias(f'sp_m{scale}')
        )
        exprs.append(
            pl.col("_speed").rolling_std(ws, min_samples=max(1, ws // 4)).alias(f'sp_s{scale}')
        )
    
    x = x.with_columns(exprs)
    
    if len(scales) >= 2:
        x = x.with_columns(
            (pl.col(f'sp_m{scales[0]}') / (pl.col(f'sp_m{scales[-1]}') + 1e-6)).alias('sp_ratio')
        )
    
    return x.drop("_speed")


def add_longrange_features(x, center_x, center_y, fps):
    exprs = []
    
    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        exprs.append(
            pl.col(center_x).rolling_mean(ws, min_samples=max(5, ws // 6)).alias(f'x_ml{window}')
        )
        exprs.append(
            pl.col(center_y).rolling_mean(ws, min_samples=max(5, ws // 6)).alias(f'y_ml{window}')
        )
    
    for span in [30, 60, 120]:
        s = _scale(span, fps)
        exprs.append(
            pl.col(center_x).ewm_mean(span=s, min_samples=1).alias(f'x_e{span}')
        )
        exprs.append(
            pl.col(center_y).ewm_mean(span=s, min_samples=1).alias(f'y_e{span}')
        )
    
    x = x.with_columns(exprs)
    
    vel_x = pl.col(center_x).diff()
    vel_y = pl.col(center_y).diff()
    speed = (vel_x.pow(2) + vel_y.pow(2)).sqrt() * float(fps)
    
    x = x.with_columns(speed.alias("_speed"))
    
    rank_exprs = []
    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        min_val = pl.col("_speed").rolling_min(ws, min_samples=max(5, ws // 6))
        max_val = pl.col("_speed").rolling_max(ws, min_samples=max(5, ws // 6))
        
        pct = ((pl.col("_speed") - min_val) / (max_val - min_val + 1e-6)).alias(f'sp_pct{window}')
        rank_exprs.append(pct)
    
    x = x.with_columns(rank_exprs)
    
    return x.drop("_speed")



def add_state_features(x, center_x, center_y, fps):
    vel_x = pl.col(center_x).diff()
    vel_y = pl.col(center_y).diff()
    speed = (vel_x.pow(2) + vel_y.pow(2)).sqrt() * float(fps)
    
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling_mean(w_ma, min_samples=max(1, w_ma // 3))
    
    x = x.with_columns(speed_ma.alias("_speed_ma"))
    
    bins = [-float('inf'), 0.5 * fps, 2.0 * fps, 5.0 * fps, float('inf')]
    
    speed_states = (
        pl.when(pl.col("_speed_ma") <= bins[1]).then(0)
        .when(pl.col("_speed_ma") <= bins[2]).then(1)
        .when(pl.col("_speed_ma") <= bins[3]).then(2)
        .otherwise(3)
        .alias("_speed_states")
    )
    
    x = x.with_columns(speed_states)
    
    exprs = []
    for window in [20, 40, 60, 80]:
        ws = _scale(window, fps)
        
        for state in [0, 1, 2, 3]:
            exprs.append(
                (pl.col("_speed_states") == state)
                .cast(pl.Float64)
                .rolling_mean(ws, min_samples=max(1, ws // 5))
                .alias(f's{state}_{window}')
            )
        
        state_changes = (pl.col("_speed_states") != pl.col("_speed_states").shift(1)).cast(pl.Float64)
        exprs.append(
            state_changes.rolling_sum(ws, min_samples=max(1, ws // 5)).alias(f'trans_{window}')
        )
    
    x = x.with_columns(exprs)
    
    return x.drop(["_speed_ma", "_speed_states"])


def add_cumulative_distance_single(x, cx, cy, fps, horizon_frames_base=180):
    L = max(1, _scale(horizon_frames_base, fps))
    
    step = ((pl.col(cx).diff().pow(2) + pl.col(cy).diff().pow(2)).sqrt()).alias("_step")
    
    x = x.with_columns(step)
    path = (
        pl.col("_step")
        .rolling_sum(window_size=2*L + 1, min_samples=max(5, L//6), center=True)
        .fill_null(0.0)
        .alias(f"path_cum{horizon_frames_base}")
    )
    
    x = x.with_columns(path).drop("_step")
    
    return x



def add_groom_microfeatures(x, fps, 
                           body_center_x='x_body_center', 
                           body_center_y='y_body_center',
                           nose_x='x_nose',
                           nose_y='y_nose',
                           tail_base_x='x_tail_base',
                           tail_base_y='y_tail_base'):
    required = [body_center_x, body_center_y, nose_x, nose_y]
    if not all(col in x.columns for col in required):
        return x
    
    body_speed = (
        (pl.col(body_center_x).diff().pow(2) + pl.col(body_center_y).diff().pow(2))
        .sqrt() * float(fps)
    ).fill_null(0).alias("_body_speed")
    
    nose_speed = (
        (pl.col(nose_x).diff().pow(2) + pl.col(nose_y).diff().pow(2))
        .sqrt() * float(fps)
    ).fill_null(0).alias("_nose_speed")
    
    x = x.with_columns([body_speed, nose_speed])
    
    w30 = _scale(30, fps)
    
    head_body_decouple = (
        (pl.col("_nose_speed") / (pl.col("_body_speed") + 1e-3))
        .clip(0, 10)
        .rolling_median(w30, min_samples=max(1, w30//3))
        .alias("head_body_decouple")
    )
    
    nose_distance = (
        (pl.col(nose_x) - pl.col(body_center_x)).pow(2) + 
        (pl.col(nose_y) - pl.col(body_center_y)).pow(2)
    ).sqrt()
    
    nose_rad_std = (
        nose_distance
        .rolling_std(w30, min_samples=max(1, w30//3))
        .fill_null(0)
        .alias("nose_rad_std")
    )
    
    x = x.with_columns([head_body_decouple, nose_rad_std])
    

    if tail_base_x in x.columns and tail_base_y in x.columns:
        head_angle = pl.arctan2(
            pl.col(nose_y) - pl.col(tail_base_y),
            pl.col(nose_x) - pl.col(tail_base_x)
        )
        
        angle_change = head_angle.diff().abs().fill_null(0)
        
        head_orient_jitter = (
            angle_change
            .rolling_mean(w30, min_samples=max(1, w30//3))
            .alias("head_orient_jitter")
        )
        
        x = x.with_columns(head_orient_jitter)
    
    return x.drop(["_body_speed", "_nose_speed"])


def add_mouth_window_features(df: pl.DataFrame, pix_per_cm: float, fps: float, window_ms_list=[200, 500, 1000]):
    eps = 1e-6

    xcol = "x_nose"
    ycol = "y_nose"

    inst_step_px = pl.col(xcol).diff().pow(2) + pl.col(ycol).diff().pow(2)
    inst_step = inst_step_px.sqrt() / pix_per_cm  # cm per frame
    df = df.with_columns(inst_step.alias("mouth_inst_step_cm"))

    for w_ms in window_ms_list:
        w = max(1, int(round(w_ms * fps / 1000.0)))
        prefix = f"mouth_w{w_ms}ms"

        df = df.with_columns([
            pl.col(xcol).rolling_min(window_size=w, min_periods=1).alias(f"{prefix}__min_x"),
            pl.col(xcol).rolling_max(window_size=w, min_periods=1).alias(f"{prefix}__max_x"),
            pl.col(ycol).rolling_min(window_size=w, min_periods=1).alias(f"{prefix}__min_y"),
            pl.col(ycol).rolling_max(window_size=w, min_periods=1).alias(f"{prefix}__max_y"),
        ])

        df = df.with_columns([
            ((pl.col(f"{prefix}__max_x") - pl.col(f"{prefix}__min_x")) / pix_per_cm).alias(f"{prefix}__w_cm"),
            ((pl.col(f"{prefix}__max_y") - pl.col(f"{prefix}__min_y")) / pix_per_cm).alias(f"{prefix}__h_cm"),
        ])

        df = df.with_columns([
            (pl.col(f"{prefix}__w_cm") * pl.col(f"{prefix}__h_cm")).alias(f"{prefix}__area_cm2"),
            (pl.col(f"{prefix}__w_cm") / (pl.col(f"{prefix}__h_cm") + eps)).alias(f"{prefix}__aspect"),
            ( (pl.col(f"{prefix}__w_cm").pow(2) + pl.col(f"{prefix}__h_cm").pow(2)).sqrt() ).alias(f"{prefix}__diag_cm"),
        ])

        df = df.with_columns([
            pl.col("mouth_inst_step_cm").rolling_sum(window_size=w, min_periods=1).alias(f"{prefix}__path_len_cm"),
        ])

        df = df.with_columns([
            (
                ((pl.col(xcol) - pl.col(xcol).shift(w - 1)).pow(2) + (pl.col(ycol) - pl.col(ycol).shift(w - 1)).pow(2)).sqrt()
                / pix_per_cm
            ).alias(f"{prefix}__displacement_cm"),
        ])

        df = df.with_columns([
            (pl.col(f"{prefix}__displacement_cm") / (pl.col(f"{prefix}__path_len_cm") + eps)).alias(f"{prefix}__straightness"),
            (pl.col(f"{prefix}__area_cm2") / (pl.col(f"{prefix}__path_len_cm") + eps)).alias(f"{prefix}__area_per_path"),
        ])

        med = float(df.select(pl.col("mouth_inst_step_cm").median()).to_numpy()[0] or 0.0)
        std = float(df.select(pl.col("mouth_inst_step_cm").std()).to_numpy()[0] or 0.0)
        v_thresh = max(1e-6, med + std)
        df = df.with_columns([
            (pl.col("mouth_inst_step_cm") > v_thresh).cast(pl.Int32).rolling_sum(window_size=w, min_periods=1).alias(f"{prefix}__moving_count"),
        ])
        df = df.with_columns([
            (pl.col(f"{prefix}__moving_count") / w).alias(f"{prefix}__moving_frac"),
        ])

    return df