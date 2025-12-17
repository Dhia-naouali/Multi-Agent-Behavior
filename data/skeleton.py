SELF_BEHAVIORS = [
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

BODY_PARTS = [
    "ear_left", "ear_right", "nose", "neck",
    "body_center", "lateral_left", "lateral_right",
    "hip_left", "hip_right", "tail_base", "tail_tip",
]

out_pairs = [
    ("neck", "nose"),
    ("ear_left", "nose"),
    ("ear_left", "neck"),
    ("ear_right", "nose"),
    ("ear_right", "neck"),
    ("hip_right", "tail_base"),
    ("hip_left", "tail_base")
]
