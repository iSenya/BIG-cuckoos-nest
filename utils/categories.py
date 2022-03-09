##############################
# This file is deprecated
print("!!!This file is deprecated!!!!")
##############################

vidvrd_categories = ['__background__',  # always index 0
            'airplane', 'antelope', 'bear', 'bicycle',
            'bird', 'bus', 'car', 'cattle',
            'dog', 'domestic_cat', 'elephant', 'fox',
            'giant_panda', 'hamster', 'horse', 'lion',
            'lizard', 'monkey', 'motorcycle', 'rabbit',
            'red_panda', 'sheep', 'snake', 'squirrel',
            'tiger', 'train', 'turtle', 'watercraft',
            'whale', 'zebra',                               # 前面几行的顺序和imagenet-DET中的顺序相同
            'ball','frisbee','sofa','skateboard','person']  # 最后一行是不在 imageNet-DET中的类
vidvrd_CatId2name = {idx:cat for idx,cat in enumerate(vidvrd_categories)}
vidvrd_CatNmae2id = {cat:idx for idx,cat in enumerate(vidvrd_categories)}

pred_categories = ["__background__",
    "taller","swim_behind","walk_away","fly_behind","creep_behind",
    "lie_with","move_left","stand_next_to","touch","follow",
    "move_away","lie_next_to","walk_with","move_next_to","creep_above",
    "stand_above","fall_off","run_with","swim_front","walk_next_to",
    "kick","stand_left","creep_right","sit_above","watch",
    "swim_with","fly_away","creep_beneath","front","run_past",
    "jump_right","fly_toward","stop_beneath","stand_inside","creep_left",
    "run_next_to","beneath","stop_left","right","jump_front",
    "jump_beneath","past","jump_toward","sit_front","sit_inside",
    "walk_beneath","run_away","stop_right","run_above","walk_right",
    "away","move_right","fly_right","behind","sit_right",
    "above","run_front","run_toward","jump_past","stand_with",
    "sit_left","jump_above","move_with","swim_beneath","stand_behind",
    "larger","walk_past","stop_front","run_right","creep_away",
    "move_toward","feed","run_left","lie_beneath","fly_front",
    "walk_behind","stand_beneath","fly_above","bite","fly_next_to",
    "stop_next_to","fight","walk_above","jump_behind","fly_with",
    "sit_beneath","sit_next_to","jump_next_to","run_behind","move_behind",
    "swim_right","swim_next_to","hold","move_past","pull",
    "stand_front","walk_left","lie_above","ride","next_to",
    "move_beneath","lie_behind","toward","jump_left","stop_above",
    "creep_toward","lie_left","fly_left","stop_with","walk_toward",
    "stand_right","chase","creep_next_to","fly_past","move_front",
    "run_beneath","creep_front","creep_past","play","lie_inside",
    "stop_behind","move_above","sit_behind","faster","lie_right",
    "walk_front","drive","swim_left","jump_away","jump_with",
    "lie_front","left"
]

pred_CatId2name = {idx:cat for idx,cat in enumerate(pred_categories)}
pred_CatNmae2id = {cat:idx for idx,cat in enumerate(pred_categories)}