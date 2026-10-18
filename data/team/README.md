Put recordings here (any format ffmpeg can read):

    data/team/drone/        clips that contain a flying drone (trim to flying segments)
    data/team/background/   ambient / confusers with no drone

The recordings already committed under `data/drone` and `data/non_drone` are picked up
automatically as drone / background; anything you add here is used in addition.

Name files with the conditions, e.g. `mavic_25m_behind_building_45deg.m4a`. Files containing `_holdout` go
entirely into the test split. Nothing in this folder is committed to git.
