# Blobs with fewer frames than this are rolled up into their parent
COLLIDER_SUITE_OFFSHOOT = 20

# Blobs that split then rejoin for fewer frames than this are collapsed
COLLIDER_SUITE_SPLIT_ABS = 5

# Blobs that are split for proportionally less time than this relative
# to the average duration of their parent and child blobs
COLLIDER_SUITE_SPLIT_REL = 0.25

# Blobs that have a duration shorter than this are collapsed into their
# parent and/or child.
COLLIDER_SUITE_ASSIMILATE_SIZE = 10
