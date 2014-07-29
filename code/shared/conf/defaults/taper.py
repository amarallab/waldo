# How the Taper defines a "good blob"
TAPE_REL_MOVE_THRESHOLD = 0.5

# The minimum number of good blob traces to allow before outright failing
TAPE_MIN_TRACE_FAIL = 1

# The minimum number of good blob traces to allow before warning the user
TAPE_MIN_TRACE_WARN = 30

# Use this many traces to generate the scoring function.  "None" implies no
# limit.
TAPE_TRACE_LIMIT_NUM = 400

# Only search for connections this many frames out
TAPE_FRAME_SEARCH_LIMIT = 300

# Take this many samples in the KDE to generate a scoring function
TAPE_KDE_SAMPLES = 100

# Factor to increase the search cone's slope (maximum speed)
TAPE_MAX_SPEED_MULTIPLIER = 1.50

# Pixels added to the radius of the search cone to account for the
# jagged nature of the tracks
TAPE_SHAKYCAM_ALLOWANCE = 10

# Moving average window size to filter speeds by
TAPE_MAX_SPEED_SMOOTHING = 10
