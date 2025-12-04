"""
Eye Shooter - Configuration
"""

# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Gaze tracking settings
EYE_ASPECT_RATIO_THRESHOLD = 0.2  # 眨眼检测阈值
BLINK_CONSEC_FRAMES = 3  # 连续帧数阈值

# Game settings
INITIAL_TARGET_SPAWN_INTERVAL = 60  # 帧数
MAX_INITIAL_TARGETS = 3
CROSSHAIR_SIZE = 20

# Display settings
SHOW_DEBUG = True
SHOW_LANDMARKS = False
