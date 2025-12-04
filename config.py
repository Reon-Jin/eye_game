"""
Eye Shooter - Configuration
优化的眼动追踪射击游戏配置
"""

# ==================== 摄像头设置 ====================
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# ==================== 眼动追踪设置 ====================
# 眨眼检测参数
EYE_ASPECT_RATIO_THRESHOLD = 0.18  # Eye Aspect Ratio 阈值（降低以提高检测灵敏度）
BLINK_CONSEC_FRAMES = 2  # 眨眼持续帧数（越小越灵敏）
BLINK_COOLDOWN = 5  # 眨眼冷却时间（帧数），防止多次触发

# 视线向量平滑
GAZE_SMOOTHING_FACTOR = 0.6  # 平滑因子 (0-1，越大越平滑)
GAZE_SMOOTH_BUFFER_SIZE = 3  # 历史缓冲区大小

# MediaPipe 检测置信度
FACE_DETECTION_CONFIDENCE = 0.7  # 人脸检测置信度
FACE_TRACKING_CONFIDENCE = 0.5  # 人脸追踪置信度

# ==================== 游戏设置 ====================
# 靶子生成
INITIAL_TARGET_SPAWN_INTERVAL = 60  # 初始靶子生成间隔（帧数）
MAX_INITIAL_TARGETS = 3  # 初始最大靶子数
MIN_SPAWN_INTERVAL = 30  # 最小生成间隔
MAX_TARGETS_LIMIT = 8  # 最大靶子数

# 准心
CROSSHAIR_SIZE = 20  # 准心大小（像素）

# ==================== 显示设置 ====================
SHOW_DEBUG = True  # 是否默认显示调试信息
SHOW_LANDMARKS = False  # 是否默认显示面部特征点

# ==================== 性能优化 ====================
# 根据您的硬件调整这些参数
# 高端设备 (RTX 3080+): 1280x720
# 中端设备 (GTX 1660+): 960x540
# 低端设备 (集成显卡): 640x480

RESOLUTION_PRESET = "high"  # "high", "medium", "low"

if RESOLUTION_PRESET == "medium":
    CAMERA_WIDTH = 960
    CAMERA_HEIGHT = 540
elif RESOLUTION_PRESET == "low":
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
