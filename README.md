# Eye Shooter - 眼动追踪射击游戏

一个基于 OpenCV 和 MediaPipe 的眼动追踪射击游戏。玩家通过视线方向控制准心，左眨眼触发射击。

## 功能特性

✨ **核心功能：**
- 🎯 **眼动追踪**：使用 MediaPipe Face Mesh 进行高精度眼球追踪
- 👁️ **准心控制**：实时将视线向量映射到屏幕坐标
- 🔫 **眨眼射击**：通过左眼眨眼触发射击机制
- 🎮 **游戏引擎**：支持多个移动靶子、碰撞检测、计分系统
- 📈 **难度递进**：随着分数增加，靶子数量和生成速度增加
- 🖥️ **实时UI**：显示分数、等级、FPS等信息

## 技术架构

```
Eye Shooter/
├── main.py                  # 主程序入口
├── config.py               # 配置文件
├── requirements.txt        # 依赖包列表
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── gaze_tracker.py  # 眼动追踪模块
│   └── game/
│       ├── __init__.py
│       └── game_engine.py   # 游戏引擎模块
└── README.md               # 本文件
```

## 模块说明

### 1. GazeTracker (src/core/gaze_tracker.py)

**眼动追踪核心模块**

#### 主要功能：
- **Face Mesh 检测**：使用 MediaPipe 检测面部 468 个特征点
- **瞳孔定位**：从眼睛关键点计算瞳孔中心位置
- **视线向量估计**：基于瞳孔相对于眼睛的位置计算 gaze vector
- **眨眼检测**：通过眼睛宽高比（EAR）检测眨眼动作
- **坐标映射**：将视线向量转换为屏幕像素坐标

#### 关键算法：

**1. 眼睛宽高比（Eye Aspect Ratio, EAR）**
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```
其中 p1-p6 是眼睛的 6 个特征点。当 EAR < 阈值时，判定为眨眼。

**2. 视线向量估计**
```
gaze_x = (pupil_x - eye_inner_x) / eye_width
gaze_y = pupil_y / screen_height
```

**3. 屏幕坐标映射**
```
screen_x = (gaze_x + 1.0) / 3.0 * screen_width
screen_y = (gaze_y + 0.5) / 2.0 * screen_height
```

#### 核心方法：
```python
# 处理单帧图像
result = gaze_tracker.process_frame(frame)
# 返回: {face_detected, gaze_vector, gaze_screen_coords, blink_detected, landmarks}

# 获取瞳孔位置
pupil_x, pupil_y = gaze_tracker.get_pupil_position(eye_points)

# 检测眨眼
is_blink = gaze_tracker.detect_left_blink(face_landmarks)
```

### 2. GameEngine (src/game/game_engine.py)

**游戏逻辑引擎**

#### 主要功能：
- **靶子管理**：创建、更新、移除目标对象
- **碰撞检测**：检测准心与靶子是否碰撞
- **计分系统**：管理分数和等级
- **难度调整**：根据分数动态调整游戏难度

#### Target 类：
```python
@dataclass
class Target:
    x, y              # 位置
    width, height     # 大小
    vx, vy            # 速度
    alive             # 是否存活
    hit_frames        # 爆炸效果持续帧数
    
    def update()      # 更新位置
    def is_hit()      # 检测碰撞
    def draw()        # 绘制目标
```

#### 游戏状态机：
```
MENU → (Space) → PLAYING ← (Space) → PAUSED → (Space) → PLAYING
                     ↓ (R)
                  RESET
```

#### 难度递进：
- 基础：分数 0-9，1个靶子，60帧生成一个
- Lv.1：分数 10-19，2个靶子，57帧生成一个
- Lv.2+：随着分数增加，靶子数和速度不断增加

### 3. EyeShooterApp (main.py)

**主程序应用层**

处理：
- 摄像头捕捉和帧处理
- UI 绘制（准心、靶子、分数）
- 按键事件处理
- 性能监测

## 安装和使用

### 1. 环境要求
- Python 3.7+
- 具有摄像头的计算机
- 充足的光照环境（提高人脸检测精度）

### 2. 安装依赖

```bash
# 使用 pip 安装
pip install -r requirements.txt
```

或者手动安装：
```bash
pip install opencv-python mediapipe numpy
```

### 3. 运行程序

```bash
python main.py
```

程序启动后会打开摄像头窗口，显示实时追踪效果。

### 4. 游戏操作

| 按键 | 功能 |
|------|------|
| **Space** | 开始/暂停游戏 |
| **D** | 切换调试信息显示 |
| **L** | 切换面部特征点显示 |
| **R** | 重置游戏 |
| **ESC** | 退出程序 |

### 5. 游戏规则

1. **启动游戏**：按 Space 开始
2. **控制准心**：通过改变视线方向控制准心（黄色十字）
3. **射击**：左眨眼触发射击（需要明显的眨眼动作）
4. **击中靶子**：准心与绿色靶子碰撞时得 10 分
5. **升级机制**：每 10 分升一级，靶子增多、移动更快

## 参数调整

所有可配置参数位于 `config.py`：

```python
# 摄像头设置
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 眨眼检测
EYE_ASPECT_RATIO_THRESHOLD = 0.2    # 降低此值使眨眼更容易检测
BLINK_CONSEC_FRAMES = 3             # 眨眼连续帧数

# 游戏设置
INITIAL_TARGET_SPAWN_INTERVAL = 60  # 靶子生成间隔
MAX_INITIAL_TARGETS = 3             # 初始最大靶子数
```

## 故障排除

### 问题 1：无法打开摄像头
```
解决方案：
1. 检查摄像头是否被其他程序占用
2. 尝试修改 main.py 中的 camera_id（0, 1, 2...）
3. 检查摄像头权限
```

### 问题 2：无法检测到人脸
```
解决方案：
1. 确保充足的光照（MediaPipe 对光线敏感）
2. 调整脸部到摄像头中央
3. 确保脸部完全在画面内
4. 尝试调整摄像头距离（30-60cm 最佳）
```

### 问题 3：眨眼不能触发射击
```
解决方案：
1. 按 D 查看 EAR（Eye Aspect Ratio）值
2. 根据实际 EAR 值调整 config.py 中的阈值
3. 确保是左眼眨眼（右眼绿色点，左眼蓝色点）
4. 尝试更大幅度地眨眼
```

### 问题 4：准心不跟踪准确
```
解决方案：
1. 按 L 显示面部特征点，检查检测是否准确
2. 调整摄像头位置和角度
3. 在不同光线条件下测试
4. 可以修改 gaze_tracker.py 中的映射参数微调
```

### 问题 5：FPS 过低
```
解决方案：
1. 降低摄像头分辨率（在 main.py 中修改 width/height）
2. 关闭特征点显示（按 L）
3. 检查 CPU/GPU 使用率
4. 检查是否有其他后台程序占用资源
```

## 性能优化建议

1. **分辨率优化**：
   - 高配置（RTX 3080+）：1280×720
   - 中配置（GTX 1660+）：960×540
   - 低配置（集成显卡）：640×480

2. **实时性优化**：
   - 关闭特征点绘制（提升 ~5-10 FPS）
   - 使用多线程处理摄像头和 AI 模型
   - 考虑使用 GPU 加速（OpenCV CUDA）

3. **准确性优化**：
   - 增加光照以提高人脸检测率
   - 使用校准流程微调映射参数
   - 实现多帧平滑（低通滤波）

## 扩展功能建议

### 短期（1-2周）
- [ ] 实现校准界面（点击屏幕四个角，优化映射）
- [ ] 添加声音反馈（射击、得分）
- [ ] 实现排行榜系统（本地或云端）
- [ ] 添加不同难度模式

### 中期（1个月）
- [ ] 多种靶子类型（移动靶、静止靶、生命值靶）
- [ ] 3D 游戏视角（使用 pygame/Unreal）
- [ ] 网络多人对战
- [ ] 使用深度学习优化眼球追踪

### 长期（2-3个月）
- [ ] 完整的 3D 游戏引擎集成
- [ ] VR 支持（HTC Vive、Meta Quest）
- [ ] 移动端适配（Android/iOS）
- [ ] AI 对手和任务关卡

## 代码示例

### 自定义游戏参数

```python
from src.core.gaze_tracker import GazeTracker
from src.game.game_engine import GameEngine

# 创建追踪器
tracker = GazeTracker(camera_width=1920, camera_height=1080)

# 创建游戏引擎
engine = GameEngine(screen_width=1920, screen_height=1080)

# 自定义参数
engine.target_spawn_interval = 45
engine.max_targets = 5

# 开始游戏
engine.start_game()
```

### 集成到其他应用

```python
import cv2
from src.core.gaze_tracker import GazeTracker

def integrate_gaze_tracking(frame):
    tracker = GazeTracker()
    result = tracker.process_frame(frame)
    
    if result['face_detected']:
        gaze_x, gaze_y = result['gaze_screen_coords']
        is_blink = result['blink_detected']
        
        # 你的自定义逻辑
        print(f"Gaze at ({gaze_x}, {gaze_y}), Blink: {is_blink}")
    
    return result
```

## 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 人脸检测 | MediaPipe Face Mesh | 0.10.8 |
| 图像处理 | OpenCV | 4.8.1 |
| 数值计算 | NumPy | 1.24.3 |
| 语言 | Python | 3.7+ |

## 参考资源

- [MediaPipe Face Mesh 文档](https://google.github.io/mediapipe/solutions/face_mesh)
- [OpenCV 官方文档](https://docs.opencv.org/)
- [眼动追踪原理](https://en.wikipedia.org/wiki/Eye_tracking)
- [眼睛宽高比论文](https://ieeexplore.ieee.org/document/5891805)

## 许可证

MIT License

## 作者

Created with ❤️ for eye-tracking enthusiasts

## 更新日志

### v1.0.0 (2025-12-04)
- ✅ 完成眼动追踪模块
- ✅ 完成游戏引擎和 UI
- ✅ 支持基础游戏流程
- ✅ 添加调试工具

---

**享受游戏！** 🎮👀

