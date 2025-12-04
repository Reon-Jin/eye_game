# Eye Shooter 快速开始指南

## 项目完成情况

✅ **已完成**：
- 核心眼动追踪模块（改进的视线向量估计、眨眼检测）
- 完整游戏逻辑引擎（靶子管理、碰撞检测、计分、难度递进）
- 实时UI和调试工具
- 完整的项目结构和文档

## 改进内容

### 1. 眼动追踪改进 ✨
- **更准确的视线向量估计**：基于瞳孔相对眼睛位置的计算
- **视线平滑处理**：使用低通滤波减少抖动
- **改进的坐标映射**：非线性映射以提高边缘准确性
- **眨眼检测优化**：
  - 降低 EAR 阈值（0.20→0.18）
  - 增加眨眼冷却时间防止重复触发
  - 使用帧计数更精确的检测

### 2. 游戏体验改进 🎮
- **眨眼反馈**：屏幕上显示眨眼闪光效果
- **实时调试信息**：显示左右眼 EAR 值、准心坐标
- **敏感度调整**：按 `-/=` 键在游戏中调整眨眼阈值
- **更好的状态提示**：清晰的游戏状态显示

### 3. 代码质量改进 💻
- 更好的错误处理和异常管理
- 完整的文档注释
- 模块化设计便于扩展
- 支持不同分辨率和硬件配置

## 安装指南

### 方案 1：使用当前环境（如果 MediaPipe 有问题）

```bash
# 1. 进入项目目录
cd d:\python_object\eye_game

# 2. 创建新的虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate

# 3. 升级 pip
pip install --upgrade pip

# 4. 安装依赖
pip install opencv-python numpy
pip install mediapipe  # 或者使用 conda install mediapipe
```

### 方案 2：使用 Conda（推荐解决 MediaPipe DLL 问题）

```bash
# 1. 创建新的 conda 环境
conda create -n eye_shooter python=3.10 -y
conda activate eye_shooter

# 2. 安装依赖
conda install opencv numpy -y
pip install mediapipe

# 3. 进入项目目录
cd d:\python_object\eye_game

# 4. 运行游戏
python main.py
```

### 方案 3：使用 Python 3.9（最稳定）

```bash
conda create -n eye_shooter python=3.9 -y
conda activate eye_shooter
cd d:\python_object\eye_game
pip install opencv-python mediapipe numpy
python main.py
```

## 快速开始

### 1. 运行测试
```bash
cd d:\python_object\eye_game
python test_modules.py
```

### 2. 启动游戏
```bash
python main.py
```

## 游戏控制

| 按键 | 功能 |
|------|------|
| **Space** | 开始/暂停/继续游戏 |
| **D** | 显示/隐藏调试信息 |
| **L** | 显示/隐藏眼睛特征点 |
| **R** | 重置游戏 |
| **-** | 降低眨眼敏感度（EAR 阈值下降） |
| **=** | 提高眨眼敏感度（EAR 阈值上升） |
| **ESC** | 退出程序 |

## 游戏玩法

1. **启动**：按 Space 开始游戏
2. **控制**：通过改变视线方向控制黄色十字准心
3. **射击**：左眼眨眼触发射击
4. **得分**：准心接触绿色靶子时得 10 分
5. **升级**：每 10 分升一级，难度增加

## 调试技巧

### 问题：眨眼不能触发射击

**解决步骤**：

1. 按 D 显示调试信息
2. 观察 `L-EAR` 和 `R-EAR` 值
3. 用力眨眼，记下 EAR 下降的最低值
4. 根据最低值调整阈值：
   - 如果最低值是 0.15，设置阈值为 0.16
   - 按 `-` 键逐步降低阈值直到能检测到眨眼
5. 观察 `Blink Threshold` 显示的当前值

### 问题：准心跟踪不准确

1. 确保光线充足（30-60cm 距离）
2. 摄像头应该在眼睛水平线上方
3. 按 L 显示人脸特征点，确保检测准确
4. 尝试调整摄像头位置或距离

### 问题：FPS 过低

1. 按 L 关闭特征点显示（可提升 5-10 FPS）
2. 在 `config.py` 中修改分辨率：
   ```python
   RESOLUTION_PRESET = "medium"  # 改为 "low" 获得更高 FPS
   ```
3. 关闭其他应用以释放系统资源

## 文件结构说明

```
eye_game/
├── main.py                          # 主程序入口
├── config.py                        # 配置文件
├── test_modules.py                  # 测试脚本
├── requirements.txt                 # 依赖包列表
├── README.md                        # 项目说明（详细版）
├── QUICKSTART.md                    # 本文件
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── gaze_tracker.py         # 眼动追踪模块（改进版）
│   └── game/
│       ├── __init__.py
│       └── game_engine.py          # 游戏引擎模块
└── assets/                          # 资源文件夹（可用于未来扩展）
```

## 性能优化建议

### 配置推荐

**高端设备**（RTX 3080+）
```python
# config.py
RESOLUTION_PRESET = "high"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
```

**中端设备**（GTX 1660+、RTX 2060）
```python
RESOLUTION_PRESET = "medium"
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
```

**低端设备**（集成显卡、旧款显卡）
```python
RESOLUTION_PRESET = "low"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
EYE_ASPECT_RATIO_THRESHOLD = 0.20  # 提高阈值以减少误触
```

## 键盘快捷键参考

| 场景 | 快捷键 | 功能 |
|------|--------|------|
| 菜单界面 | Space | 开始游戏 |
| 游戏中 | Space | 暂停游戏 |
| 暂停中 | Space | 继续游戏 |
| 任意界面 | D | 切换调试信息 |
| 任意界面 | L | 切换特征点显示 |
| 任意界面 | R | 重置游戏 |
| 游戏中 | - | 降低眨眼敏感度 |
| 游戏中 | = | 提高眨眼敏感度 |
| 任意界面 | ESC | 退出程序 |

## 常见问题

### Q1: 如何卸载重新安装 MediaPipe？

```bash
# 完全卸载
pip uninstall mediapipe opencv-python -y

# 清除缓存
pip cache purge

# 重新安装
pip install opencv-python mediapipe numpy
```

### Q2: 游戏很卡怎么办？

- 降低分辨率（在 `config.py` 修改）
- 关闭特征点显示（按 L）
- 关闭其他应用
- 检查是否有病毒扫描程序在后台运行

### Q3: 眼睛检测不到怎么办？

1. 增加光照（亮度不足是最常见原因）
2. 清洁摄像头镜头
3. 调整摄像头角度（应该朝向眼睛）
4. 确保脸部完全在画面内
5. 尝试不同的距离（30-60cm）

### Q4: 能修改游戏参数吗？

当然可以！所有参数都在 `config.py` 中：

```python
# 调整眨眼灵敏度
EYE_ASPECT_RATIO_THRESHOLD = 0.18  # 越小越灵敏

# 调整靶子生成速度
INITIAL_TARGET_SPAWN_INTERVAL = 60  # 越小越快

# 调整最大靶子数
MAX_INITIAL_TARGETS = 3  # 越大越难
```

## 下一步开发

可以考虑添加以下功能：

1. **校准系统**：点击屏幕四个角进行视线校准
2. **声音反馈**：射击、得分、升级的音效
3. **排行榜**：保存和显示历史最高分
4. **不同靶子类型**：移动靶、静止靶、小靶
5. **关卡系统**：多个关卡和任务
6. **网络多人**：通过网络与其他玩家对战

## 许可证

MIT License

## 需要帮助？

如果遇到问题：

1. 检查 `README.md` 的完整故障排除指南
2. 查看本文件的常见问题部分
3. 运行 `test_modules.py` 检查环境配置
4. 按 D 显示调试信息了解实时状态

---

**祝你游戏愉快！** 🎮👀
