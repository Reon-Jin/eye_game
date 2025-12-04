# 准心跟踪调试指南

## 快速诊断

### 1. 启动游戏时的诊断
```bash
cd d:\python_object\eye_game
python main.py
```

**需要检查的指标（按 D 显示调试信息）：**
- L-EAR：左眼宽高比（正常范围 0.3-0.5，眨眼时 <0.18）
- R-EAR：右眼宽高比
- Gaze：准心当前坐标
- Targets：当前活跃靶子数量

### 2. 如果准心不跟踪视线

#### 症状 1：准心总是在屏幕中央不动
**原因：** 人脸检测失败或视线向量计算出错

**调试步骤：**
1. 按 `L` 显示面部特征点
2. 检查蓝色（左眼）和绿色（右眼）的点是否准确
3. 确保摄像头距离在 30-60cm 之间
4. 增加光线亮度

#### 症状 2：准心跟踪有严重延迟
**原因：** 平滑缓冲太大，造成延迟

**调整方法：**
编辑 `src/core/gaze_tracker.py` 第 82-83 行：
```python
# 降低缓冲大小
self.gaze_smooth_buffer = deque(maxlen=5)  # 从 7 改为 5
```

#### 症状 3：准心跳动/抖动
**原因：** 平滑缓冲太小或摄像头不稳定

**调整方法：**
```python
# 增加缓冲大小
self.gaze_smooth_buffer = deque(maxlen=9)  # 从 7 改为 9
```

### 3. 如果眨眼时准心晃动

#### 症状：眨眼时准心明显跳跃
**根本原因：** 眼睛闭合时虚拟瞳孔点位置不稳定

**调整方法：**
编辑 `main.py` 第 76 行，增加锁定时间：
```python
self.blink_lock_frames = 8  # 从 6 改为 8，锁定更久

# 时间对应表（30 FPS）：
# 6 frames = ~200ms
# 8 frames = ~267ms
# 10 frames = ~333ms
```

**或编辑** `src/core/gaze_tracker.py` 第 88 行：
```python
# 增加眨眼冷却时间
self.blink_cooldown = 8  # 从 5 改为 8
```

### 4. 如果准心在屏幕的某个方向偏离

#### 症状 A：准心偏向屏幕上方
**原因：** 摄像头位置补正值太大（向上过度补偿）

**调整：**
编辑 `src/core/gaze_tracker.py` 第 101 行：
```python
self.camera_height_offset = -0.10  # 从 -0.15 改为 -0.10
```

#### 症状 B：准心偏向屏幕下方
**原因：** 摄像头位置补正值不足（向上补偿不够）

**调整：**
```python
self.camera_height_offset = -0.20  # 从 -0.15 改为 -0.20
```

#### 症状 C：准心偏向左/右
**原因：** 摄像头未正对屏幕中央

**解决方案：**
1. 调整摄像头水平位置，使其正对屏幕中央
2. 或在游戏中按 `-/=` 调整灵敏度（这是临时办法）

### 5. 如果准心跟踪的区域太小/太大

#### 症状：准心无法覆盖屏幕边缘
**原因：** 视线范围限制太小

**调整：**
编辑 `src/core/gaze_tracker.py` 第 167-168 行：
```python
# 扩大范围
avg_gaze_x = np.clip(avg_gaze_x, -1.1, 1.1)  # 从 -1.0, 1.0
avg_gaze_y = np.clip(avg_gaze_y, -0.9, 0.9)  # 从 -0.8, 0.8
```

#### 症状：准心控制范围过大，超出屏幕
**原因：** 视线范围限制太大

**调整：**
```python
avg_gaze_x = np.clip(avg_gaze_x, -0.9, 0.9)  # 从 -1.0, 1.0
avg_gaze_y = np.clip(avg_gaze_y, -0.7, 0.7)  # 从 -0.8, 0.8
```

## 进阶调优

### 非线性变换参数

编辑 `src/core/gaze_tracker.py` 的 `gaze_to_screen_coords` 方法：

```python
# 当前：使用 0.9 次方（轻微非线性）
gx_nonlinear = np.sign(gx) * (abs(gx) ** 0.9)

# 调整选项：
# 0.85：更多非线性，中央控制更精细
# 0.90：当前值，平衡点
# 0.95：更接近线性，更灵敏
# 1.00：完全线性，无非线性变换
```

**效果说明：**
- 指数 < 1.0：中央区域变大，边缘缩小（类似"S曲线"）
- 指数 = 1.0：线性映射，每个视线位置对应相同屏幕移动量
- 指数 > 1.0：边缘区域变大，中央缩小（不推荐）

### 坐标平滑自适应参数

编辑 `src/core/gaze_tracker.py` 的 `gaze_to_screen_coords` 方法：

```python
# 调整阈值来改变平滑行为
if distance < 30:  # ← 小移动阈值
    smooth_factor = 0.6  # ← 小移动时的平滑度

elif distance < 100:  # ← 中等移动阈值
    smooth_factor = 0.4  # ← 中等移动时的平滑度

else:
    smooth_factor = 0.2  # ← 大移动时的平滑度
```

**调优示例：**
```python
# 如果眼睛微动时还是有抖动，增加平滑
if distance < 25:  # 更严格的"微动"定义
    smooth_factor = 0.7  # 更强的平滑
```

## 性能检查

### 检查 FPS
1. 启动游戏
2. 按 `D` 显示调试信息
3. 观察左上角的 FPS 值

**目标 FPS：**
- 高端电脑：50-60 FPS
- 中端电脑：40-50 FPS
- 低端电脑：30-40 FPS

**如果 FPS < 30，优化步骤：**
1. 按 `L` 关闭特征点显示（+5-10 FPS）
2. 在 `config.py` 改为低分辨率预设：
   ```python
   RESOLUTION_PRESET = "low"  # 改为 "low"
   ```
3. 降低缓冲大小：
   ```python
   self.gaze_smooth_buffer = deque(maxlen=5)  # 从 7 改为 5
   ```

## 参数配置清单

### 关键参数位置

| 参数 | 文件 | 行号 | 说明 |
|------|------|------|------|
| `gaze_smooth_buffer.maxlen` | gaze_tracker.py | 82 | 视线平滑缓冲 |
| `camera_height_offset` | gaze_tracker.py | 101 | 摄像头高度补正 |
| `gaze_nonlinear_factor` | gaze_tracker.py | 234-235 | 非线性变换 |
| `blink_lock_frames` | main.py | 76 | 眨眼锁定时间 |
| `smooth_factor` | gaze_tracker.py | 247-252 | 坐标平滑 |
| `RESOLUTION_PRESET` | config.py | 44 | 分辨率预设 |

### 推荐配置方案

#### 方案 A：灵敏度优先（快速响应）
```python
# gaze_tracker.py
self.gaze_smooth_buffer = deque(maxlen=5)
self.camera_height_offset = -0.15

# gaze_to_screen_coords
gx_nonlinear = np.sign(gx) * (abs(gx) ** 0.95)  # 更线性
smooth_factor = 0.2  # 默认条件下

# main.py
self.blink_lock_frames = 4
```

#### 方案 B：稳定性优先（平滑准心）
```python
# gaze_tracker.py
self.gaze_smooth_buffer = deque(maxlen=9)
self.camera_height_offset = -0.15

# gaze_to_screen_coords
gx_nonlinear = np.sign(gx) * (abs(gx) ** 0.85)  # 更非线性
smooth_factor = 0.6  # 默认条件下

# main.py
self.blink_lock_frames = 8
```

#### 方案 C：平衡方案（当前默认，推荐）
```python
# 保持 v1.2.0 的默认值
# gaze_tracker.py
self.gaze_smooth_buffer = deque(maxlen=7)
self.camera_height_offset = -0.15

# 保持所有其他默认值
```

## 快速命令参考

| 按键 | 功能 | 用途 |
|------|------|------|
| `D` | 显示/隐藏调试 | 查看 FPS、EAR、坐标 |
| `L` | 显示/隐藏特征点 | 诊断人脸检测 |
| `-` | 降低眨眼灵敏度 | 不容易误触 |
| `=` | 提高眨眼灵敏度 | 容易触发 |
| `R` | 重置游戏 | 清除状态 |
| `Space` | 开始/暂停 | 游戏控制 |
| `ESC` | 退出 | 关闭程序 |

## 故障排除流程

```
准心不跟踪
├─ 显示特征点（L键）
│  ├─ 特征点正确 → 问题在视线计算
│  │  └─ 检查 estimate_gaze_vector() 的向量范围
│  └─ 特征点错误 → 人脸检测问题
│     └─ 增加光线、调整摄像头距离
│
准心跟踪有延迟
├─ 减少缓冲大小 (maxlen: 7 → 5)
└─ 增加 FPS 检查（显示调试信息）
│
眨眼时晃动
├─ 增加眨眼锁定时间 (blink_lock_frames: 6 → 8)
├─ 或增加冷却时间 (blink_cooldown: 5 → 7)
└─ 检查眼睛检测是否正确
│
准心偏离
├─ 调整摄像头补正 (camera_height_offset)
├─ 或物理调整摄像头位置
└─ 考虑用户校准
```

## 获取帮助

如果问题仍未解决：

1. 检查 `README.md` 的故障排除部分
2. 查看 `IMPROVEMENTS_v1.2.md` 了解技术细节
3. 运行 `test_modules.py` 检查环境
4. 启用调试模式（D键）收集诊断信息
5. 检查摄像头是否被其他应用占用

---

**祝调试愉快！** 🎯👀
