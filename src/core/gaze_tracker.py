"""
Eye Gaze Tracking Module
使用 OpenCV Haar Cascade 和基础眼动检测进行眼球追踪
计算视线向量（gaze vector）和眨眼检测
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import deque

# 加载 Haar Cascade 分类器（用于人脸和眼睛检测）
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'

try:
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
except Exception as e:
    print(f"Warning: Failed to load Haar cascades: {e}")
    face_cascade = None
    eye_cascade = None


class GazeTracker:
    """眼动追踪器 - 基于OpenCV Haar Cascade眼睛检测"""
    
    def __init__(self, camera_width: int = 1280, camera_height: int = 720):
        """
        初始化眼动追踪器
        
        Args:
            camera_width: 摄像头宽度
            camera_height: 摄像头高度
        """
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        if face_cascade is None or eye_cascade is None:
            raise RuntimeError("Failed to load Haar cascades. OpenCV may not be properly installed.")
        
        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade
        
        # 眨眼检测参数
        self.EAR_THRESHOLD = 0.18  # Eye Aspect Ratio threshold
        self.BLINK_CONSEC_FRAMES = 2  # 连续帧数阈值
        self.blink_counter = 0
        self.blink_detected = False
        self.last_blink_frame = -10  # 上一次眨眼的帧数
        self.blink_cooldown = 5  # 眨眼冷却时间（帧数）
        
        # 视线向量平滑
        self.gaze_smooth_buffer = deque(maxlen=7)
        self.gaze_smoothing_factor = 0.4
        self.prev_gaze_coords = None
        self.coord_smooth_factor = 0.5
        
        # 校准参数
        self.calibrated = False
        self.calibration_points = []
        self.screen_points = []
        
        # 调试信息
        self.last_ear_left = 0.0
        self.last_ear_right = 0.0
        
        # 摄像头位置补正
        self.camera_height_offset = -0.15
        
        # 眼睛位置历史（用于平滑）
        self.left_eye_history = deque(maxlen=5)
        self.right_eye_history = deque(maxlen=5)
        
        # 上一帧的眼睛状态
        self.prev_left_eye_state = None
        self.prev_right_eye_state = None
        
    def calculate_eye_aspect_ratio(self, eye_region: np.ndarray) -> float:
        """
        基于眼睛区域计算眼睛宽高比
        
        Args:
            eye_region: 眼睛ROI区域
            
        Returns:
            眼睛宽高比（0-1）
        """
        if eye_region is None or eye_region.size == 0:
            return 1.0
        
        # 计算眼睛区域的填充比例（作为宽高比的代理）
        # 眼睛闭合时，区域会变小或变暗
        h, w = eye_region.shape[:2]
        if h == 0 or w == 0:
            return 1.0
        
        # 计算区域的亮度平均值（眼睛打开时较亮）
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        brightness = np.mean(gray) / 255.0
        
        # 计算宽高比
        aspect_ratio = min(w, h) / max(w, h, 1)
        
        # 综合考虑宽高比和亮度
        ear = brightness * aspect_ratio
        return ear
    
    def smooth_eye_position(self, eye_rect: Tuple[int, int, int, int], is_left: bool = True) -> Optional[Tuple[int, int, int, int]]:
        """
        平滑眼睛位置以减少检测噪声
        
        Args:
            eye_rect: 眼睛矩形 (x, y, w, h)
            is_left: 是否为左眼
            
        Returns:
            平滑后的眼睛位置或None
        """
        if eye_rect is None:
            return None
        
        # 使用历史缓冲区平滑
        history = self.left_eye_history if is_left else self.right_eye_history
        history.append(eye_rect)
        
        if len(history) > 0:
            avg_x = int(np.mean([r[0] for r in history]))
            avg_y = int(np.mean([r[1] for r in history]))
            avg_w = int(np.mean([r[2] for r in history]))
            avg_h = int(np.mean([r[3] for r in history]))
            return (avg_x, avg_y, avg_w, avg_h)
        
        return None
    
    def estimate_gaze_vector(self, left_eye_rect: Tuple, right_eye_rect: Tuple, face_rect: Tuple) -> np.ndarray:
        """
        基于眼睛位置估计视线向量
        
        Args:
            left_eye_rect: 左眼矩形 (x, y, w, h)
            right_eye_rect: 右眼矩形 (x, y, w, h)
            face_rect: 脸部矩形 (x, y, w, h)
            
        Returns:
            归一化的视线向量 (2,)，范围约为 [-1, 1]
        """
        if left_eye_rect is None or right_eye_rect is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # 提取眼睛中心位置
        left_eye_center_x = left_eye_rect[0] + left_eye_rect[2] // 2
        left_eye_center_y = left_eye_rect[1] + left_eye_rect[3] // 2
        right_eye_center_x = right_eye_rect[0] + right_eye_rect[2] // 2
        right_eye_center_y = right_eye_rect[1] + right_eye_rect[3] // 2
        
        # 平均眼睛位置
        avg_eye_x = (left_eye_center_x + right_eye_center_x) / 2.0
        avg_eye_y = (left_eye_center_y + right_eye_center_y) / 2.0
        
        # 脸部中心
        face_center_x = face_rect[0] + face_rect[2] // 2
        face_center_y = face_rect[1] + face_rect[3] // 2
        
        # 计算相对于脸部中心的眼睛位置（归一化）
        # X: -1 (左) 到 1 (右)
        # Y: -1 (上) 到 1 (下)
        max_horizontal_offset = face_rect[2] / 3.0
        max_vertical_offset = face_rect[3] / 4.0
        
        gaze_x = (avg_eye_x - face_center_x) / max_horizontal_offset
        gaze_y = (avg_eye_y - face_center_y) / max_vertical_offset
        
        # 应用摄像头位置补正
        gaze_y += self.camera_height_offset
        
        # 约束范围
        gaze_x = np.clip(gaze_x, -1.0, 1.0)
        gaze_y = np.clip(gaze_y, -0.8, 0.8)
        
        gaze_vector = np.array([gaze_x, gaze_y], dtype=np.float32)
        
        # 应用平滑滤波
        gaze_vector = self._smooth_gaze_vector(gaze_vector)
        
        return gaze_vector
    
    def _smooth_gaze_vector(self, gaze_vector: np.ndarray) -> np.ndarray:
        """
        使用多级低通滤波平滑视线向量，减少噪声和抖动
        
        Args:
            gaze_vector: 当前视线向量
            
        Returns:
            平滑后的视线向量
        """
        self.gaze_smooth_buffer.append(gaze_vector)
        
        if len(self.gaze_smooth_buffer) < 2:
            return gaze_vector
        
        # 级联平滑：首先做简单移动平均
        smoothed = np.mean(list(self.gaze_smooth_buffer), axis=0)
        
        return smoothed
    
    def gaze_to_screen_coords(self, 
                             gaze_vector: np.ndarray,
                             screen_width: int,
                             screen_height: int) -> Tuple[int, int]:
        """
        改进的视线向量到屏幕坐标映射，考虑眼睛的非线性特性
        和摄像头位置（屏幕正上方）
        
        Args:
            gaze_vector: 归一化的视线向量 (2,) 范围约 [-1, 1]
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            
        Returns:
            屏幕上的坐标 (x, y)
        """
        gx, gy = gaze_vector[0], gaze_vector[1]
        
        # 非线性变换：使用立方函数增强中心响应，改善边缘精度
        # 这模仿了眼睛在中央视野区域的更细致控制
        gx_nonlinear = np.sign(gx) * (abs(gx) ** 0.9)  # 稍微压缩非线性
        gy_nonlinear = np.sign(gy) * (abs(gy) ** 0.9)
        
        # 映射到屏幕坐标
        # X轴：[-1, 1] -> [0, width]，中心为 0.5 * width
        screen_x = (gx_nonlinear + 1.0) / 2.0 * screen_width
        
        # Y轴：[-0.8, 0.8] -> [0, height]，因为垂直视野范围较小
        # 调整以适应摄像头在屏幕上方的情况
        screen_y = (gy_nonlinear + 0.8) / 1.6 * screen_height
        
        # 应用坐标级别的平滑，防止准心在眼睛停止时还在跳跃
        if self.prev_gaze_coords is not None:
            prev_x, prev_y = self.prev_gaze_coords
            # 加权平均：根据移动距离调整平滑度
            distance = np.sqrt((screen_x - prev_x)**2 + (screen_y - prev_y)**2)
            
            # 如果移动距离很小，增加平滑度（防止抖动）
            if distance < 30:  # 像素
                smooth_factor = 0.6
            elif distance < 100:
                smooth_factor = 0.4
            else:
                smooth_factor = 0.2  # 大幅移动时减少平滑度以保持响应性
            
            screen_x = prev_x * smooth_factor + screen_x * (1 - smooth_factor)
            screen_y = prev_y * smooth_factor + screen_y * (1 - smooth_factor)
        
        self.prev_gaze_coords = (screen_x, screen_y)
        
        # 约束在屏幕范围内，留有安全边距
        screen_x = np.clip(screen_x, 5, screen_width - 5)
        screen_y = np.clip(screen_y, 5, screen_height - 5)
        
        return int(screen_x), int(screen_y)
    
    def detect_left_blink(self, left_eye_region: np.ndarray, frame_id: int = 0) -> bool:
        """
        基于眼睛区域的眨眼检测
        
        Args:
            left_eye_region: 左眼ROI区域
            frame_id: 当前帧ID
            
        Returns:
            是否检测到左眼眨眼
        """
        if left_eye_region is None or left_eye_region.size == 0:
            return False
        
        # 计算左眼的宽高比
        ear = self.calculate_eye_aspect_ratio(left_eye_region)
        self.last_ear_left = ear
        
        # 检测眨眼：EAR下降
        if ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            # 眼睛打开
            if self.blink_counter >= self.BLINK_CONSEC_FRAMES:
                # 检查冷却时间
                if frame_id - self.last_blink_frame > self.blink_cooldown:
                    self.blink_detected = True
                    self.last_blink_frame = frame_id
            self.blink_counter = 0
        
        return self.blink_detected
    
    def detect_right_blink(self, right_eye_region: np.ndarray) -> bool:
        """
        右眼眨眼检测
        
        Args:
            right_eye_region: 右眼ROI区域
            
        Returns:
            是否检测到右眼眨眼
        """
        if right_eye_region is None or right_eye_region.size == 0:
            return False
        
        ear = self.calculate_eye_aspect_ratio(right_eye_region)
        self.last_ear_right = ear
        
        if ear < self.EAR_THRESHOLD:
            return True
        return False
    
    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> Dict:
        """
        处理单帧图像，检测人脸和眼睛，计算视线
        
        Args:
            frame: 输入图像 (H, W, 3) BGR格式
            frame_id: 当前帧ID
            
        Returns:
            包含追踪结果的字典
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        result_dict = {
            'face_detected': False,
            'gaze_vector': None,
            'gaze_screen_coords': None,
            'blink_detected': False,
            'landmarks': None,
            'left_eye_ear': None,
            'right_eye_ear': None,
            'face_confidence': 0.0,
            'blink_triggered': False
        }
        
        try:
            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                # 取第一个检测到的人脸
                (fx, fy, fw, fh) = faces[0]
                result_dict['face_detected'] = True
                result_dict['face_confidence'] = 0.8
                
                # 在人脸区域中检测眼睛
                face_roi = gray[fy:fy + fh, fx:fx + fw]
                eyes = self.eye_cascade.detectMultiScale(
                    face_roi,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(15, 15)
                )
                
                if len(eyes) >= 2:
                    # 按照X坐标排序眼睛（左眼在前）
                    eyes = sorted(eyes, key=lambda e: e[0])
                    
                    # 提取左右眼
                    left_eye_rect_roi = eyes[0]
                    right_eye_rect_roi = eyes[1]
                    
                    # 转换为全图坐标
                    left_eye_rect = (
                        fx + left_eye_rect_roi[0],
                        fy + left_eye_rect_roi[1],
                        left_eye_rect_roi[2],
                        left_eye_rect_roi[3]
                    )
                    right_eye_rect = (
                        fx + right_eye_rect_roi[0],
                        fy + right_eye_rect_roi[1],
                        right_eye_rect_roi[2],
                        right_eye_rect_roi[3]
                    )
                    
                    # 平滑眼睛位置
                    left_eye_rect = self.smooth_eye_position(left_eye_rect, is_left=True)
                    right_eye_rect = self.smooth_eye_position(right_eye_rect, is_left=False)
                    
                    if left_eye_rect is not None and right_eye_rect is not None:
                        # 提取眼睛区域用于EAR计算
                        lx, ly, lw, lh = left_eye_rect
                        rx, ry, rw, rh = right_eye_rect
                        
                        left_eye_region = frame[ly:ly + lh, lx:lx + lw]
                        right_eye_region = frame[ry:ry + rh, rx:rx + rw]
                        
                        # 计算视线向量
                        gaze_vector = self.estimate_gaze_vector(left_eye_rect, right_eye_rect, (fx, fy, fw, fh))
                        result_dict['gaze_vector'] = gaze_vector
                        
                        # 转换为屏幕坐标
                        screen_coords = self.gaze_to_screen_coords(gaze_vector, w, h)
                        result_dict['gaze_screen_coords'] = screen_coords
                        
                        # 检测眨眼
                        blink_state = self.detect_left_blink(left_eye_region, frame_id)
                        result_dict['blink_detected'] = blink_state
                        
                        if blink_state:
                            result_dict['blink_triggered'] = True
                            self.blink_detected = False  # 重置
                        
                        # 获取EAR值
                        self.detect_right_blink(right_eye_region)
                        result_dict['left_eye_ear'] = self.last_ear_left
                        result_dict['right_eye_ear'] = self.last_ear_right
        
        except Exception as e:
            print(f"Error in process_frame: {e}")
            pass
        
        return result_dict
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray = None) -> np.ndarray:
        """
        在图像上绘制眼睛检测区域（用于调试）
        
        Args:
            frame: 输入图像
            landmarks: 面部特征点坐标（此参数在Haar Cascade版本中未使用）
            
        Returns:
            绘制后的图像
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # 绘制人脸矩形
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            
            # 在人脸区域中检测眼睛
            face_roi = gray[fy:fy + fh, fx:fx + fw]
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            
            # 绘制眼睛矩形
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh), (0, 255, 0), 2)
        
        cv2.putText(frame, "OpenCV Haar Cascade Eye Tracking", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
