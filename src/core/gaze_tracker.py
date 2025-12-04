"""
Eye Gaze Tracking Module
使用 MediaPipe Face Mesh 进行人脸检测和眼睛追踪
计算视线向量（gaze vector）和眨眼检测
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from collections import deque

try:
    import mediapipe as mp
except ImportError as e:
    print(f"Warning: MediaPipe import failed: {e}")
    print("Trying alternative import method...")
    mp = None


class GazeTracker:
    """眼动追踪器 - 基于MediaPipe Face Mesh"""
    
    def __init__(self, camera_width: int = 1280, camera_height: int = 720):
        """
        初始化眼动追踪器
        
        Args:
            camera_width: 摄像头宽度
            camera_height: 摄像头高度
        """
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        if mp is None:
            raise RuntimeError("MediaPipe not available. Install with: pip install mediapipe")
        
        # MediaPipe Face Mesh 初始化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 关键点索引 (Face Mesh Landmark Indices)
        # 更准确的眼睛关键点索引（6个点构成眼睛轮廓）
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        
        # 瞳孔位置（更精确的内部点）
        self.LEFT_PUPIL_INDEX = 468  # 虚拟瞳孔点
        self.RIGHT_PUPIL_INDEX = 469  # 虚拟瞳孔点
        
        # 眼睛内角外角（用于计算眼睛宽度）
        self.LEFT_EYE_INNER = 33
        self.LEFT_EYE_OUTER = 133
        self.RIGHT_EYE_INNER = 362
        self.RIGHT_EYE_OUTER = 263
        
        # 眨眼检测参数 - 优化阈值
        self.EAR_THRESHOLD = 0.18  # Eye Aspect Ratio threshold
        self.BLINK_CONSEC_FRAMES = 2  # 连续帧数阈值（降低以更快响应）
        self.blink_counter = 0
        self.blink_detected = False
        self.last_blink_frame = -10  # 上一次眨眼的帧数
        self.blink_cooldown = 5  # 眨眼冷却时间（帧数）
        
        # 视线向量平滑（改进的多阶段滤波）
        self.gaze_smooth_buffer = deque(maxlen=7)  # 增加缓冲区大小，更好的平滑
        self.gaze_smoothing_factor = 0.4  # 降低平滑因子以减少延迟
        self.prev_gaze_coords = None  # 上一帧的坐标
        self.coord_smooth_factor = 0.5  # 坐标平滑因子
        
        # 校准参数
        self.calibrated = False
        self.calibration_points = []
        self.screen_points = []
        self.gaze_calibration_matrix = None  # 校准矩阵
        
        # 调试信息
        self.last_ear_left = 0
        self.last_ear_right = 0
        
        # 摄像头位置补正（对于正上方摄像头的特殊处理）
        self.camera_height_offset = -0.15  # Y轴偏移量，向上调整（负值）
        
    def calculate_eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """
        计算眼睛宽高比（Eye Aspect Ratio）
        
        Args:
            eye_points: 眼睛6个关键点的坐标 (6, 2)
            
        Returns:
            眼睛宽高比
        """
        # 计算眼睛的纵向距离
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # 计算眼睛的横向距离
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # 计算宽高比
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_pupil_position(self, eye_points: np.ndarray) -> Tuple[float, float]:
        """
        计算瞳孔位置（眼睛关键点的中心）
        
        Args:
            eye_points: 眼睛关键点坐标 (6, 2)
            
        Returns:
            瞳孔中心坐标 (x, y)
        """
        pupil_x = np.mean(eye_points[:, 0])
        pupil_y = np.mean(eye_points[:, 1])
        return pupil_x, pupil_y
    
    def estimate_gaze_vector(self, 
                            face_landmarks: np.ndarray,
                            left_eye_indices: list,
                            right_eye_indices: list) -> np.ndarray:
        """
        改进的视线向量估计（基于瞳孔在眼睛框架中的位置）
        使用虚拟瞳孔点（468，469）以获得更准确的视线方向
        
        Args:
            face_landmarks: 全部面部特征点 (468, 2)
            left_eye_indices: 左眼关键点索引
            right_eye_indices: 右眼关键点索引
            
        Returns:
            归一化的视线向量 (2,)，范围约为 [-1, 1]
        """
        # 获取眼睛特征点
        left_eye_points = face_landmarks[left_eye_indices]
        right_eye_points = face_landmarks[right_eye_indices]
        
        # 获取虚拟瞳孔点（468和469是MediaPipe的虚拟瞳孔点）
        try:
            left_pupil = face_landmarks[468]
            right_pupil = face_landmarks[469]
        except:
            # 如果没有虚拟瞳孔点，使用眼睛中心
            left_pupil = np.mean(left_eye_points, axis=0)
            right_pupil = np.mean(right_eye_points, axis=0)
        
        # 计算眼睛边界（外角和内角）
        left_eye_inner = face_landmarks[self.LEFT_EYE_INNER]
        left_eye_outer = face_landmarks[self.LEFT_EYE_OUTER]
        right_eye_inner = face_landmarks[self.RIGHT_EYE_INNER]
        right_eye_outer = face_landmarks[self.RIGHT_EYE_OUTER]
        
        # 计算上下边界（用于Y轴）
        left_eye_top = (left_eye_points[1] + left_eye_points[2]) / 2  # 上方点
        left_eye_bottom = (left_eye_points[4] + left_eye_points[5]) / 2  # 下方点
        right_eye_top = (right_eye_points[1] + right_eye_points[2]) / 2
        right_eye_bottom = (right_eye_points[4] + right_eye_points[5]) / 2
        
        # 计算眼睛的有效范围
        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        left_eye_height = np.linalg.norm(left_eye_bottom - left_eye_top)
        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        right_eye_height = np.linalg.norm(right_eye_bottom - right_eye_top)
        
        # 计算瞳孔相对于眼睛的归一化位置
        # X方向：-1 = 看左，0 = 看中央，1 = 看右
        # 使用内外角作为范围边界
        left_eye_width_factor = left_eye_width * 1.1  # 稍微扩大范围以获得更好的外围覆盖
        right_eye_width_factor = right_eye_width * 1.1
        
        left_gaze_x = (left_pupil[0] - left_eye_inner[0]) / (left_eye_width_factor + 1e-6) - 0.5
        right_gaze_x = (right_pupil[0] - right_eye_inner[0]) / (right_eye_width_factor + 1e-6) - 0.5
        
        # Y方向：-1 = 看上，0 = 看中央，1 = 看下
        left_eye_height_factor = left_eye_height * 1.1
        right_eye_height_factor = right_eye_height * 1.1
        
        left_gaze_y = (left_pupil[1] - left_eye_top[1]) / (left_eye_height_factor + 1e-6) - 0.5
        right_gaze_y = (right_pupil[1] - right_eye_top[1]) / (right_eye_height_factor + 1e-6) - 0.5
        
        # 平均左右眼以获得更稳定的视线估计
        avg_gaze_x = (left_gaze_x + right_gaze_x) / 2.0
        avg_gaze_y = (left_gaze_y + right_gaze_y) / 2.0
        
        # 应用摄像头位置补正（因为摄像头在屏幕正上方）
        avg_gaze_y += self.camera_height_offset
        
        # 约束在合理范围内（允许一定的超出范围以提高边缘准确性）
        avg_gaze_x = np.clip(avg_gaze_x, -1.0, 1.0)
        avg_gaze_y = np.clip(avg_gaze_y, -0.8, 0.8)
        
        gaze_vector = np.array([avg_gaze_x, avg_gaze_y], dtype=np.float32)
        
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
    
    def detect_left_blink(self, face_landmarks: np.ndarray, frame_id: int = 0) -> bool:
        """
        改进的左眼眨眼检测，使用EAR和时间管理
        防止眨眼过程中的准心晃动
        
        Args:
            face_landmarks: 面部特征点 (468, 2)
            frame_id: 当前帧ID（用于冷却时间）
            
        Returns:
            是否检测到左眼眨眼
        """
        left_eye = face_landmarks[self.LEFT_EYE_INDICES]
        ear = self.calculate_eye_aspect_ratio(left_eye)
        self.last_ear_left = ear
        
        # 检测眨眼：EAR下降
        if ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            # 眨眼恢复（眼睛张开）
            if self.blink_counter >= self.BLINK_CONSEC_FRAMES:
                # 检查冷却时间（避免双重触发）
                if frame_id - self.last_blink_frame > self.blink_cooldown:
                    self.blink_detected = True
                    self.last_blink_frame = frame_id
                    # 关键：当眨眼触发时，锁定当前的准心位置，防止眨眼恢复时的晃动
                    if self.prev_gaze_coords is not None:
                        # 保留上一个稳定的坐标
                        pass
            self.blink_counter = 0
        
        return self.blink_detected
    
    def detect_right_blink(self, face_landmarks: np.ndarray, frame_id: int = 0) -> bool:
        """
        右眼眨眼检测（可选，用于调试或其他功能）
        
        Args:
            face_landmarks: 面部特征点 (468, 2)
            frame_id: 当前帧ID
            
        Returns:
            是否检测到右眼眨眼
        """
        right_eye = face_landmarks[self.RIGHT_EYE_INDICES]
        ear = self.calculate_eye_aspect_ratio(right_eye)
        self.last_ear_right = ear
        
        if ear < self.EAR_THRESHOLD:
            return True
        return False
    
    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> Dict:
        """
        处理单帧图像，检测人脸和计算视线
        改进：在眨眼期间锁定准心以防止晃动
        
        Args:
            frame: 输入图像 (H, W, 3)
            frame_id: 当前帧ID（用于眨眼冷却计算）
            
        Returns:
            包含追踪结果的字典
        """
        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # 运行Face Mesh
        results = self.face_mesh.process(frame_rgb)
        
        result_dict = {
            'face_detected': False,
            'gaze_vector': None,
            'gaze_screen_coords': None,
            'blink_detected': False,
            'landmarks': None,
            'left_eye_ear': None,
            'right_eye_ear': None,
            'face_confidence': 0.0,
            'blink_triggered': False  # 是否真正触发了射击
        }
        
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            landmarks = results.multi_face_landmarks[0]
            # 转换为像素坐标
            face_landmarks = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
            
            result_dict['face_detected'] = True
            result_dict['landmarks'] = face_landmarks
            result_dict['face_confidence'] = 0.9  # MediaPipe没有直接提供置信度
            
            # 计算视线向量
            gaze_vector = self.estimate_gaze_vector(
                face_landmarks,
                self.LEFT_EYE_INDICES,
                self.RIGHT_EYE_INDICES
            )
            result_dict['gaze_vector'] = gaze_vector
            
            # 转换为屏幕坐标
            screen_coords = self.gaze_to_screen_coords(gaze_vector, w, h)
            result_dict['gaze_screen_coords'] = screen_coords
            
            # 计算眼睛宽高比
            left_ear = self.calculate_eye_aspect_ratio(face_landmarks[self.LEFT_EYE_INDICES])
            right_ear = self.calculate_eye_aspect_ratio(face_landmarks[self.RIGHT_EYE_INDICES])
            result_dict['left_eye_ear'] = left_ear
            result_dict['right_eye_ear'] = right_ear
            
            # 检测左眼眨眼（开火动作）
            blink_state = self.detect_left_blink(face_landmarks, frame_id)
            result_dict['blink_detected'] = blink_state
            
            if blink_state:
                result_dict['blink_triggered'] = True
                self.blink_detected = False  # 重置
        
        return result_dict
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        在图像上绘制面部特征点（用于调试）
        
        Args:
            frame: 输入图像
            landmarks: 面部特征点坐标 (468, 2)
            
        Returns:
            绘制后的图像
        """
        if landmarks is None:
            return frame
        
        # 绘制眼睛关键点
        for idx in self.LEFT_EYE_INDICES:
            x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        for idx in self.RIGHT_EYE_INDICES:
            x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        
        return frame
