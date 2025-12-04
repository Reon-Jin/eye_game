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
        
        # 视线向量平滑（低通滤波）
        self.gaze_smooth_buffer = deque(maxlen=3)  # 历史缓冲
        self.gaze_smoothing_factor = 0.6  # 平滑因子 (0-1)
        
        # 校准参数
        self.calibrated = False
        self.calibration_points = []
        self.screen_points = []
        self.gaze_calibration_matrix = None  # 校准矩阵
        
        # 调试信息
        self.last_ear_left = 0
        self.last_ear_right = 0
        
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
        估计视线向量（改进的算法 - 基于瞳孔相对位置）
        
        Args:
            face_landmarks: 全部面部特征点 (468, 2)
            left_eye_indices: 左眼关键点索引
            right_eye_indices: 右眼关键点索引
            
        Returns:
            归一化的视线向量 (2,)
        """
        # 获取眼睛特征点
        left_eye_points = face_landmarks[left_eye_indices]
        right_eye_points = face_landmarks[right_eye_indices]
        
        # 计算眼睛边界
        left_eye_inner = face_landmarks[self.LEFT_EYE_INNER]
        left_eye_outer = face_landmarks[self.LEFT_EYE_OUTER]
        right_eye_inner = face_landmarks[self.RIGHT_EYE_INNER]
        right_eye_outer = face_landmarks[self.RIGHT_EYE_OUTER]
        
        # 计算眼睛中心
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)
        
        # 计算眼睛宽度（用于归一化）
        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        
        # 计算眼睛高度
        left_eye_height = abs(left_eye_points[1, 1] - left_eye_points[4, 1])
        right_eye_height = abs(right_eye_points[1, 1] - right_eye_points[4, 1])
        
        # 计算瞳孔位置（眼睛内部的黑点）
        # 使用眼睛内部点的平均值作为瞳孔估计
        left_pupil = np.mean([left_eye_points[0], left_eye_points[1], 
                             left_eye_points[4], left_eye_points[5]], axis=0)
        right_pupil = np.mean([right_eye_points[0], right_eye_points[1], 
                              right_eye_points[4], right_eye_points[5]], axis=0)
        
        # 计算瞳孔相对于眼睛内外角的位置比例
        # X方向：-1为看左，1为看右
        left_gaze_x = (left_pupil[0] - left_eye_inner[0]) / (left_eye_width + 1e-6)
        right_gaze_x = (right_pupil[0] - right_eye_inner[0]) / (right_eye_width + 1e-6)
        
        # Y方向：-1为看上，1为看下
        left_gaze_y = (left_pupil[1] - (left_eye_points[1, 1] + left_eye_points[4, 1]) / 2) / (left_eye_height + 1e-6)
        right_gaze_y = (right_pupil[1] - (right_eye_points[1, 1] + right_eye_points[4, 1]) / 2) / (right_eye_height + 1e-6)
        
        # 平均左右眼
        avg_gaze_x = (left_gaze_x + right_gaze_x) / 2.0
        avg_gaze_y = (left_gaze_y + right_gaze_y) / 2.0
        
        # 约束在合理范围内
        avg_gaze_x = np.clip(avg_gaze_x, -1.2, 1.2)
        avg_gaze_y = np.clip(avg_gaze_y, -0.8, 0.8)
        
        gaze_vector = np.array([avg_gaze_x, avg_gaze_y])
        
        # 应用平滑
        gaze_vector = self._smooth_gaze_vector(gaze_vector)
        
        return gaze_vector
    
    def _smooth_gaze_vector(self, gaze_vector: np.ndarray) -> np.ndarray:
        """
        使用低通滤波平滑视线向量
        
        Args:
            gaze_vector: 当前视线向量
            
        Returns:
            平滑后的视线向量
        """
        self.gaze_smooth_buffer.append(gaze_vector)
        
        if len(self.gaze_smooth_buffer) < 2:
            return gaze_vector
        
        # 简单的移动平均
        smoothed = np.mean(list(self.gaze_smooth_buffer), axis=0)
        
        return smoothed
    
    def gaze_to_screen_coords(self, 
                             gaze_vector: np.ndarray,
                             screen_width: int,
                             screen_height: int) -> Tuple[int, int]:
        """
        将视线向量映射到屏幕坐标（改进版）
        
        Args:
            gaze_vector: 归一化的视线向量 (2,) 范围 [-1.2, 1.2] x [-0.8, 0.8]
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            
        Returns:
            屏幕上的坐标 (x, y)
        """
        # 改进的非线性映射，使得准心跟踪更准确
        # X轴映射：[-1.2, 1.2] -> [0, width]
        screen_x = (gaze_vector[0] + 1.2) / 2.4 * screen_width
        
        # Y轴映射：[-0.8, 0.8] -> [0, height]
        screen_y = (gaze_vector[1] + 0.8) / 1.6 * screen_height
        
        # 应用非线性变换以改善边缘区域的准确性
        # screen_x = np.sign(screen_x - screen_width/2) * (abs(screen_x - screen_width/2) ** 0.95) + screen_width/2
        # screen_y = np.sign(screen_y - screen_height/2) * (abs(screen_y - screen_height/2) ** 0.95) + screen_height/2
        
        # 约束在屏幕范围内
        screen_x = np.clip(screen_x, 0, screen_width - 1)
        screen_y = np.clip(screen_y, 0, screen_height - 1)
        
        return int(screen_x), int(screen_y)
    
    def detect_left_blink(self, face_landmarks: np.ndarray, frame_id: int = 0) -> bool:
        """
        改进的左眼眨眼检测（使用EAR和时间限制）
        
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
            # 眨眼恢复
            if self.blink_counter >= self.BLINK_CONSEC_FRAMES:
                # 检查冷却时间（避免双重触发）
                if frame_id - self.last_blink_frame > self.blink_cooldown:
                    self.blink_detected = True
                    self.last_blink_frame = frame_id
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
