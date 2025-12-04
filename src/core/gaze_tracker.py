"""
Eye Gaze Tracking Module
使用 MediaPipe Face Mesh 进行人脸检测和眼睛追踪
计算视线向量（gaze vector）和眨眼检测
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, Dict


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
        
        # MediaPipe Face Mesh 初始化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 关键点索引 (Face Mesh Landmark Indices)
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # 左眼关键点
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # 右眼关键点
        
        # 眨眼检测阈值
        self.EAR_THRESHOLD = 0.2  # Eye Aspect Ratio threshold
        self.BLINK_CONSEC_FRAMES = 3  # 连续帧数
        self.blink_counter = 0
        self.blink_detected = False
        
        # 校准参数
        self.calibrated = False
        self.calibration_points = []
        self.screen_points = []
        
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
        估计视线向量（基于瞳孔位置和眼睛中心）
        
        Args:
            face_landmarks: 全部面部特征点 (468, 2)
            left_eye_indices: 左眼关键点索引
            right_eye_indices: 右眼关键点索引
            
        Returns:
            归一化的视线向量 (2,)
        """
        # 获取左右眼的瞳孔位置
        left_eye = face_landmarks[left_eye_indices]
        right_eye = face_landmarks[right_eye_indices]
        
        left_pupil = self.get_pupil_position(left_eye)
        right_pupil = self.get_pupil_position(right_eye)
        
        # 计算眼睛内角和外角
        left_inner = face_landmarks[left_eye_indices[0]]  # 眼内角
        left_outer = face_landmarks[left_eye_indices[3]]  # 眼外角
        right_inner = face_landmarks[right_eye_indices[0]]
        right_outer = face_landmarks[right_eye_indices[3]]
        
        # 计算瞳孔相对于眼睛的位置比例
        left_eye_width = np.linalg.norm(left_outer - left_inner)
        left_gaze_x = (left_pupil[0] - left_inner[0]) / (left_eye_width + 1e-6)
        
        right_eye_width = np.linalg.norm(right_outer - right_inner)
        right_gaze_x = (right_pupil[0] - right_inner[0]) / (right_eye_width + 1e-6)
        
        # 平均左右眼的视线向量
        avg_gaze_x = (left_gaze_x + right_gaze_x) / 2.0
        avg_gaze_y = (left_pupil[1] + right_pupil[1]) / 2.0
        
        # 相对于图像高度的归一化
        avg_gaze_y = avg_gaze_y / self.camera_height
        
        # 约束在 [-1, 1] 范围内
        avg_gaze_x = np.clip(avg_gaze_x, -1.0, 2.0)
        avg_gaze_y = np.clip(avg_gaze_y, -0.5, 1.5)
        
        return np.array([avg_gaze_x, avg_gaze_y])
    
    def gaze_to_screen_coords(self, 
                             gaze_vector: np.ndarray,
                             screen_width: int,
                             screen_height: int) -> Tuple[int, int]:
        """
        将视线向量映射到屏幕坐标
        
        Args:
            gaze_vector: 归一化的视线向量 (2,)
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            
        Returns:
            屏幕上的坐标 (x, y)
        """
        # 简单的线性映射：[-1, 2] -> [0, width]，[-0.5, 1.5] -> [0, height]
        screen_x = int((gaze_vector[0] + 1.0) / 3.0 * screen_width)
        screen_y = int((gaze_vector[1] + 0.5) / 2.0 * screen_height)
        
        # 约束在屏幕范围内
        screen_x = np.clip(screen_x, 0, screen_width - 1)
        screen_y = np.clip(screen_y, 0, screen_height - 1)
        
        return screen_x, screen_y
    
    def detect_left_blink(self, face_landmarks: np.ndarray) -> bool:
        """
        检测左眼眨眼
        
        Args:
            face_landmarks: 面部特征点 (468, 2)
            
        Returns:
            是否检测到左眼眨眼
        """
        left_eye = face_landmarks[self.LEFT_EYE_INDICES]
        ear = self.calculate_eye_aspect_ratio(left_eye)
        
        if ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.BLINK_CONSEC_FRAMES:
                self.blink_detected = True
            self.blink_counter = 0
        
        return self.blink_detected
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        处理单帧图像，检测人脸和计算视线
        
        Args:
            frame: 输入图像 (H, W, 3)
            
        Returns:
            包含追踪结果的字典
            {
                'face_detected': bool,
                'gaze_vector': np.ndarray (2,) 或 None,
                'gaze_screen_coords': tuple (x, y) 或 None,
                'blink_detected': bool,
                'landmarks': np.ndarray (468, 2) 或 None
            }
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
            'right_eye_ear': None
        }
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # 转换为像素坐标
            face_landmarks = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
            
            result_dict['face_detected'] = True
            result_dict['landmarks'] = face_landmarks
            
            # 计算视线向量
            gaze_vector = self.estimate_gaze_vector(
                face_landmarks,
                self.LEFT_EYE_INDICES,
                self.RIGHT_EYE_INDICES
            )
            result_dict['gaze_vector'] = gaze_vector
            
            # 计算眼睛宽高比
            left_ear = self.calculate_eye_aspect_ratio(face_landmarks[self.LEFT_EYE_INDICES])
            right_ear = self.calculate_eye_aspect_ratio(face_landmarks[self.RIGHT_EYE_INDICES])
            result_dict['left_eye_ear'] = left_ear
            result_dict['right_eye_ear'] = right_ear
            
            # 检测左眼眨眼（开火动作）
            if self.detect_left_blink(face_landmarks):
                result_dict['blink_detected'] = True
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
