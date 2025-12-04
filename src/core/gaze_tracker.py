"""
Eye Gaze Tracking Module with ML-based Calibration
使用 OpenCV Haar Cascade 和机器学习模型进行眼球追踪
实现精准的视线到屏幕坐标的映射
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import deque
import os
import joblib

# 加载 Haar Cascade 分类器
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
    """眼动追踪器 - 基于OpenCV Haar Cascade + 机器学习模型"""
    
    def __init__(self, camera_width: int = 1280, camera_height: int = 720, model_path: str = 'gaze_model.pkl'):
        """
        初始化眼动追踪器
        
        Args:
            camera_width: 摄像头宽度
            camera_height: 摄像头高度
            model_path: 预训练模型路径
        """
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.model_path = model_path
        
        if face_cascade is None or eye_cascade is None:
            raise RuntimeError("Failed to load Haar cascades. OpenCV may not be properly installed.")
        
        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade
        
        # 眨眼检测参数
        self.EAR_THRESHOLD = 0.18
        self.BLINK_CONSEC_FRAMES = 2
        self.blink_counter = 0
        self.blink_detected = False
        self.last_blink_frame = -10
        self.blink_cooldown = 5
        
        # 视线向量平滑
        self.gaze_smooth_buffer = deque(maxlen=7)
        self.gaze_smoothing_factor = 0.4
        self.prev_gaze_coords = None
        self.coord_smooth_factor = 0.5
        
        # 校准和模型相关
        self.calibrated = False
        self.gaze_model = None
        self.model_trained = False
        self.training_data = []
        self.training_labels = []
        
        # 校准点配置
        self.calibration_points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]
        self.current_calibration_point = 0
        self.calibrating = False
        self.calibration_samples_per_point = 10
        self.current_point_samples = 0
        
        # 调试信息
        self.last_ear_left = 0.0
        self.last_ear_right = 0.0
        
        # 眼睛位置历史
        self.left_eye_history = deque(maxlen=5)
        self.right_eye_history = deque(maxlen=5)
        self.eye_feature_history = deque(maxlen=10)
        
        # 保存眼睛矩形用于特征提取
        self.last_left_eye_rect = None
        self.last_right_eye_rect = None
        self.last_face_rect = None
        
        # 尝试加载预训练模型
        self._load_model()
    
    def _load_model(self):
        """加载预训练的模型"""
        if os.path.exists(self.model_path):
            try:
                self.gaze_model = joblib.load(self.model_path)
                self.model_trained = True
                print(f"[*] Model loaded: {self.model_path}")
            except Exception as e:
                print(f"[!] Failed to load model {self.model_path}: {e}")
                self.model_trained = False
        else:
            print(f"[i] Model not found {self.model_path}, using geometric method")
    
    def _save_model(self):
        """保存训练好的模型"""
        if self.gaze_model is not None:
            joblib.dump(self.gaze_model, self.model_path)
            print(f"✓ 模型已保存：{self.model_path}")
    
    def extract_eye_features(self, left_eye_rect: Tuple, right_eye_rect: Tuple, face_rect: Tuple) -> Optional[np.ndarray]:
        """
        提取眼睛特征用于模型训练和预测
        
        Args:
            left_eye_rect: 左眼矩形 (x, y, w, h)
            right_eye_rect: 右眼矩形 (x, y, w, h)
            face_rect: 脸部矩形 (x, y, w, h)
            
        Returns:
            眼睛特征向量
        """
        if not all([left_eye_rect, right_eye_rect, face_rect]):
            return None
        
        lx, ly, lw, lh = left_eye_rect
        rx, ry, rw, rh = right_eye_rect
        fx, fy, fw, fh = face_rect
        
        features = []
        
        # 1. 眼睛中心位置（相对脸部）
        left_center_x = (lx + lw / 2 - fx) / fw
        left_center_y = (ly + lh / 2 - fy) / fh
        right_center_x = (rx + rw / 2 - fx) / fw
        right_center_y = (ry + rh / 2 - fy) / fh
        
        features.extend([left_center_x, left_center_y, right_center_x, right_center_y])
        
        # 2. 眼睛大小（相对脸部大小）
        left_size = (lw * lh) / (fw * fh)
        right_size = (rw * rh) / (fw * fh)
        features.extend([left_size, right_size])
        
        # 3. 眼睛宽高比
        left_aspect = lw / max(lh, 1)
        right_aspect = rw / max(rh, 1)
        features.extend([left_aspect, right_aspect])
        
        # 4. 两眼间距
        eye_distance = np.sqrt((left_center_x - right_center_x) ** 2 + 
                              (left_center_y - right_center_y) ** 2)
        features.append(eye_distance)
        
        # 5. 历史特征（平均和方差）
        if len(self.eye_feature_history) > 0:
            history_array = np.array(self.eye_feature_history)
            mean_features = np.mean(history_array, axis=0)
            std_features = np.std(history_array, axis=0)
            features.extend(mean_features.tolist())
            features.extend(std_features.tolist())
        else:
            features.extend([0] * (len(features) * 2))
        
        # 保存基础特征到历史
        self.eye_feature_history.append(features[:9])
        
        return np.array(features, dtype=np.float32)
    
    def start_calibration(self):
        """开始校准流程"""
        print("\n" + "="*60)
        print("开始眼动校准 - 请依次看屏幕上出现的9个点")
        print("每个点显示时请尽量凝视，系统将自动采集样本")
        print("="*60 + "\n")
        
        self.calibrating = True
        self.current_calibration_point = 0
        self.current_point_samples = 0
        self.training_data = []
        self.training_labels = []
    
    def add_calibration_sample(self, eye_features: np.ndarray, screen_width: int, screen_height: int) -> bool:
        """添加校准样本"""
        if not self.calibrating or eye_features is None:
            return False
        
        if self.current_calibration_point >= len(self.calibration_points):
            return False
        
        # 获取当前校准点的目标坐标
        x_ratio, y_ratio = self.calibration_points[self.current_calibration_point]
        target_x = x_ratio * screen_width
        target_y = y_ratio * screen_height
        
        # 添加样本
        self.training_data.append(eye_features)
        self.training_labels.append([target_x, target_y])
        
        self.current_point_samples += 1
        
        # 检查是否完成当前点的采集
        if self.current_point_samples >= self.calibration_samples_per_point:
            print(f"✓ 点 {self.current_calibration_point + 1}/{len(self.calibration_points)} 采集完成")
            self.current_calibration_point += 1
            self.current_point_samples = 0
            
            # 检查是否完成所有点
            if self.current_calibration_point >= len(self.calibration_points):
                print("\n✓ 校准数据采集完成！")
                return self.train_gaze_model()
        
        return True
    
    def train_gaze_model(self) -> bool:
        """训练视线映射模型"""
        min_samples = len(self.calibration_points) * self.calibration_samples_per_point
        
        if len(self.training_data) < min_samples:
            print(f"⚠ 样本不足: 需要{min_samples}个，仅有{len(self.training_data)}个")
            return False
        
        try:
            from sklearn.neural_network import MLPRegressor
            
            X = np.array(self.training_data)
            y = np.array(self.training_labels)
            
            # 使用神经网络模型
            self.gaze_model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=2000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=100
            )
            
            self.gaze_model.fit(X, y)
            self.model_trained = True
            
            # 评估模型
            train_score = self.gaze_model.score(X, y)
            print(f"[*] Model trained successfully")
            print(f"  - Samples: {len(X)}")
            print(f"  - Accuracy: {train_score:.4f}")
            
            # 保存模型
            self._save_model()
            
            self.calibrating = False
            return True
            
        except ImportError:
            print("⚠ sklearn 未安装，请运行: pip install scikit-learn")
            return False
        except Exception as e:
            print(f"✗ 模型训练失败: {e}")
            return False
    
    def predict_gaze_with_model(self, eye_features: np.ndarray, screen_width: int, screen_height: int) -> Optional[Tuple[int, int]]:
        """使用训练好的模型预测视线位置"""
        if not self.model_trained or self.gaze_model is None or eye_features is None:
            return None
        
        try:
            X = np.array([eye_features])
            prediction = self.gaze_model.predict(X)[0]
            
            # 约束到屏幕范围
            screen_x = int(np.clip(prediction[0], 0, screen_width))
            screen_y = int(np.clip(prediction[1], 0, screen_height))
            
            return (screen_x, screen_y)
            
        except Exception as e:
            print(f"预测错误: {e}")
            return None
    
    def calculate_eye_aspect_ratio(self, eye_region: np.ndarray) -> float:
        """基于眼睛区域计算眼睛宽高比"""
        if eye_region is None or eye_region.size == 0:
            return 1.0
        
        h, w = eye_region.shape[:2]
        if h == 0 or w == 0:
            return 1.0
        
        # 计算区域的亮度平均值
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        brightness = np.mean(gray) / 255.0
        
        # 计算宽高比
        aspect_ratio = min(w, h) / max(w, h, 1)
        
        # 综合考虑宽高比和亮度
        ear = brightness * aspect_ratio
        return ear
    
    def smooth_eye_position(self, eye_rect: Tuple, is_left: bool = True) -> Optional[Tuple]:
        """平滑眼睛位置"""
        if eye_rect is None:
            return None
        
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
        """几何方法估计视线向量（备选方案）"""
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
        
        # 计算相对位置
        max_horizontal_offset = face_rect[2] / 3.0
        max_vertical_offset = face_rect[3] / 4.0
        
        gaze_x = (avg_eye_x - face_center_x) / max_horizontal_offset
        gaze_y = (avg_eye_y - face_center_y) / max_vertical_offset
        
        # 约束范围
        gaze_x = np.clip(gaze_x, -1.0, 1.0)
        gaze_y = np.clip(gaze_y, -0.8, 0.8)
        
        gaze_vector = np.array([gaze_x, gaze_y], dtype=np.float32)
        
        # 应用平滑滤波
        gaze_vector = self._smooth_gaze_vector(gaze_vector)
        
        return gaze_vector
    
    def _smooth_gaze_vector(self, gaze_vector: np.ndarray) -> np.ndarray:
        """平滑视线向量"""
        self.gaze_smooth_buffer.append(gaze_vector)
        
        if len(self.gaze_smooth_buffer) < 2:
            return gaze_vector
        
        smoothed = np.mean(list(self.gaze_smooth_buffer), axis=0)
        return smoothed
    
    def gaze_to_screen_coords(self, gaze_vector: np.ndarray, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """将视线向量映射到屏幕坐标（几何方法）"""
        gx, gy = gaze_vector[0], gaze_vector[1]
        
        # 非线性变换
        gx_nonlinear = np.sign(gx) * (abs(gx) ** 0.9)
        gy_nonlinear = np.sign(gy) * (abs(gy) ** 0.9)
        
        # 映射到屏幕坐标
        screen_x = (gx_nonlinear + 1.0) / 2.0 * screen_width
        screen_y = (gy_nonlinear + 0.8) / 1.6 * screen_height
        
        # 应用坐标级别的平滑
        if self.prev_gaze_coords is not None:
            prev_x, prev_y = self.prev_gaze_coords
            distance = np.sqrt((screen_x - prev_x)**2 + (screen_y - prev_y)**2)
            
            if distance < 30:
                smooth_factor = 0.6
            elif distance < 100:
                smooth_factor = 0.4
            else:
                smooth_factor = 0.2
            
            screen_x = prev_x * smooth_factor + screen_x * (1 - smooth_factor)
            screen_y = prev_y * smooth_factor + screen_y * (1 - smooth_factor)
        
        self.prev_gaze_coords = (screen_x, screen_y)
        
        # 约束在屏幕范围内
        screen_x = np.clip(screen_x, 5, screen_width - 5)
        screen_y = np.clip(screen_y, 5, screen_height - 5)
        
        return int(screen_x), int(screen_y)
    
    def detect_left_blink(self, left_eye_region: np.ndarray, frame_id: int = 0) -> bool:
        """检测左眼眨眼"""
        if left_eye_region is None or left_eye_region.size == 0:
            return False
        
        ear = self.calculate_eye_aspect_ratio(left_eye_region)
        self.last_ear_left = ear
        
        if ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.BLINK_CONSEC_FRAMES:
                if frame_id - self.last_blink_frame > self.blink_cooldown:
                    self.blink_detected = True
                    self.last_blink_frame = frame_id
            self.blink_counter = 0
        
        return self.blink_detected
    
    def detect_right_blink(self, right_eye_region: np.ndarray) -> bool:
        """检测右眼眨眼"""
        if right_eye_region is None or right_eye_region.size == 0:
            return False
        
        ear = self.calculate_eye_aspect_ratio(right_eye_region)
        self.last_ear_right = ear
        
        if ear < self.EAR_THRESHOLD:
            return True
        return False
    
    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> Dict:
        """处理单帧图像"""
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
            'blink_triggered': False,
            'calibrating': self.calibrating,
            'model_trained': self.model_trained
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
                
                self.last_face_rect = (fx, fy, fw, fh)
                
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
                        # 保存眼睛矩形
                        self.last_left_eye_rect = left_eye_rect
                        self.last_right_eye_rect = right_eye_rect
                        
                        # 提取眼睛区域
                        lx, ly, lw, lh = left_eye_rect
                        rx, ry, rw, rh = right_eye_rect
                        
                        left_eye_region = frame[ly:ly + lh, lx:lx + lw]
                        right_eye_region = frame[ry:ry + rh, rx:rx + rw]
                        
                        # 提取眼睛特征
                        eye_features = self.extract_eye_features(left_eye_rect, right_eye_rect, (fx, fy, fw, fh))
                        
                        # 校准模式
                        if self.calibrating:
                            self.add_calibration_sample(eye_features, w, h)
                        
                        # 使用模型预测或几何方法
                        if self.model_trained and eye_features is not None:
                            # 使用ML模型预测
                            screen_coords = self.predict_gaze_with_model(eye_features, w, h)
                            
                            if screen_coords:
                                result_dict['gaze_screen_coords'] = screen_coords
                                # 从屏幕坐标反推视线向量
                                gaze_x = (screen_coords[0] / w * 2) - 1
                                gaze_y = (screen_coords[1] / h * 2) - 1
                                result_dict['gaze_vector'] = np.array([gaze_x, gaze_y], dtype=np.float32)
                        else:
                            # 使用几何方法（备选）
                            gaze_vector = self.estimate_gaze_vector(left_eye_rect, right_eye_rect, (fx, fy, fw, fh))
                            result_dict['gaze_vector'] = gaze_vector
                            
                            screen_coords = self.gaze_to_screen_coords(gaze_vector, w, h)
                            result_dict['gaze_screen_coords'] = screen_coords
                        
                        # 检测眨眼
                        blink_state = self.detect_left_blink(left_eye_region, frame_id)
                        result_dict['blink_detected'] = blink_state
                        
                        if blink_state:
                            result_dict['blink_triggered'] = True
                            self.blink_detected = False
                        
                        # 获取EAR值
                        self.detect_right_blink(right_eye_region)
                        result_dict['left_eye_ear'] = self.last_ear_left
                        result_dict['right_eye_ear'] = self.last_ear_right
        
        except Exception as e:
            print(f"Error in process_frame: {e}")
        
        return result_dict
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray = None) -> np.ndarray:
        """绘制眼睛检测区域"""
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
        
        # 显示状态
        status_text = "ML Model" if self.model_trained else "Geometric Method"
        cv2.putText(frame, f"Eye Tracking - {status_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.calibrating:
            calib_text = f"CALIBRATING: Point {self.current_calibration_point + 1}/{len(self.calibration_points)}"
            cv2.putText(frame, calib_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
