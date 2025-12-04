"""
Eye Gaze Tracking Module with ML-based Calibration
使用 OpenCV Haar Cascade 和机器学习模型进行眼球追踪
实现视线到屏幕坐标的映射
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from collections import deque
import os
import joblib
import random


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


# ===== 集成模型 - 定义为顶级类以支持 pickle =====
class EnsembleModel:
    """集成预测模型 - MLPRegressor + Ridge 回归"""

    def __init__(self, m1, m2, scaler):
        # m1: MLPRegressor, m2: Ridge, scaler: 特征缩放器
        self.m1 = m1
        self.m2 = m2
        self.scaler = scaler

    def predict(self, X):
        # 确保 X 是 numpy 数组
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)

        # 处理 1D 数组 -> 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)
        pred1 = self.m1.predict(X_scaled)
        final_pred = self.m2.predict(pred1)  # 直接输出像素坐标（x, y）
        return final_pred

    def score(self, X, y):
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot)


class GazeTracker:
    """眼动追踪器 - 基于 OpenCV Haar Cascade + 机器学习模型"""

    def __init__(self, camera_width: int = 1280, camera_height: int = 720,
                 model_path: str = 'gaze_model.pkl'):
        """
        初始化眼动追踪器

        Args:
            camera_width: 摄像头宽度
            camera_height: 摄像头高度
            model_path: 模型保存/读取路径
        """
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.model_path = model_path

        if face_cascade is None or eye_cascade is None:
            raise RuntimeError("Failed to load Haar cascades. OpenCV may not be properly installed.")

        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade

        # ---------- 眨眼检测参数 ----------
        self.EAR_THRESHOLD = 0.18
        self.BLINK_CONSEC_FRAMES = 2
        self.blink_counter = 0
        self.blink_detected = False
        self.last_blink_frame = -10
        self.blink_cooldown = 5

        # ---------- 视线平滑 ----------
        self.gaze_smooth_buffer = deque(maxlen=7)
        self.prev_gaze_coords = None

        # ---------- 校准 & 模型 ----------
        self.calibrated = False
        self.gaze_model: Optional[EnsembleModel] = None
        self.model_trained = False
        self.training_data = []
        self.training_labels = []

        # 使用 5×5 网格 + 随机扰动的 25 个校准点
        self.num_calibration_points = 25
        self.calibration_points = self._generate_calibration_points()

        self.current_calibration_point = 0
        self.calibrating = False
        self.calibration_samples_per_point = 30
        self.current_point_samples = 0

        # 调试信息
        self.last_ear_left = 0.0
        self.last_ear_right = 0.0

        # 位置历史
        self.left_eye_history = deque(maxlen=5)
        self.right_eye_history = deque(maxlen=5)
        self.eye_feature_history = deque(maxlen=10)

        # 最近一次检测到的矩形
        self.last_left_eye_rect = None
        self.last_right_eye_rect = None
        self.last_face_rect = None

        # 读取已有模型
        self._load_model()

    # ------------------------------------------------------------------
    # 模型加载 / 生成校准点
    # ------------------------------------------------------------------
    def _generate_calibration_points(self):
        """生成分布均匀但带随机扰动的校准点集合（归一化坐标 0~1）"""
        points = []
        grid_x = np.linspace(0.1, 0.9, 5)
        grid_y = np.linspace(0.1, 0.9, 5)
        for x in grid_x:
            for y in grid_y:
                jitter_x = x + np.random.uniform(-0.03, 0.03)
                jitter_y = y + np.random.uniform(-0.03, 0.03)
                points.append((
                    float(np.clip(jitter_x, 0.05, 0.95)),
                    float(np.clip(jitter_y, 0.05, 0.95))
                ))
        random.shuffle(points)
        return points

    def _load_model(self):
        """加载已训练的模型"""
        if os.path.exists(self.model_path):
            try:
                self.gaze_model = joblib.load(self.model_path)
                self.model_trained = True
                self.calibrated = True
                print(f"[*] Gaze model loaded from: {self.model_path}")
            except Exception as e:
                print(f"[!] Failed to load model {self.model_path}: {e}")
                self.gaze_model = None
                self.model_trained = False
        else:
            print(f"[i] No pre-trained model at {self.model_path}, using geometric method.")

    def _save_model(self):
        """保存当前模型"""
        if self.gaze_model is not None:
            joblib.dump(self.gaze_model, self.model_path)
            print(f"✓ 模型已保存：{self.model_path}")

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------
    def extract_eye_features(
        self,
        left_eye_rect: Tuple[int, int, int, int],
        right_eye_rect: Tuple[int, int, int, int],
        face_rect: Tuple[int, int, int, int],
        frame: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        提取用于训练/预测的眼睛特征向量（>30 维）

        返回:
            np.ndarray(shape=(n_features,)) 或 None
        """
        if not all([left_eye_rect, right_eye_rect, face_rect]):
            return None

        lx, ly, lw, lh = left_eye_rect
        rx, ry, rw, rh = right_eye_rect
        fx, fy, fw, fh = face_rect

        features = []

        # 1. 眼睛中心（相对脸部坐标，0~1）
        left_center_x = (lx + lw / 2 - fx) / fw
        left_center_y = (ly + lh / 2 - fy) / fh
        right_center_x = (rx + rw / 2 - fx) / fw
        right_center_y = (ry + rh / 2 - fy) / fh
        features.extend([left_center_x, left_center_y, right_center_x, right_center_y])

        # 2. 眼睛大小与宽高比
        left_size = (lw * lh) / (fw * fh)
        right_size = (rw * rh) / (fw * fh)
        left_aspect = lw / max(lh, 1)
        right_aspect = rw / max(rh, 1)
        features.extend([left_size, right_size, left_aspect, right_aspect])

        # 3. 左右眼相对关系
        avg_size = (left_size + right_size) / 2
        size_ratio = left_size / max(right_size, 0.01)
        aspect_diff = left_aspect - right_aspect
        features.extend([avg_size, size_ratio, aspect_diff])

        # 4. 两眼间距离与整体中心
        eye_distance = np.sqrt((left_center_x - right_center_x) ** 2 +
                               (left_center_y - right_center_y) ** 2)
        eye_center_x = (left_center_x + right_center_x) / 2
        eye_center_y = (left_center_y + right_center_y) / 2
        features.extend([eye_distance, eye_center_x, eye_center_y])

        # 5. 相对脸部中心偏移
        face_center_x = 0.5
        face_center_y = 0.5
        left_offset_x = left_center_x - face_center_x
        left_offset_y = left_center_y - face_center_y
        right_offset_x = right_center_x - face_center_x
        right_offset_y = right_center_y - face_center_y
        features.extend([left_offset_x, left_offset_y, right_offset_x, right_offset_y])

        # 6. 亮度特征
        if frame is not None:
            try:
                left_eye_roi = frame[ly:ly + lh, lx:lx + lw]
                right_eye_roi = frame[ry:ry + rh, rx:rx + rw]

                if left_eye_roi.size > 0 and right_eye_roi.size > 0:
                    gray_left = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
                    gray_right = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
                    left_brightness = float(np.mean(gray_left) / 255.0)
                    right_brightness = float(np.mean(gray_right) / 255.0)
                    features.extend([left_brightness, right_brightness])
                else:
                    features.extend([0.5, 0.5])
            except Exception:
                features.extend([0.5, 0.5])
        else:
            features.extend([0.5, 0.5])

        # 7. 历史运动特征
        if len(self.eye_feature_history) >= 1:
            prev_features = np.array(self.eye_feature_history[-1], dtype=np.float32)
            curr_features = np.array(
                [left_center_x, left_center_y, right_center_x, right_center_y],
                dtype=np.float32
            )
            motion_vector = curr_features - prev_features
            motion_magnitude = float(np.linalg.norm(motion_vector))
            features.extend(motion_vector.tolist()[:2])  # dx, dy
            features.append(motion_magnitude)
        else:
            features.extend([0.0, 0.0, 0.0])

        # 8. 稳定性特征
        if len(self.eye_feature_history) >= 5:
            history_array = np.array(self.eye_feature_history[-5:], dtype=np.float32)
            position_std = float(np.std(history_array[:, :4]))
            position_mean = float(np.mean(history_array[:, :4]))
            features.extend([position_std, position_mean])
        else:
            features.extend([0.0, 0.0])

        # 保存基础位置到历史
        base_features = [left_center_x, left_center_y, right_center_x, right_center_y]
        self.eye_feature_history.append(base_features)

        features_array = np.array(features, dtype=np.float32)

        # 大部分特征在 [0,1]，统一映射到 [-1,1]
        features_array = 2 * (features_array - 0.5)

        return features_array

    # ------------------------------------------------------------------
    # 校准 / 训练
    # ------------------------------------------------------------------
    def start_calibration(self):
        """开始校准流程"""
        print("\n" + "=" * 60)
        print(f"开始眼动校准 - 请依次看屏幕上出现的 {len(self.calibration_points)} 个点")
        print("每个点显示时请尽量凝视，系统将自动采集样本")
        print("=" * 60 + "\n")

        self.calibrating = True
        self.current_calibration_point = 0
        self.current_point_samples = 0
        self.training_data = []
        self.training_labels = []
        self.calibrated = False

    def add_calibration_sample(
        self,
        eye_features: np.ndarray,
        screen_width: int,
        screen_height: int
    ) -> bool:
        """添加一条校准样本"""
        if not self.calibrating or eye_features is None:
            return False

        if self.current_calibration_point >= len(self.calibration_points):
            return False

        x_ratio, y_ratio = self.calibration_points[self.current_calibration_point]
        target_x = x_ratio * screen_width
        target_y = y_ratio * screen_height

        self.training_data.append(eye_features)
        self.training_labels.append([target_x, target_y])
        self.current_point_samples += 1

        if self.current_point_samples >= self.calibration_samples_per_point:
            print(f"✓ 点 {self.current_calibration_point + 1}/{len(self.calibration_points)} 采集完成")
            self.current_calibration_point += 1
            self.current_point_samples = 0

            if self.current_calibration_point >= len(self.calibration_points):
                print("\n✓ 校准数据采集完成，开始训练模型...")
                ok = self.train_gaze_model()
                if ok:
                    self.calibrating = False
                    self.calibrated = True
                return ok
        return True

    def train_gaze_model(self) -> bool:
        """使用采集到的数据训练视线映射模型"""
        min_samples = len(self.calibration_points) * self.calibration_samples_per_point
        if len(self.training_data) < min_samples:
            print(f"[!] 样本不足：需要 {min_samples}，当前 {len(self.training_data)}")
            return False

        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            X = np.array(self.training_data, dtype=np.float32)
            y = np.array(self.training_labels, dtype=np.float32)

            # 数据增强
            noises = [0.02, 0.05, 0.08]
            X_aug = [X]
            y_aug = [y]
            for s in noises:
                X_aug.append(X + np.random.normal(0, s, X.shape))
                y_aug.append(y)
            X_aug = np.vstack(X_aug)
            y_aug = np.vstack(y_aug)
            print(f"[*] 数据增强完成，样本总数：{len(X_aug)}")

            # 特征缩放到 [-1,1]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_scaled = scaler.fit_transform(X_aug)

            # 训练/验证集划分
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_aug, test_size=0.2, random_state=42
            )

            # 主模型：MLP
            model1 = MLPRegressor(
                hidden_layer_sizes=(64, 128, 64),
                activation='relu',
                solver='adam',
                learning_rate_init=3e-5,
                batch_size=256,
                max_iter=2000,
                early_stopping=True,
                n_iter_no_change=300,
                validation_fraction=0.15,
                random_state=42,
                alpha=0.00005,
                verbose=False
            )

            print("[*] 训练主 MLP 模型中...")
            model1.fit(X_train, y_train)
            print("✓ 主模型训练完成")

            # 第二层：Ridge 线性微调
            pred_train = model1.predict(X_train)
            model2 = Ridge(alpha=0.3)
            model2.fit(pred_train, y_train)

            # 验证集评估
            val_pred = model2.predict(model1.predict(X_val))
            mse = mean_squared_error(y_val, val_pred)
            mae = mean_absolute_error(y_val, val_pred)
            r2 = r2_score(y_val, val_pred)

            print("✅ 验证集结果：")
            print(f"  R²   = {r2:.4f}")
            print(f"  MAE  = {mae:.2f} px")
            print(f"  RMSE = {np.sqrt(mse):.2f} px")

            # 构建集成模型并保存
            self.gaze_model = EnsembleModel(model1, model2, scaler)
            self.model_trained = True
            self._save_model()

            print("✅ 视线模型训练成功！")
            return True

        except ImportError as e:
            print(f"[!] 缺少依赖（scikit-learn）: {e}")
            return False
        except Exception as e:
            print(f"[!] 模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # 预测（ML 模型）
    # ------------------------------------------------------------------
    def predict_gaze_with_model(
        self,
        eye_features: np.ndarray,
        screen_width: int,
        screen_height: int
    ) -> Optional[Tuple[int, int]]:
        """使用训练好的 ML 模型预测屏幕坐标"""
        if not self.model_trained or self.gaze_model is None or eye_features is None:
            return None

        try:
            if not isinstance(eye_features, np.ndarray):
                eye_features = np.array(eye_features, dtype=np.float32)
            if eye_features.ndim == 1:
                eye_features = eye_features.reshape(1, -1)

            # 模型直接输出像素坐标
            pred = self.gaze_model.predict(eye_features)[0]
            screen_x = int(np.clip(pred[0], 0, screen_width))
            screen_y = int(np.clip(pred[1], 0, screen_height))

            # 坐标级平滑
            if self.prev_gaze_coords is not None:
                prev_x, prev_y = self.prev_gaze_coords
                distance = np.sqrt((screen_x - prev_x) ** 2 + (screen_y - prev_y) ** 2)

                if distance < 20:
                    smooth_factor = 0.7
                elif distance < 60:
                    smooth_factor = 0.5
                else:
                    smooth_factor = 0.2

                screen_x = int(prev_x * smooth_factor + screen_x * (1 - smooth_factor))
                screen_y = int(prev_y * smooth_factor + screen_y * (1 - smooth_factor))

            self.prev_gaze_coords = (screen_x, screen_y)
            return screen_x, screen_y

        except Exception as e:
            print(f"[!] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # 眨眼 & 辅助方法
    # ------------------------------------------------------------------
    def calculate_eye_aspect_ratio(self, eye_region: np.ndarray) -> float:
        """简单 EAR 估计（亮度 + 宽高比）"""
        if eye_region is None or eye_region.size == 0:
            return 1.0

        h, w = eye_region.shape[:2]
        if h == 0 or w == 0:
            return 1.0

        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if eye_region.ndim == 3 else eye_region
        brightness = float(np.mean(gray) / 255.0)
        aspect_ratio = min(w, h) / max(w, h, 1)
        ear = brightness * aspect_ratio
        return ear

    def smooth_eye_position(self, eye_rect: Tuple[int, int, int, int],
                            is_left: bool = True) -> Optional[Tuple[int, int, int, int]]:
        """对眼睛矩形位置做时间平滑"""
        if eye_rect is None:
            return None

        history = self.left_eye_history if is_left else self.right_eye_history
        history.append(eye_rect)

        if len(history) == 0:
            return eye_rect

        avg_x = int(np.mean([r[0] for r in history]))
        avg_y = int(np.mean([r[1] for r in history]))
        avg_w = int(np.mean([r[2] for r in history]))
        avg_h = int(np.mean([r[3] for r in history]))
        return avg_x, avg_y, avg_w, avg_h

    def estimate_gaze_vector(self, left_eye_rect, right_eye_rect, face_rect) -> np.ndarray:
        """几何法估计视线向量（备选方案）"""
        if left_eye_rect is None or right_eye_rect is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        lx, ly, lw, lh = left_eye_rect
        rx, ry, rw, rh = right_eye_rect
        fx, fy, fw, fh = face_rect

        left_eye_center_x = lx + lw // 2
        left_eye_center_y = ly + lh // 2
        right_eye_center_x = rx + rw // 2
        right_eye_center_y = ry + rh // 2

        avg_eye_x = (left_eye_center_x + right_eye_center_x) / 2.0
        avg_eye_y = (left_eye_center_y + right_eye_center_y) / 2.0

        face_center_x = fx + fw // 2
        face_center_y = fy + fh // 2

        max_horizontal_offset = fw / 3.0
        max_vertical_offset = fh / 4.0

        gaze_x = (avg_eye_x - face_center_x) / max_horizontal_offset
        gaze_y = (avg_eye_y - face_center_y) / max_vertical_offset

        gaze_x = float(np.clip(gaze_x, -1.0, 1.0))
        gaze_y = float(np.clip(gaze_y, -0.8, 0.8))
        gaze_vector = np.array([gaze_x, gaze_y], dtype=np.float32)

        # 平滑
        self.gaze_smooth_buffer.append(gaze_vector)
        if len(self.gaze_smooth_buffer) > 1:
            gaze_vector = np.mean(self.gaze_smooth_buffer, axis=0)
        return gaze_vector

    def gaze_to_screen_coords(self, gaze_vector: np.ndarray,
                              screen_width: int, screen_height: int) -> Tuple[int, int]:
        """几何视线向量 -> 屏幕坐标"""
        gx, gy = float(gaze_vector[0]), float(gaze_vector[1])

        gx_nonlinear = np.sign(gx) * (abs(gx) ** 0.9)
        gy_nonlinear = np.sign(gy) * (abs(gy) ** 0.9)

        screen_x = (gx_nonlinear + 1.0) / 2.0 * screen_width
        screen_y = (gy_nonlinear + 0.8) / 1.6 * screen_height

        if self.prev_gaze_coords is not None:
            prev_x, prev_y = self.prev_gaze_coords
            distance = np.sqrt((screen_x - prev_x) ** 2 + (screen_y - prev_y) ** 2)

            if distance < 30:
                smooth_factor = 0.6
            elif distance < 100:
                smooth_factor = 0.4
            else:
                smooth_factor = 0.2

            screen_x = prev_x * smooth_factor + screen_x * (1 - smooth_factor)
            screen_y = prev_y * smooth_factor + screen_y * (1 - smooth_factor)

        self.prev_gaze_coords = (screen_x, screen_y)

        screen_x = int(np.clip(screen_x, 5, screen_width - 5))
        screen_y = int(np.clip(screen_y, 5, screen_height - 5))
        return screen_x, screen_y

    def detect_left_blink(self, left_eye_region: np.ndarray, frame_id: int = 0) -> bool:
        """检测左眼眨眼，带冷却逻辑"""
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
        """检测右眼眨眼（目前只记录 EAR）"""
        if right_eye_region is None or right_eye_region.size == 0:
            return False

        ear = self.calculate_eye_aspect_ratio(right_eye_region)
        self.last_ear_right = ear
        return ear < self.EAR_THRESHOLD

    # ------------------------------------------------------------------
    # 主处理函数
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> Dict:
        """处理单帧图像，输出检测和视线结果"""
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
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) > 0:
                (fx, fy, fw, fh) = faces[0]
                result_dict['face_detected'] = True
                result_dict['face_confidence'] = 0.8
                self.last_face_rect = (fx, fy, fw, fh)

                face_roi = gray[fy:fy + fh, fx:fx + fw]
                eyes = self.eye_cascade.detectMultiScale(
                    face_roi,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(15, 15)
                )

                if len(eyes) >= 2:
                    eyes = sorted(eyes, key=lambda e: e[0])
                    left_eye_rect_roi = eyes[0]
                    right_eye_rect_roi = eyes[1]

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

                    left_eye_rect = self.smooth_eye_position(left_eye_rect, is_left=True)
                    right_eye_rect = self.smooth_eye_position(right_eye_rect, is_left=False)

                    if left_eye_rect and right_eye_rect:
                        self.last_left_eye_rect = left_eye_rect
                        self.last_right_eye_rect = right_eye_rect

                        lx, ly, lw, lh = left_eye_rect
                        rx, ry, rw, rh = right_eye_rect

                        left_eye_region = frame[ly:ly + lh, lx:lx + lw]
                        right_eye_region = frame[ry:ry + rh, rx:rx + rw]

                        eye_features = self.extract_eye_features(
                            left_eye_rect, right_eye_rect, (fx, fy, fw, fh), frame
                        )

                        # 校准模式：收集样本
                        if self.calibrating and eye_features is not None:
                            self.add_calibration_sample(eye_features, w, h)

                        # 视线预测
                        if self.model_trained and eye_features is not None:
                            screen_coords = self.predict_gaze_with_model(eye_features, w, h)
                            if screen_coords is not None:
                                result_dict['gaze_screen_coords'] = screen_coords
                                gaze_x = (screen_coords[0] / w * 2) - 1
                                gaze_y = (screen_coords[1] / h * 2) - 1
                                result_dict['gaze_vector'] = np.array([gaze_x, gaze_y],
                                                                      dtype=np.float32)
                        else:
                            gaze_vector = self.estimate_gaze_vector(
                                left_eye_rect, right_eye_rect, (fx, fy, fw, fh)
                            )
                            result_dict['gaze_vector'] = gaze_vector
                            result_dict['gaze_screen_coords'] = self.gaze_to_screen_coords(
                                gaze_vector, w, h
                            )

                        # 眨眼检测
                        blink_state = self.detect_left_blink(left_eye_region, frame_id)
                        result_dict['blink_detected'] = blink_state
                        if blink_state:
                            result_dict['blink_triggered'] = True
                            self.blink_detected = False

                        self.detect_right_blink(right_eye_region)
                        result_dict['left_eye_ear'] = self.last_ear_left
                        result_dict['right_eye_ear'] = self.last_ear_right

        except Exception as e:
            print(f"[!] Error in process_frame: {e}")
            import traceback
            traceback.print_exc()

        return result_dict

    # ------------------------------------------------------------------
    # UI 绘制辅助
    # ------------------------------------------------------------------
    def draw_calibration_ui(self, frame: np.ndarray,
                            screen_width: int, screen_height: int) -> np.ndarray:
        """绘制校准 UI 和当前校准点"""
        if not self.calibrating:
            return frame

        h, w = frame.shape[:2]

        # 当前点
        current_point = self.calibration_points[self.current_calibration_point]
        _target_x = int(current_point[0] * screen_width)
        _target_y = int(current_point[1] * screen_height)

        for idx, (x_ratio, y_ratio) in enumerate(self.calibration_points):
            point_x = int(x_ratio * screen_width)
            point_y = int(y_ratio * screen_height)

            if idx == self.current_calibration_point:
                cv2.circle(frame, (point_x, point_y), 30, (0, 0, 255), -1)
                cv2.circle(frame, (point_x, point_y), 35, (0, 255, 0), 3)
                cv2.line(frame, (point_x - 20, point_y),
                         (point_x + 20, point_y), (255, 255, 255), 2)
                cv2.line(frame, (point_x, point_y - 20),
                         (point_x, point_y + 20), (255, 255, 255), 2)
            else:
                cv2.circle(frame, (point_x, point_y), 8, (100, 100, 100), -1)
                cv2.circle(frame, (point_x, point_y), 10, (150, 150, 150), 1)

        progress_text = (f"CALIBRATING: Point {self.current_calibration_point + 1}/"
                         f"{len(self.calibration_points)}  "
                         f"Sample: {self.current_point_samples}/{self.calibration_samples_per_point}")
        cv2.putText(frame, progress_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        instruction_text = "Please gaze at the red point"
        cv2.putText(frame, instruction_text, (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def draw_landmarks(self, frame: np.ndarray,
                       landmarks: np.ndarray = None) -> np.ndarray:
        """简单绘制人脸和眼睛矩形（调试用）"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            face_roi = gray[fy:fy + fh, fx:fx + fw]
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (fx + ex, fy + ey),
                              (fx + ex + ew, fy + ey + eh), (0, 255, 0), 2)

        status_text = "ML Model" if self.model_trained else "Geometric Method"
        cv2.putText(frame, f"Eye Tracking - {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.calibrating:
            calib_text = (f"CALIBRATING: Point {self.current_calibration_point + 1}/"
                          f"{len(self.calibration_points)}")
            cv2.putText(frame, calib_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame
