"""
Eye Shooter - Main Application
眼动追踪射击游戏主程序

使用OpenCV + MediaPipe实现眼动追踪
玩法：屏幕上出现移动的靶子，玩家用视线控制准心，左眨眼开火
"""

import cv2
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.gaze_tracker import GazeTracker
from src.game.game_engine import GameEngine, GameState


class EyeShooterApp:
    """Eye Shooter 应用程序"""
    
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720):
        """
        初始化应用
        
        Args:
            camera_id: 摄像头ID
            width: 游戏窗口宽度
            height: 游戏窗口高度
        """
        self.width = width
        self.height = height
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # 初始化眼动追踪器和游戏引擎
        self.gaze_tracker = GazeTracker(width, height)
        self.game_engine = GameEngine(width, height)
        
        # UI状态
        self.show_debug = True  # 是否显示调试信息
        self.show_landmarks = False  # 是否显示面部特征点
        
        # 性能监测
        self.fps = 30
        self.frame_count = 0
        self.last_blink_time = 0  # 上一次眨眼的帧数
        
        # 保存EAR信息用于调试
        self.last_ear_left = 0
        self.last_ear_right = 0
        
        # 准心锁定机制（防止眨眼时晃动）
        self.blink_lock_frames = 6  # 眨眼时锁定准心的帧数
        self.locked_crosshair_x = None
        self.locked_crosshair_y = None
        
        print("=" * 60)
        print("Eye Shooter - Gaze-Controlled Shooting Game (ML Edition)")
        print("=" * 60)
        print("\nControls:")
        print("  Space: Start/Pause/Resume game")
        print("  D: Toggle debug info")
        print("  L: Toggle landmarks display")
        print("  R: Reset game")
        print("  C: Calibrate gaze tracking (ML training)")
        print("  -/=: Adjust blink sensitivity")
        print("  ESC: Quit")
        print("\nGame Rules:")
        print("  - Move crosshair with your gaze direction")
        print("  - Blink LEFT eye to shoot")
        print("  - Hit targets to increase score")
        print("  - Game becomes harder as you level up")
        print("\n" + "=" * 60)
        print("Tips:")
        print("  - First time? Press 'C' to calibrate for best accuracy")
        print("  - Good lighting improves face detection")
        print("  - Keep your face at 30-60cm from camera")
        print("  - Use - and = keys to adjust blink sensitivity")
        print("=" * 60 + "\n")
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """绘制UI元素"""
        h, w = frame.shape[:2]
        
        # 绘制准心
        cx, cy = self.game_engine.crosshair_x, self.game_engine.crosshair_y
        size = self.game_engine.crosshair_size
        
        # 准心十字（黄色）
        cv2.line(frame, (cx - size, cy), (cx + size, cy), (0, 255, 255), 2)
        cv2.line(frame, (cx, cy - size), (cx, cy + size), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        
        # 绘制靶子
        for target in self.game_engine.targets:
            target.draw(frame)
        
        # 绘制分数和等级
        score_text = f"Score: {self.game_engine.score}"
        level_text = f"Level: {self.game_engine.level}"
        
        cv2.putText(frame, score_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, level_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # 游戏状态提示
        if self.game_engine.state == GameState.MENU:
            state_text = "Press SPACE to start"
            cv2.putText(frame, state_text, (w // 2 - 200, h // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        elif self.game_engine.state == GameState.PAUSED:
            state_text = "PAUSED - Press SPACE to resume"
            cv2.putText(frame, state_text, (w // 2 - 250, h // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        
        # 眨眼指示器
        if self.last_blink_time > 0 and self.frame_count - self.last_blink_time < 8:
            # 眨眼闪光效果
            flash_alpha = (8 - (self.frame_count - self.last_blink_time)) / 8.0
            flash_color = (0, 100 * flash_alpha, 255 * flash_alpha)
            cv2.circle(frame, (self.game_engine.crosshair_x, self.game_engine.crosshair_y), 
                      size + 10, flash_color, 2)
        
        # 调试信息
        if self.show_debug:
            debug_y = h - 220
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, debug_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # 显示模型状态
            model_status = "✓ ML Model" if self.gaze_tracker.model_trained else "○ Geometric"
            cv2.putText(frame, f"Mode: {model_status}", (20, debug_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            if self.game_engine.state == GameState.PLAYING:
                gaze_text = f"Gaze: ({self.game_engine.crosshair_x}, {self.game_engine.crosshair_y})"
                targets_text = f"Targets: {len([t for t in self.game_engine.targets if t.alive])}"
                
                cv2.putText(frame, gaze_text, (20, debug_y + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame, targets_text, (20, debug_y + 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # 显示眼睛宽高比
                if hasattr(self, 'last_ear_left') and hasattr(self, 'last_ear_right'):
                    ear_text = f"L-EAR: {self.last_ear_left:.2f} | R-EAR: {self.last_ear_right:.2f}"
                    threshold_text = f"Blink Threshold: {self.gaze_tracker.EAR_THRESHOLD:.2f}"
                    cv2.putText(frame, ear_text, (20, debug_y + 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cv2.putText(frame, threshold_text, (20, debug_y + 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # 帮助信息
        help_text = "Space:Start | D:Debug | L:Landmarks | C:Calibrate | R:Reset | ESC:Quit"
        cv2.putText(frame, help_text, (20, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """主程序循环"""
        prev_frame_time = cv2.getTickCount()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # 处理每一帧 - 传入帧ID用于眨眼冷却计算
                gaze_result = self.gaze_tracker.process_frame(frame, self.frame_count)
                
                # 保存EAR用于调试显示
                self.last_ear_left = gaze_result['left_eye_ear'] or 0
                self.last_ear_right = gaze_result['right_eye_ear'] or 0
                
                # 更新游戏
                fire_trigger = gaze_result['blink_triggered']  # 使用真正触发的射击
                
                # 获取准心坐标
                if gaze_result['face_detected'] and gaze_result['gaze_screen_coords']:
                    gaze_coords = gaze_result['gaze_screen_coords']
                    
                    # 眨眼时锁定准心，防止眼睛闭合时的准心晃动
                    if fire_trigger:
                        # 记录当前位置作为锁定点
                        self.locked_crosshair_x = gaze_coords[0]
                        self.locked_crosshair_y = gaze_coords[1]
                        self.last_blink_time = self.frame_count
                    elif (self.locked_crosshair_x is not None and 
                          self.frame_count - self.last_blink_time < self.blink_lock_frames):
                        # 在锁定期间，使用锁定的坐标
                        gaze_coords = (self.locked_crosshair_x, self.locked_crosshair_y)
                    else:
                        # 锁定期结束，释放锁定
                        self.locked_crosshair_x = None
                        self.locked_crosshair_y = None
                else:
                    # 没有检测到脸，保持上一次的准心位置
                    gaze_coords = (self.game_engine.crosshair_x, self.game_engine.crosshair_y)
                
                # 更新游戏状态
                game_state = self.game_engine.update(gaze_coords, fire_trigger)
                
                # 绘制UI
                frame = self.draw_ui(frame)
                
                # 绘制面部特征点（可选调试）
                if self.show_landmarks and gaze_result['face_detected']:
                    frame = self.gaze_tracker.draw_landmarks(frame, gaze_result['landmarks'])
                
                # 计算FPS
                current_time = cv2.getTickCount()
                fps_time = (current_time - prev_frame_time) / cv2.getTickFrequency()
                self.fps = 1 / fps_time if fps_time > 0 else 30
                prev_frame_time = current_time
                
                # 显示画面
                cv2.imshow('Eye Shooter', frame)
                self.frame_count += 1
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\nExiting...")
                    break
                elif key == ord(' '):  # Space - 开始/暂停
                    if self.game_engine.state == GameState.MENU:
                        self.game_engine.start_game()
                        print("Game started!")
                    elif self.game_engine.state == GameState.PLAYING:
                        self.game_engine.pause()
                        print("Game paused")
                    elif self.game_engine.state == GameState.PAUSED:
                        self.game_engine.resume()
                        print("Game resumed")
                
                elif key == ord('d'):  # D - 切换调试信息
                    self.show_debug = not self.show_debug
                    debug_status = "ON" if self.show_debug else "OFF"
                    print(f"Debug mode: {debug_status}")
                
                elif key == ord('l'):  # L - 切换特征点显示
                    self.show_landmarks = not self.show_landmarks
                    landmarks_status = "ON" if self.show_landmarks else "OFF"
                    print(f"Landmarks display: {landmarks_status}")
                
                elif key == ord('r'):  # R - 重置游戏
                    self.game_engine.reset()
                    print("Game reset")
                
                elif key == ord('-'):  # 降低眨眼阈值
                    self.gaze_tracker.EAR_THRESHOLD = max(0.1, self.gaze_tracker.EAR_THRESHOLD - 0.02)
                    print(f"EAR Threshold: {self.gaze_tracker.EAR_THRESHOLD:.2f}")
                
                elif key == ord('='):  # 提高眨眼阈值
                    self.gaze_tracker.EAR_THRESHOLD = min(0.4, self.gaze_tracker.EAR_THRESHOLD + 0.02)
                    print(f"EAR Threshold: {self.gaze_tracker.EAR_THRESHOLD:.2f}")
                
                elif key == ord('c'):  # C - 开始校准
                    print("\nStarting calibration...")
                    self.gaze_tracker.start_calibration()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Total frames processed: {self.frame_count}")
        print("Bye!")


def main():
    """程序入口"""
    try:
        app = EyeShooterApp(width=1280, height=720)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
