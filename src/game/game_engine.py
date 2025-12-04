"""
Game Logic Module - Eye Shooter
处理游戏逻辑：目标生成、移动、碰撞检测、计分等
"""

import numpy as np
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


class GameState(Enum):
    """游戏状态"""
    MENU = 0
    PLAYING = 1
    PAUSED = 2
    GAME_OVER = 3


@dataclass
class Target:
    """移动靶子类"""
    x: float
    y: float
    width: int = 40
    height: int = 40
    vx: float = 0  # x方向速度
    vy: float = 0  # y方向速度
    alive: bool = True
    hit_frames: int = 0  # 被击中后的显示帧数（用于显示爆炸效果）
    
    def update(self, screen_width: int, screen_height: int):
        """更新靶子位置"""
        if not self.alive:
            return
        
        self.x += self.vx
        self.y += self.vy
        
        # 边界反弹
        if self.x <= 0 or self.x >= screen_width:
            self.vx *= -1
            self.x = np.clip(self.x, 0, screen_width)
        
        if self.y <= 0 or self.y >= screen_height:
            self.vy *= -1
            self.y = np.clip(self.y, 0, screen_height)
    
    def is_hit(self, point_x: float, point_y: float, radius: float = 20) -> bool:
        """检查是否被击中（距离检测）"""
        dist = np.sqrt((self.x - point_x) ** 2 + (self.y - point_y) ** 2)
        return dist < (self.width / 2 + radius)
    
    def draw(self, frame, color=(0, 255, 0)):
        """绘制靶子"""
        import cv2
        
        if not self.alive:
            return
        
        x, y = int(self.x), int(self.y)
        w, h = self.width, self.height
        
        if self.hit_frames > 0:
            # 被击中后显示爆炸效果
            color = (0, 0, 255)  # 红色
            radius = w // 2 + self.hit_frames
            cv2.circle(frame, (x, y), radius, color, 2)
            self.hit_frames -= 1
        else:
            # 正常显示
            cv2.rectangle(frame, (x - w // 2, y - h // 2), 
                         (x + w // 2, y + h // 2), color, 2)
            cv2.circle(frame, (x, y), 3, color, -1)
        
        return frame


class GameEngine:
    """游戏引擎"""
    
    def __init__(self, screen_width: int = 1280, screen_height: int = 720):
        """
        初始化游戏引擎
        
        Args:
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.state = GameState.MENU
        self.score = 0
        self.level = 1
        self.targets: List[Target] = []
        self.frame_count = 0
        
        # 游戏参数
        self.target_spawn_interval = 60  # 帧数间隔
        self.last_spawn_time = 0
        self.max_targets = 3
        
        # 准心参数
        self.crosshair_x = screen_width // 2
        self.crosshair_y = screen_height // 2
        self.crosshair_size = 20
        
    def start_game(self):
        """开始游戏"""
        self.state = GameState.PLAYING
        self.score = 0
        self.level = 1
        self.targets = []
        self.frame_count = 0
        self.last_spawn_time = 0
    
    def update(self, gaze_coords: Tuple[int, int], fire_trigger: bool = False) -> Dict:
        """
        更新游戏状态
        
        Args:
            gaze_coords: 准心坐标 (x, y)
            fire_trigger: 是否触发射击（眨眼）
            
        Returns:
            包含游戏状态信息的字典
        """
        if self.state != GameState.PLAYING:
            return {'targets': self.targets, 'score': self.score, 'level': self.level}
        
        # 更新准心位置
        self.crosshair_x, self.crosshair_y = gaze_coords
        
        # 更新靶子
        for target in self.targets:
            target.update(self.screen_width, self.screen_height)
        
        # 检查射击
        if fire_trigger:
            self._handle_fire()
        
        # 生成新靶子
        self.frame_count += 1
        if self.frame_count - self.last_spawn_time > self.target_spawn_interval:
            self._spawn_target()
            self.last_spawn_time = self.frame_count
        
        # 移除死亡的靶子
        self.targets = [t for t in self.targets if t.alive or t.hit_frames > 0]
        
        # 检查游戏结束条件（可选）
        # if self.targets == [] and self.frame_count > 300:
        #     self.state = GameState.GAME_OVER
        
        return {
            'targets': self.targets,
            'score': self.score,
            'level': self.level,
            'crosshair_x': self.crosshair_x,
            'crosshair_y': self.crosshair_y,
            'frame_count': self.frame_count
        }
    
    def _spawn_target(self):
        """生成新靶子"""
        if len([t for t in self.targets if t.alive]) >= self.max_targets:
            return
        
        # 随机位置（避免太靠近边界）
        x = random.randint(100, self.screen_width - 100)
        y = random.randint(100, self.screen_height - 100)
        
        # 随机速度
        vx = random.uniform(-3, 3)
        vy = random.uniform(-3, 3)
        
        target = Target(x=x, y=y, vx=vx, vy=vy)
        self.targets.append(target)
    
    def _handle_fire(self):
        """处理射击事件"""
        hit_any = False
        for target in self.targets:
            if target.alive and target.is_hit(self.crosshair_x, self.crosshair_y):
                target.alive = False
                target.hit_frames = 10  # 爆炸效果持续帧数
                self.score += 10
                hit_any = True
        
        # 每升10分升级
        self.level = self.score // 10 + 1
        
        # 难度随等级增加
        self.target_spawn_interval = max(30, 60 - self.level * 3)
        self.max_targets = min(8, 3 + self.level // 2)
    
    def pause(self):
        """暂停游戏"""
        if self.state == GameState.PLAYING:
            self.state = GameState.PAUSED
    
    def resume(self):
        """继续游戏"""
        if self.state == GameState.PAUSED:
            self.state = GameState.PLAYING
    
    def reset(self):
        """重置游戏"""
        self.state = GameState.MENU
        self.score = 0
        self.level = 1
        self.targets = []
        self.frame_count = 0
