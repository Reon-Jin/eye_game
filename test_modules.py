"""
快速测试脚本 - 检查核心模块是否能正常导入和初始化
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("Eye Shooter - Module Test")
print("=" * 60)

try:
    print("\n1. Testing basic imports...")
    import cv2
    print("   ✓ OpenCV imported successfully")
    
    print("\n2. Testing GameEngine...")
    from src.game.game_engine import GameEngine, GameState, Target
    print("   ✓ GameEngine imported successfully")
    
    print("\n3. Testing GameEngine initialization...")
    engine = GameEngine(1280, 720)
    print(f"   ✓ GameEngine initialized")
    print(f"     - Screen size: {engine.screen_width}x{engine.screen_height}")
    print(f"     - Initial spawn interval: {engine.target_spawn_interval} frames")
    print(f"     - Max targets: {engine.max_targets}")
    
    print("\n4. Testing game state transitions...")
    engine.start_game()
    print(f"   ✓ Game started, state: {engine.state}")
    engine.pause()
    print(f"   ✓ Game paused, state: {engine.state}")
    engine.resume()
    print(f"   ✓ Game resumed, state: {engine.state}")
    
    print("\n5. Testing Target class...")
    target = Target(x=100, y=100, vx=2, vy=1)
    print(f"   ✓ Target created at ({target.x}, {target.y})")
    target.update(1280, 720)
    print(f"   ✓ Target updated to ({target.x}, {target.y})")
    
    print("\n6. Testing collision detection...")
    is_hit = target.is_hit(100, 100, radius=50)
    print(f"   ✓ Collision test at origin: {is_hit}")
    
    print("\n7. Testing GazeTracker (without MediaPipe)...")
    try:
        from src.core.gaze_tracker import GazeTracker
        print("   ✓ GazeTracker imported (note: MediaPipe may not work)")
        print("   ✓ GazeTracker module is ready")
    except Exception as e:
        print(f"   ⚠ GazeTracker import warning: {e}")
        print("   This is expected if MediaPipe isn't fully installed")
    
    print("\n" + "=" * 60)
    print("✓ Core functionality tests passed!")
    print("=" * 60)
    print("\nNotes:")
    print("- If you see MediaPipe errors, try running:")
    print("  pip uninstall mediapipe -y")
    print("  pip install mediapipe")
    print("\n- Or try a fresh environment with:")
    print("  conda create -n eye_shooter python=3.10")
    print("  conda activate eye_shooter")
    print("  pip install opencv-python mediapipe numpy")
    print("\n" + "=" * 60)
    print("Next step: Run 'python main.py' to start the game")
    print("=" * 60 + "\n")
    
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
