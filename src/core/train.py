"""
独立训练脚本：只负责采集校准数据并训练/保存模型
运行前请确保安装了 scikit-learn：
    pip install scikit-learn
"""

import cv2
import time

from gaze_tracker import GazeTracker


def countdown(frame, width, height, seconds=3):
    """校准前倒计时"""
    start = time.time()

    while True:
        now = time.time()
        elapsed = now - start
        remaining = int(seconds - elapsed)

        if remaining <= 0:
            break

        # 显示倒计时数字
        msg = f"准备校准：{remaining}"
        cv2.putText(frame, msg, (width // 2 - 200, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)

        cv2.imshow("Gaze Calibration & Training", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            return False  # 用户按 ESC 中断

    # 显示“开始采集！”
    msg = "开始采集！"
    cv2.putText(frame, msg, (width // 2 - 200, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    cv2.imshow("Gaze Calibration & Training", frame)
    cv2.waitKey(800)

    return True


def main():
    width, height = 1280, 720
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[!] 无法打开摄像头")
        return 1

    tracker = GazeTracker(camera_width=width, camera_height=height)

    print("=" * 60)
    print("Gaze Tracker - Calibration & Training")
    print("=" * 60)
    print("说明：")
    print("  - 程序会依次在画面上显示一系列红点")
    print("  - 请保持头部基本不动，眼睛凝视当前红点")
    print("  - 每个点会自动采集若干帧，无需按键")
    print("操作：")
    print("  - ESC：中途退出")
    print("=" * 60 + "\n")

    # ---- 显示窗口并进行准备倒计时 ----
    ret, frame = cap.read()
    if ret:
        cv2.namedWindow("Gaze Calibration & Training", cv2.WINDOW_NORMAL)
        cv2.imshow("Gaze Calibration & Training", frame)

        if not countdown(frame.copy(), width, height, seconds=3):
            print("用户中止校准")
            return 0

    # ---- 开始校准 ----
    tracker.start_calibration()
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] 读取摄像头帧失败")
                break

            tracker.process_frame(frame, frame_id)

            # 绘制校准 UI
            frame = tracker.draw_calibration_ui(frame, width, height)

            cv2.imshow("Gaze Calibration & Training", frame)
            frame_id += 1

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("用户中止校准")
                break

            # 如果训练结束，退出
            if not tracker.calibrating and tracker.model_trained:
                print("\n✅ 校准 + 模型训练已完成，可以关闭窗口。")
                cv2.waitKey(1000)
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
