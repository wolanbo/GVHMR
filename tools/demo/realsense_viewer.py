#!/usr/bin/env python3
"""
RealSense 相机实时预览工具
独立脚本,用于测试和调试RealSense相机数据采集与显示
"""

import cv2
import numpy as np
import argparse
import sys

try:
    import pyrealsense2 as rs
except ImportError:
    print("[ERROR] pyrealsense2 not installed. Please run: pip install pyrealsense2")
    sys.exit(1)


class TemporalFilter:
    """时域滤波器,平滑深度图"""
    
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.previous_frame = None
    
    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


class RealSenseViewer:
    """RealSense相机查看器"""
    
    def __init__(self, width=640, height=480, fps=30, show_depth=False):
        self.width = width
        self.height = height
        self.fps = fps
        self.show_depth = show_depth
        self.running = False
        self.pipeline = None
        self.temporal_filter = TemporalFilter(alpha=0.5)
        
    def start(self):
        """启动相机"""
        print(f"[Info] Initializing RealSense camera ({self.width}x{self.height}@{self.fps}fps)...")
        
        # 配置pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        # 启动pipeline
        profile = self.pipeline.start(config)
        
        # 获取相机内参
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        print(f"[Info] Camera intrinsics:")
        print(f"  - fx: {intrinsics.fx:.2f}, fy: {intrinsics.fy:.2f}")
        print(f"  - cx: {intrinsics.ppx:.2f}, cy: {intrinsics.ppy:.2f}")
        print(f"  - Width: {intrinsics.width}, Height: {intrinsics.height}")
        
        # 创建align对象
        self.align = rs.align(rs.stream.color)
        
        self.running = True
        print("[Info] Camera started successfully!")
        print("[Info] Press 'q' to quit, 's' to save snapshot")
        
        return intrinsics
    
    def run(self):
        """运行主循环"""
        if not self.running:
            self.start()
        
        frame_count = 0
        
        try:
            while self.running:
                # 等待帧数据 (超时5秒)
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                
                if frames is None:
                    print("[Warning] No frames received")
                    continue
                
                # 对齐深度和彩色图像
                aligned_frames = self.align.process(frames)
                
                # 获取彩色图像
                color_frame = aligned_frames.get_color_frame()
                if not color_frame:
                    continue
                color_img = np.asanyarray(color_frame.get_data())
                
                # 获取深度图像
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame:
                    continue
                depth_img = np.asanyarray(depth_frame.get_data())
                
                # 应用时域滤波
                depth_filtered = self.temporal_filter.process(depth_img)
                
                # 可视化深度图 (归一化到0-255)
                if self.show_depth:
                    # 设置深度范围 (mm)
                    MIN_DEPTH = 20
                    MAX_DEPTH = 2000
                    depth_clipped = np.clip(depth_filtered, MIN_DEPTH, MAX_DEPTH)
                    depth_normalized = ((depth_clipped - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH) * 255).astype(np.uint8)
                    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                else:
                    depth_colormap = None
                
                # 在彩色图像上添加信息
                info_text = f"Resolution: {self.width}x{self.height} | Frame: {frame_count}"
                cv2.putText(color_img, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示图像
                if self.show_depth and depth_colormap is not None:
                    # 水平拼接彩色图和深度图
                    combined = np.hstack([color_img, depth_colormap])
                    cv2.imshow("RealSense Viewer - Color + Depth", combined)
                else:
                    cv2.imshow("RealSense Viewer - Color", color_img)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[Info] Quitting...")
                    break
                elif key == ord('s'):
                    # 保存快照
                    filename = f"realsense_snapshot_{frame_count}.png"
                    if self.show_depth and depth_colormap is not None:
                        cv2.imwrite(filename, combined)
                    else:
                        cv2.imwrite(filename, color_img)
                    print(f"[Info] Snapshot saved: {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n[Info] Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """停止相机"""
        self.running = False
        if self.pipeline is not None:
            self.pipeline.stop()
            print("[Info] Pipeline stopped")
        cv2.destroyAllWindows()
        print("[Info] Viewer closed")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='RealSense Camera Real-time Viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 默认模式 (只显示彩色图)
  python tools/demo/realsense_viewer.py
  
  # 同时显示彩色图和深度图
  python tools/demo/realsense_viewer.py --show-depth
  
  # 自定义分辨率和帧率
  python tools/demo/realsense_viewer.py --width 1280 --height 720 --fps 30
  
  # 高分辨率模式
  python tools/demo/realsense_viewer.py --width 1920 --height 1080 --fps 30
        """
    )
    
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS (default: 30)")
    parser.add_argument("--show-depth", action="store_true", help="Show depth map alongside color image")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("="*60)
    print("RealSense Camera Viewer")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Resolution: {args.width}x{args.height}")
    print(f"  - FPS: {args.fps}")
    print(f"  - Show Depth: {args.show_depth}")
    print("="*60)
    
    # 创建查看器
    viewer = RealSenseViewer(
        width=args.width,
        height=args.height,
        fps=args.fps,
        show_depth=args.show_depth
    )
    
    # 运行
    viewer.run()
    
    print("="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
