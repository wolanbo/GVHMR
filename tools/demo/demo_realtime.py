import torch
import numpy as np
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import deque
import zmq
import json
import base64
from io import BytesIO
import sys
import threading
from queue import Queue, Empty

import hydra

from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import get_writer, merge_videos_horizontal
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum


CRF = 23  # Video quality setting
WINDOW_SIZE = 30  # Fixed sliding window size
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 2000  # 2000mm


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


def realsense_producer(queue, width=640, height=480, fps=30):
    """RealSense相机数据采集线程
    
    Args:
        queue: 帧数据队列
        width: 采集宽度
        height: 采集高度
        fps: 采集帧率
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        Log.error("[RealSense] pyrealsense2 not installed. Please run: pip install pyrealsense2")
        return
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    
    # 获取相机内参
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    Log.info(f"[RealSense] Camera intrinsics - fx:{fx:.2f}, fy:{fy:.2f}, cx:{cx:.2f}, cy:{cy:.2f}")
    
    temporal_filter = TemporalFilter(alpha=0.5)
    
    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            
            # 获取彩色图像
            color_img = np.asanyarray(aligned.get_color_frame().get_data())
            
            # 获取并处理深度图像
            depth_img = np.asanyarray(aligned.get_depth_frame().get_data())
            depth_filtered = temporal_filter.process(depth_img)
            
            # 如果队列满了,丢弃最旧的帧
            if queue.full():
                try:
                    queue.get_nowait()
                except Empty:
                    pass
            
            queue.put((color_img, depth_filtered, (fx, fy, cx, cy)))
    finally:
        pipeline.stop()
        Log.info("[RealSense] Pipeline stopped")


def setup_zmq_publisher(port=5555):
    """Setup ZeroMQ publisher socket"""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    Log.info(f"[ZMQ] Publisher bound to port {port}")
    return context, socket


def send_frame_data(socket, frame_idx, pred_data, metadata=None):
    """Send frame prediction data via ZMQ
    
    Args:
        socket: ZMQ publisher socket
        frame_idx: Current frame index
        pred_data: Dictionary containing SMPL parameters for current frame
        metadata: Optional metadata (timestamp, fps, etc.)
    """
    try:
        # Prepare message payload
        message = {
            'frame_idx': int(frame_idx),
            'timestamp': None,  # Can be added if needed
            'smpl_params_global': {},
            'smpl_params_incam': {},
        }
        
        # Convert tensors to lists for JSON serialization
        for key in ['smpl_params_global', 'smpl_params_incam']:
            if key in pred_data:
                for param_name, param_tensor in pred_data[key].items():
                    if isinstance(param_tensor, torch.Tensor):
                        message[key][param_name] = param_tensor.squeeze(0).cpu().numpy().tolist()
                    else:
                        message[key][param_name] = param_tensor
        
        # Add metadata if provided
        if metadata:
            message['metadata'] = metadata
        
        # Serialize to JSON
        json_str = json.dumps(message)
        
        # Send with topic prefix for filtering
        topic = "gvhmr.frame"
        socket.send_string(f"{topic} {json_str}")
        
        return True
    except Exception as e:
        Log.error(f"[ZMQ] Failed to send frame {frame_idx}: {e}")
        return False


def parse_args():
    """Parse command line arguments for video file or RealSense processing"""
    parser = argparse.ArgumentParser(
        description='GVHMR Real-time Processing with Sliding Window + ZMQ Streaming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file
  python tools/demo/demo_realtime.py --video test.mp4
  
  # Use RealSense camera
  python tools/demo/demo_realtime.py --realsense
  
  # RealSense with custom resolution
  python tools/demo/demo_realtime.py --realsense --width 1280 --height 720 --fps 30
  
  # With ZMQ streaming
  python tools/demo/demo_realtime.py --video test.mp4 --zmq_port 5555
  
  # Static camera mode
  python tools/demo/demo_realtime.py --video test.mp4 -s --max_frames 100
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="Input video file path")
    input_group.add_argument("--realsense", action="store_true", help="Use RealSense camera as input")
    
    parser.add_argument("--output_root", type=str, default="outputs/realtime", help="Output directory (for optional saving)")
    parser.add_argument("-s", "--static_cam", action="store_true", help="Use static camera mode (skip SLAM/VO)")
    parser.add_argument("--f_mm", type=int, default=None, help="Focal length in mm (e.g., 24 for standard)")
    parser.add_argument("--verbose", action="store_true", help="Show intermediate results")
    parser.add_argument("--max_frames", type=int, default=-1, help="Max frames to process (-1 for all)")
    parser.add_argument("--show_preview", action="store_true", help="Show real-time preview window")
    parser.add_argument("--zmq_port", type=int, default=5555, help="ZeroMQ publisher port (default: 5555)")
    parser.add_argument("--save_results", action="store_true", help="Also save results to disk (default: only stream via ZMQ)")
    
    # RealSense specific parameters
    parser.add_argument("--width", type=int, default=640, help="RealSense capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="RealSense capture height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="RealSense capture FPS (default: 30)")
    
    args = parser.parse_args()
    
    # Validate input
    if args.video:
        video_path = Path(args.video)
        assert video_path.exists(), f"Video file not found: {video_path}"
        Log.info(f"[Input Mode] Video file: {args.video}")
    elif args.realsense:
        Log.info(f"[Input Mode] RealSense camera ({args.width}x{args.height}@{args.fps}fps)")
    
    return args


def load_config(args):
    """Load configuration for GVHMR model"""
    from datetime import datetime
    from hydra import initialize_config_module, compose
    from hmr4d.configs import register_store_gvhmr
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate video_name based on input source
    if args.realsense:
        video_name = f"realsense_{args.width}x{args.height}_{timestamp}"
        Log.info(f"[Input]: RealSense camera ({args.width}x{args.height}@{args.fps}fps)")
    else:
        video_name = f"video_{Path(args.video).stem}_{timestamp}"
        Log.info(f"[Input]: Video file: {args.video}")
    
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_name}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"use_dpvo=False",
            f"output_root={args.output_root}",
        ]
        if args.f_mm is not None:
            overrides.append(f"f_mm={args.f_mm}")

        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    Log.info(f"[Output Dir]: {cfg.output_dir}")
    if args.save_results:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    return cfg


def process_video_sliding_window(cap_or_queue, cfg, args, width, height, fps, max_frames, start_time, zmq_socket, use_realsense=False, realsense_intrinsics=None):
    """Process video/RealSense with fixed-size sliding window and stream via ZMQ
    
    Args:
        cap_or_queue: cv2.VideoCapture object (video) or Queue (RealSense)
        use_realsense: Whether using RealSense input
        realsense_intrinsics: RealSense camera intrinsics (fx, fy, cx, cy)
    """
    Log.info(f"[Processing] Starting sliding window processing (window_size={WINDOW_SIZE})")
    Log.info(f"[ZMQ] Streaming enabled on port {args.zmq_port}")
    if not args.save_results:
        Log.info("[Storage] Results will NOT be saved to disk (only streamed via ZMQ)")
    
    # ===== Initialize Preprocessing Modules ===== #
    Log.info("[Preprocess] Initializing modules...")
    tracker = Tracker()
    vitpose_extractor = VitPoseExtractor()
    extractor = Extractor()
    
    # ===== Initialize Model ===== #
    Log.info("[Model] Loading GVHMR model...")
    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    Log.info("[Model] Model loaded successfully")
    
    # ===== Storage for results (only if save_results is True) ===== #
    all_preds = [] if args.save_results else None
    total_processed = 0
    frame_idx = 0
    
    # Sliding window buffers
    window_frames = deque(maxlen=WINDOW_SIZE)
    window_bbx_xyxy = deque(maxlen=WINDOW_SIZE)
    window_kp2d = deque(maxlen=WINDOW_SIZE)
    window_vit_features = deque(maxlen=WINDOW_SIZE)
    
    # Video writers (only if save_results is True)
    raw_writer = None
    if args.save_results:
        raw_video_path = Path(cfg.output_dir) / "raw_video.mp4"
        raw_writer = get_writer(str(raw_video_path), fps=fps, crf=CRF)
    
    Log.info("[Inference] Processing with sliding window...")
    Log.info("Press 'q' to stop inference")
    
    try:
        while True:
            # Read frame based on input source
            if use_realsense:
                # RealSense mode: get from queue
                try:
                    frame_bgr, depth_img, cam_intrinsics = cap_or_queue.get(timeout=2.0)
                    realsense_intrinsics = cam_intrinsics  # Update intrinsics
                except Empty:
                    Log.warning("[RealSense] Timeout waiting for frame")
                    continue
                
                if frame_bgr is None:
                    Log.warning("[RealSense] Received None frame")
                    continue
                    
                ret = True
            else:
                # Video mode: read from capture
                ret, frame_bgr = cap_or_queue.read()
                if not ret:
                    Log.info(f"[Video] Reached end of video at frame {frame_idx}")
                    break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Save raw frame (optional)
            if raw_writer is not None:
                raw_writer.write_frame(frame_bgr)
            
            # ===== Real-time Tracking ===== #
            track_results = tracker.yolo.track(
                frame_bgr,
                device="cuda",
                conf=0.5,
                classes=0,
                verbose=False,
                stream=False
            )
            
            bbx_xyxy_frame = None
            kp2d_frame = None
            vit_feat_frame = None
            
            if len(track_results) > 0 and track_results[0].boxes.id is not None:
                track_ids = track_results[0].boxes.id.int().cpu().tolist()
                bbx_xyxy = track_results[0].boxes.xyxy.cpu().numpy()
                
                if len(track_ids) > 0:
                    # Sort by area, take largest
                    areas = [(bbx_xyxy[j, 2] - bbx_xyxy[j, 0]) * (bbx_xyxy[j, 3] - bbx_xyxy[j, 1]) 
                             for j in range(len(track_ids))]
                    best_idx = np.argmax(areas)
                    
                    bbx_xyxy_best = torch.tensor(bbx_xyxy[best_idx:best_idx+1]).float()
                    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy_best, base_enlarge=1.2).float()
                    
                    bbx_xyxy_frame = bbx_xyxy_best[0]
                    
                    # Extract 2D Pose
                    kp2d_frame = vitpose_extractor.extract_single_frame(frame_rgb, bbx_xys)
                    
                    # Extract Visual Features
                    vit_feat_frame = extractor.extract_single_frame_features(frame_rgb, bbx_xys)
                    
                    # Show Preview (Optional)
                    if args.show_preview or args.verbose:
                        preview = draw_bbx_xyxy_on_image_batch(
                            bbx_xyxy_best, 
                            frame_bgr.copy()[None]
                        )[0]
                        preview = draw_coco17_skeleton_batch(
                            preview[None], 
                            kp2d_frame[None], 
                            0.5
                        )[0]
                        
                        cv2.putText(preview, f"Frame: {frame_idx}/{max_frames}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("GVHMR Sliding Window", preview)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            Log.info("[Inference] User pressed 'q', stopping...")
                            break
                else:
                    Log.warning(f"[Frame {frame_idx}] No person detected")
            else:
                Log.warning(f"[Frame {frame_idx}] Tracking failed")
            
            # If detection failed, use dummy data
            if bbx_xyxy_frame is None:
                h, w = frame_rgb.shape[:2]
                bbx_xyxy_frame = torch.tensor([w*0.25, h*0.25, w*0.75, h*0.75]).float()
                kp2d_frame = np.zeros((17, 3), dtype=np.float32)
                vit_feat_frame = np.zeros(1024, dtype=np.float32)
            
            # Add to sliding window
            window_frames.append(frame_rgb)
            window_bbx_xyxy.append(bbx_xyxy_frame)
            window_kp2d.append(kp2d_frame)
            window_vit_features.append(vit_feat_frame)
            
            current_window_size = len(window_frames)
            
            # Only start inference when we have enough frames
            if current_window_size < WINDOW_SIZE:
                Log.info(f"[Window] Collecting frames: {current_window_size}/{WINDOW_SIZE}")
                frame_idx += 1
                continue
            
            # ===== Run Inference on Current Window ===== #
            tic_infer = Log.sync_time()
            
            # Stack window data
            bbx_xyxy_stack = torch.stack(list(window_bbx_xyxy))  # (W, 4)
            bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy_stack, base_enlarge=1.2)  # (W, 3)
            
            # kp2d and vit_features might be numpy or tensor, handle both
            kp2d_list = []
            vit_feat_list = []
            for k, v in zip(window_kp2d, window_vit_features):
                if isinstance(k, np.ndarray):
                    kp2d_list.append(torch.from_numpy(k).float())  # 确保float32
                else:
                    kp2d_list.append(k.float() if isinstance(k, torch.Tensor) else k)
                
                if isinstance(v, np.ndarray):
                    vit_feat_list.append(torch.from_numpy(v).float())  # 确保float32
                else:
                    vit_feat_list.append(v.float() if isinstance(v, torch.Tensor) else v)
            
            # 确保所有张量在同一设备上 (统一移到CPU再stack,避免设备不一致)
            kp2d_list = [t.cpu() for t in kp2d_list]
            vit_feat_list = [t.cpu() for t in vit_feat_list]
            
            kp2d_stack = torch.stack(kp2d_list)  # (W, 17, 3)
            vit_features_stack = torch.stack(vit_feat_list)  # (W, 1024)
            
            # Calculate camera parameters
            K_fullimg = estimate_K(width, height).repeat(WINDOW_SIZE, 1, 1)
            
            # Use RealSense intrinsics if available
            if use_realsense and realsense_intrinsics is not None:
                fx, fy, cx, cy = realsense_intrinsics
                K_fullimg = torch.tensor([
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                ]).repeat(WINDOW_SIZE, 1, 1).float()
                Log.debug(f"[Camera] Using RealSense intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            elif args.f_mm is not None:
                K_fullimg = create_camera_sensor(width, height, args.f_mm)[2].repeat(WINDOW_SIZE, 1, 1)
            
            R_w2c = torch.eye(3).repeat(WINDOW_SIZE, 1, 1)
            cam_angvel = compute_cam_angvel(R_w2c)
            
            data = {
                "length": torch.tensor(WINDOW_SIZE),
                "bbx_xys": bbx_xys,
                "kp2d": kp2d_stack,
                "K_fullimg": K_fullimg,
                "cam_angvel": cam_angvel,
                "f_imgseq": vit_features_stack,
            }
            
            # Move to GPU
            data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            
            # Run prediction
            pred = model.predict(data, static_cam=args.static_cam)
            pred = detach_to_cpu(pred)
            
            infer_time = Log.sync_time() - tic_infer
            
            # Extract the LAST frame's prediction (the newest one)
            last_frame_pred = {}
            for key in pred:
                if isinstance(pred[key], dict):
                    last_frame_pred[key] = {}
                    for sub_key in pred[key]:
                        if isinstance(pred[key][sub_key], torch.Tensor):
                            last_frame_pred[key][sub_key] = pred[key][sub_key][-1:]  # Take last frame
                        else:
                            last_frame_pred[key][sub_key] = pred[key][sub_key]
                elif isinstance(pred[key], torch.Tensor):
                    last_frame_pred[key] = pred[key][-1:]  # Take last frame
                else:
                    last_frame_pred[key] = pred[key]
            
            # Accumulate results (only if saving)
            if args.save_results and all_preds is not None:
                all_preds.append(last_frame_pred)
            
            total_processed += 1
            
            # ===== Stream via ZMQ ===== #
            metadata = {
                'fps': fps,
                'infer_time': infer_time,
                'window_size': WINDOW_SIZE,
            }
            send_success = send_frame_data(zmq_socket, frame_idx, last_frame_pred, metadata)
            
            elapsed = Log.sync_time() - start_time
            avg_fps = total_processed / elapsed if elapsed > 0 else 0
            
            status = "✓" if send_success else "✗"
            Log.info(f"[Progress] Frame {frame_idx+1}/{max_frames}, "
                    f"Window: {current_window_size}, "
                    f"Infer: {infer_time:.3f}s, "
                    f"ZMQ: {status}, "
                    f"Total: {total_processed}, "
                    f"Avg FPS: {avg_fps:.2f}")
            
            frame_idx += 1
            
            # Check termination
            if max_frames > 0 and total_processed >= max_frames:
                Log.info(f"[Inference] Reached max frames ({max_frames})")
                break
    
    except KeyboardInterrupt:
        Log.info("[Inference] Interrupted by user")
    finally:
        # Release video capture (only for video mode, not Queue)
        if not use_realsense:
            cap_or_queue.release()
        
        if raw_writer is not None:
            raw_writer.close()
        
        # Destroy all windows
        cv2.destroyAllWindows()
    
    # ===== Final Post-processing (only if saving) ===== #
    if args.save_results and all_preds and len(all_preds) > 0:
        Log.info(f"\n{'='*60}")
        Log.info(f"[Post-process] Stacking accumulated results...")
        
        # Stack all predictions
        final_pred = {}
        for key in ['smpl_params_global', 'smpl_params_incam']:
            final_pred[key] = {}
            for sub_key in all_preds[0][key]:
                stacked = torch.cat([p[key][sub_key] for p in all_preds], dim=0)
                final_pred[key][sub_key] = stacked
        
        if 'K_fullimg' in all_preds[0]:
            final_pred['K_fullimg'] = torch.cat([p['K_fullimg'] for p in all_preds], dim=0)
        
        # Save predictions
        pred_path = Path(cfg.output_dir) / "gvhmr_results.pt"
        torch.save(final_pred, pred_path)
        Log.info(f"[Save] Predictions saved to {pred_path}")
        
        # Optionally render videos (commented out by default for performance)
        # Log.info("[Render] Rendering final videos...")
        # render_incam(cfg, final_pred, width, height, fps, args.static_cam)
        # render_global(cfg, final_pred, width, height, fps)
    
    total_time = Log.sync_time() - start_time
    Log.info(f"\n{'='*60}")
    Log.info(f"[Complete] Total frames processed: {total_processed}")
    Log.info(f"[Complete] Total time: {total_time:.2f}s")
    Log.info(f"[Complete] Average FPS: {total_processed/total_time:.2f}")
    Log.info(f"[Complete] Frames streamed via ZMQ: {total_processed}")
    Log.info(f"{'='*60}")


@torch.no_grad()
def main():
    """Main function for video/RealSense processing with ZMQ streaming"""
    args = parse_args()
    cfg = load_config(args)
    
    Log.info(f"[Realtime Inference] Start!")
    tic = Log.time()
    
    # ===== Setup ZMQ Publisher ===== #
    zmq_context, zmq_socket = setup_zmq_publisher(args.zmq_port)
    
    # ===== Initialize Input Source ===== #
    use_realsense = args.realsense
    cap_or_queue = None
    width, height, fps = 640, 480, 30  # defaults
    realsense_intrinsics = None
    producer_thread = None
    
    if use_realsense:
        # RealSense mode
        Log.info(f"[Input] Initializing RealSense camera...")
        frame_queue = Queue(maxsize=10)
        
        # Start producer thread
        producer_thread = threading.Thread(
            target=realsense_producer,
            args=(frame_queue, args.width, args.height, args.fps),
            daemon=True
        )
        producer_thread.start()
        
        cap_or_queue = frame_queue
        width, height, fps = args.width, args.height, args.fps
        
        Log.info(f"[Input] Type: RealSense Camera")
        Log.info(f"[Input] Resolution: {width}x{height}, FPS: {fps}")
        Log.info(f"[Input] Producer thread started")
    else:
        # Video file mode
        Log.info(f"[Input] Opening video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {args.video}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            fps = 30
        
        max_frames = args.max_frames if args.max_frames > 0 else total_frames
        max_frames = min(max_frames, total_frames) if total_frames > 0 else max_frames
        
        cap_or_queue = cap
        
        Log.info(f"[Input] Type: Video File")
        Log.info(f"[Input] Resolution: {width}x{height}, FPS: {fps:.1f}")
        Log.info(f"[Input] Total frames: {total_frames}, Duration: {total_frames/fps:.1f}s")
        Log.info(f"[Input] Processing {max_frames} frames")
    
    Log.info(f"[Config] Sliding window size: {WINDOW_SIZE}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        Log.info(f"[GPU]: {gpu_name}")
        Log.info(f"[GPU]: {gpu_props}")
    
    # Process with sliding window and ZMQ streaming
    process_video_sliding_window(
        cap_or_queue, cfg, args, width, height, fps, 
        max_frames if not use_realsense else args.max_frames, 
        tic, zmq_socket, 
        use_realsense=use_realsense,
        realsense_intrinsics=realsense_intrinsics
    )
    
    # Cleanup
    if use_realsense:
        if producer_thread and producer_thread.is_alive():
            Log.info("[Cleanup] Waiting for producer thread to finish...")
            producer_thread.join(timeout=2.0)
        Log.info("[Cleanup] RealSense resources released")
    else:
        cap_or_queue.release()
    
    zmq_socket.close()
    zmq_context.term()
    Log.info("[Complete] All done! ZMQ publisher closed.")


if __name__ == "__main__":
    main()
