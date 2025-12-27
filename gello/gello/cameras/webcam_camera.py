"""
웹캠(USB 카메라)에서 이미지를 읽는 카메라 드라이버

read_webcam.py의 방식을 참고하여 CameraDriver 프로토콜을 구현합니다.
"""

import glob
import os
import threading
from typing import List, Optional, Tuple

import numpy as np

from gello.cameras.camera import CameraDriver


def find_video_devices() -> List[str]:
    """시스템에서 사용 가능한 비디오 장치를 찾습니다.
    
    Returns:
        비디오 장치 경로 리스트 (예: ['/dev/video0', '/dev/video1', ...])
    """
    devices = sorted(glob.glob("/dev/video*"))
    return devices


class WebcamCamera(CameraDriver):
    """웹캠(USB 카메라)에서 이미지를 읽는 카메라 드라이버.
    
    OpenCV를 사용하여 V4L2 장치에서 이미지를 읽습니다.
    Depth 이미지는 제공되지 않으므로 더미 데이터를 반환합니다.
    """
    
    def __repr__(self) -> str:
        return f"WebcamCamera(device={self._device})"
    
    def __init__(
        self,
        device: Optional[str] = None,
        flip: bool = False,
        width: int = 640,
        height: int = 480,
    ):
        """WebcamCamera 초기화.
        
        Args:
            device: 비디오 장치 경로 (예: '/dev/video0'). None이면 자동으로 찾습니다.
            flip: 이미지를 180도 회전할지 여부
            width: 이미지 너비
            height: 이미지 높이
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV가 설치되어 있지 않습니다. 다음 명령으로 설치하세요:\n"
                "pip install opencv-python"
            )
        
        self._cv2 = cv2
        self._flip = flip
        self._width = width
        self._height = height
        
        # 장치 찾기
        if device is None:
            devices = find_video_devices()
            if not devices:
                raise RuntimeError(
                    "비디오 장치를 찾을 수 없습니다. "
                    "카메라가 연결되어 있고 UVC 모드로 설정되어 있는지 확인하세요."
                )
            device = devices[0]
            print(f"자동으로 장치를 찾았습니다: {device}")
        
        self._device = device
        
        # 카메라 열기
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없습니다: {device}")
        
        # 카메라 최적화 설정
        # 버퍼 크기를 1로 설정하여 최신 프레임만 유지 (지연 최소화)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 해상도 설정
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # FPS 설정 (가능한 경우)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        
        # 실제 해상도 확인
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"카메라 해상도: {actual_width}x{actual_height}, FPS: {actual_fps}")
        
        # 백그라운드 스레드에서 이미지 읽기 (성능 최적화)
        self._latest_frame = None
        self._latest_depth = None
        self._frame_lock = threading.Lock()
        self._stop_thread = threading.Event()
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
    
    def _read_loop(self):
        """백그라운드에서 계속 프레임을 읽는 루프"""
        import time
        while not self._stop_thread.is_set():
            ret, frame = self._cap.read()
            if ret:
                # BGR -> RGB 변환 (view 사용, 복사 없음)
                rgb_image = frame[:, :, ::-1]
                
                # 180도 회전 (필요한 경우)
                if self._flip:
                    rgb_image = self._cv2.rotate(rgb_image, self._cv2.ROTATE_180)
                
                # Depth 이미지 생성 (더미 데이터) - 매번 생성하지 않고 재사용 가능
                h, w = rgb_image.shape[:2]
                depth_image = np.zeros((h, w, 1), dtype=np.uint16)
                
                # 최신 프레임 업데이트 (복사해서 저장)
                with self._frame_lock:
                    self._latest_frame = rgb_image.copy()
                    self._latest_depth = depth_image.copy()
            else:
                # 읽기 실패 시 잠시 대기
                time.sleep(0.001)
    
    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """카메라에서 프레임을 읽습니다.
        
        Args:
            img_size: 반환할 이미지 크기 (width, height). None이면 원본 크기 반환.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (RGB 이미지, Depth 이미지)
                - RGB 이미지: shape=(H, W, 3), dtype=uint8, RGB 순서
                - Depth 이미지: shape=(H, W, 1), dtype=uint16, 더미 데이터 (모두 0)
        """
        # 최신 프레임 가져오기 (락 사용)
        with self._frame_lock:
            if self._latest_frame is None or self._latest_depth is None:
                # 아직 프레임이 없으면 동기적으로 읽기
                ret, frame = self._cap.read()
                if not ret:
                    raise RuntimeError(f"카메라에서 프레임을 읽을 수 없습니다: {self._device}")
                rgb_image = frame[:, :, ::-1]
                if self._flip:
                    rgb_image = self._cv2.rotate(rgb_image, self._cv2.ROTATE_180)
                h, w = rgb_image.shape[:2]
                depth_image = np.zeros((h, w, 1), dtype=np.uint16)
            else:
                # 최신 프레임 복사
                rgb_image = self._latest_frame.copy()
                depth_image = self._latest_depth.copy()
        
        # 이미지 크기 조정
        # img_size가 지정되면 그 크기로, 아니면 초기화 시 설정한 해상도로 리사이즈
        target_size = img_size if img_size is not None else (self._width, self._height)
        current_height, current_width = rgb_image.shape[:2]
        
        # 현재 크기와 목표 크기가 다르면 리사이즈
        if current_width != target_size[0] or current_height != target_size[1]:
            rgb_image = self._cv2.resize(
                rgb_image, target_size, interpolation=self._cv2.INTER_LINEAR
            )
            depth_image = self._cv2.resize(
                depth_image, target_size, interpolation=self._cv2.INTER_NEAREST
            )
            if len(depth_image.shape) == 2:
                depth_image = depth_image[:, :, None]
        
        return rgb_image, depth_image
    
    def release(self) -> None:
        """카메라 리소스를 해제합니다."""
        # 스레드 종료
        if hasattr(self, "_stop_thread"):
            self._stop_thread.set()
        if hasattr(self, "_read_thread") and self._read_thread.is_alive():
            self._read_thread.join(timeout=1.0)
        
        # 카메라 해제
        if hasattr(self, "_cap") and self._cap is not None:
            self._cap.release()
    
    def __del__(self):
        """소멸자에서 카메라 리소스 해제."""
        self.release()


def get_available_devices() -> List[str]:
    """사용 가능한 비디오 장치 목록을 반환합니다.
    
    Returns:
        비디오 장치 경로 리스트
    """
    return find_video_devices()


if __name__ == "__main__":
    # 테스트 코드
    import time
    
    print("사용 가능한 비디오 장치:")
    devices = get_available_devices()
    for i, dev in enumerate(devices):
        print(f"  {i}: {dev}")
    
    if not devices:
        print("비디오 장치가 없습니다.")
        exit(1)
    exit()
    # 첫 번째 장치 사용
    camera = WebcamCamera(device=devices[1], flip=False)
    
    print("\n카메라 테스트 시작 (5초간 프레임 읽기)...")
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < 5:
            rgb, depth = camera.read()
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  {frame_count} 프레임 읽음... (shape: {rgb.shape})")
    except KeyboardInterrupt:
        pass
    
    print(f"\n테스트 완료: 총 {frame_count} 프레임 읽음")
    print(f"평균 FPS: {frame_count / 5:.2f}")
    
    camera.release()

