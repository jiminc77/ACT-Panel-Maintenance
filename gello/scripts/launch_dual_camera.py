from dataclasses import dataclass, field
from typing import List, Tuple

import tyro

from gello.cameras.webcam_camera import WebcamCamera
from gello.zmq_core.camera_node import ZMQServerCamera


@dataclass
class Args:
    """이중 웹캠을 ZMQ 서버로 공개하기 위한 CLI 인자."""

    devices: List[str] = field(default_factory=lambda: ["/dev/video0", "/dev/video4"])
    ports: List[int] = field(default_factory=lambda: [7001, 8001])
    hostname: str = "0.0.0.0"
    width: int = 640
    height: int = 480
    flip: bool = False


def _start_camera_server(device: str, port: int, args: Args) -> Tuple[WebcamCamera, ZMQServerCamera]:
    camera = WebcamCamera(
        device=device,
        flip=args.flip,
        width=args.width,
        height=args.height,
    )
    server = ZMQServerCamera(camera=camera, port=port, host=args.hostname)
    return camera, server


def main(args: Args):
    if len(args.devices) != len(args.ports):
        raise ValueError("devices 개수와 ports 개수가 같아야 합니다.")

    import threading
    import time

    cameras: List[WebcamCamera] = []
    servers: List[ZMQServerCamera] = []
    threads: List[threading.Thread] = []

    for device, port in zip(args.devices, args.ports):
        camera, server = _start_camera_server(device, port, args)
        cameras.append(camera)
        servers.append(server)
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()
        threads.append(thread)
        print(f"Started camera server on port {port} for {device}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Camera servers interrupted, shutting down...")
    finally:
        for server in servers:
            server.stop()
        for thread in threads:
            thread.join(timeout=1.0)
        for camera in cameras:
            camera.release()


if __name__ == "__main__":
    main(tyro.cli(Args))
