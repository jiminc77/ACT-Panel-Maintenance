import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 50
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False
    
    # 웹캠 카메라 옵션
    use_webcam: bool = False  # 웹캠 사용 여부
    webcam_device: Optional[str] = None  # 단일 웹캠 장치 경로 (예: '/dev/video0'), None이면 자동 탐지
    webcam_name: str = "wrist"  # 단일 웹캠 이름 (observations에 {name}_rgb, {name}_depth로 저장됨)
    
    # 두 개의 웹캠 사용 (wrist와 base)
    webcam_wrist_device: Optional[str] = None  # wrist 웹캠 장치 경로 (예: '/dev/video0')
    webcam_base_device: Optional[str] = None  # base 웹캠 장치 경로 (예: '/dev/video1')
    
    # 이미지 최적화 옵션
    webcam_width: int = 1280  # 웹캠 이미지 너비 (기본값: 160, 작을수록 빠름)
    webcam_height: int = 720  # 웹캠 이미지 높이 (기본값: 120, 작을수록 빠름)
    webcam_read_interval: int = 1  # 이미지 읽기 주기 (N프레임마다 1번 읽기, 1이면 매 프레임)


def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        
        # 웹캠 카메라 추가
        if args.use_webcam:
            from gello.cameras.webcam_camera import WebcamCamera, get_available_devices
            
            # 두 개의 웹캠 사용 (wrist와 base) - 명시적으로 지정된 경우
            if args.webcam_wrist_device is not None or args.webcam_base_device is not None:
                # wrist 카메라 추가
                if args.webcam_wrist_device is not None:
                    try:
                        wrist_cam = WebcamCamera(
                            device=args.webcam_wrist_device,
                            flip=False,
                            width=args.webcam_width,
                            height=args.webcam_height
                        )
                        camera_clients["wrist"] = wrist_cam
                        print_color(
                            f"웹캠 카메라 추가됨: wrist (device: {wrist_cam._device})",
                            color="green"
                        )
                    except Exception as e:
                        print_color(
                            f"wrist 웹캠 카메라 초기화 실패: {e}",
                            color="red"
                        )
                
                # base 카메라 추가
                if args.webcam_base_device is not None:
                    try:
                        base_cam = WebcamCamera(
                            device=args.webcam_base_device,
                            flip=False,
                            width=args.webcam_width,
                            height=args.webcam_height
                        )
                        camera_clients["base"] = base_cam
                        print_color(
                            f"웹캠 카메라 추가됨: base (device: {base_cam._device})",
                            color="green"
                        )
                    except Exception as e:
                        print_color(
                            f"base 웹캠 카메라 초기화 실패: {e}",
                            color="red"
                        )
            
            # 단일 웹캠 사용 (기존 방식) - webcam_device가 명시된 경우
            elif args.webcam_device is not None:
                try:
                    webcam = WebcamCamera(
                        device=args.webcam_device,
                        flip=False,
                        width=args.webcam_width,
                        height=args.webcam_height
                    )
                    camera_clients[args.webcam_name] = webcam
                    print_color(
                        f"웹캠 카메라 추가됨: {args.webcam_name} (device: {webcam._device})",
                        color="green"
                    )
                except Exception as e:
                    print_color(
                        f"웹캠 카메라 초기화 실패: {e}",
                        color="red"
                    )
                    print("웹캠 없이 계속 진행합니다...")
            
            # 자동 탐지 모드: 두 개의 웹캠을 자동으로 찾아서 wrist와 base에 할당
            else:
                available_devices = get_available_devices()
                if len(available_devices) >= 2:
                    try:
                        wrist_cam = WebcamCamera(
                            device=available_devices[0],
                            flip=False,
                            width=args.webcam_width,
                            height=args.webcam_height
                        )
                        camera_clients["wrist1"] = wrist_cam
                        print_color(
                            f"웹캠 카메라 자동 추가됨: wrist1 (device: {available_devices[0]})",
                            color="green"
                        )
                        
                        base_cam = WebcamCamera(
                            device=available_devices[2],
                            flip=False,
                            width=args.webcam_width,
                            height=args.webcam_height
                        )
                        camera_clients["wrist2"] = base_cam
                        print_color(
                            f"웹캠 카메라 자동 추가됨: wrist2 (device: {available_devices[1]})",
                            color="green"
                        )
                    except Exception as e:
                        print_color(
                            f"웹캠 카메라 자동 초기화 실패: {e}",
                            color="red"
                        )
                elif len(available_devices) == 1:
                    print_color(
                        f"경고: 웹캠이 1개만 발견되었습니다. {args.webcam_name}에 추가합니다.",
                        color="yellow"
                    )
                    try:
                        webcam = WebcamCamera(
                            device=available_devices[0],
                            flip=False,
                            width=args.webcam_width,
                            height=args.webcam_height
                        )
                        camera_clients[args.webcam_name] = webcam
                        print_color(
                            f"웹캠 카메라 추가됨: {args.webcam_name} (device: {available_devices[0]})",
                            color="green"
                        )
                    except Exception as e:
                        print_color(
                            f"웹캠 카메라 초기화 실패: {e}",
                            color="red"
                        )
                else:
                    print_color(
                        "경고: 사용 가능한 웹캠이 없습니다.",
                        color="yellow"
                    )
        
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(
        robot_client,
        control_rate_hz=args.hz,
        camera_dict=camera_clients,
        camera_read_interval=args.webcam_read_interval
    )

    if args.bimanual:
        if args.agent == "gello":
            # dynamixel control box port map (to distinguish left and right gello)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT45BG45-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3FSJGP-if00-port0"
            left_agent = GelloAgent(port=left)
            right_agent = GelloAgent(port=right)
            agent = BimanualAgent(left_agent, right_agent)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            left_agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
            right_agent = SingleArmQuestAgent(
                robot_type=args.robot_type, which_hand="r"
            )
            agent = BimanualAgent(left_agent, right_agent)
            # raise NotImplementedError
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            left_agent = SpacemouseAgent(
                robot_type=args.robot_type, device_path=left_path, verbose=args.verbose
            )
            right_agent = SpacemouseAgent(
                robot_type=args.robot_type,
                device_path=right_path,
                verbose=args.verbose,
                invert_button=True,
            )
            agent = BimanualAgent(left_agent, right_agent)
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        # reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        # reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints_left = np.deg2rad([0, 0, 0, -90, 0, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, 0, 0, -90, 0, 90, 0, 0])
        
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        if args.agent == "gello":
            gello_port = args.gello_port
            if gello_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    gello_port = usb_ports[0]
                    print(f"using port {gello_port}")
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
            if args.start_joints is None:
                reset_joints = np.deg2rad(
                    [0, 0, 0, -90, 0, 90, 0, 0]
                )  # Change this to your own reset joints
            else:
                reset_joints = args.start_joints
            agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
            curr_joints = env.get_obs()["joint_positions"]
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
        elif args.agent == "dummy" or args.agent == "none":
            agent = DummyAgent(num_dofs=robot_client.num_dofs())
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    # going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.5).any():
        print("Action is too big")

        # print which joints are too big
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    if args.use_save_interface:
        from gello.data_utils.keyboard_interface import KBReset

        kb_interface = KBReset()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    save_path = None
    start_time = time.time()
    
    # 데이터 수집 frequency 측정을 위한 변수
    save_start_time = None
    save_frame_count = 0
    last_frequency_print_time = None
    frequency_print_interval = 5.0  # 5초마다 frequency 출력
    
    # 메모리 버퍼에 데이터 저장 (파일 I/O 없이)
    data_buffer = []  # (timestamp, obs, action) 튜플 리스트
    is_recording = False  # 현재 버퍼에 저장 중인지 여부
    
    try:
        while True:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "
            print_color(
                message,
                color="white",
                attrs=("bold",),
                end="",
                flush=True,
            )
            action = agent.act(obs)
            dt = datetime.datetime.now()
            if args.use_save_interface:
                state = kb_interface.update()
                if state == "start":
                    dt_time = datetime.datetime.now()
                    save_path = (
                        Path(args.data_dir).expanduser()
                        / args.agent
                        / dt_time.strftime("%m%d_%H%M%S")
                    )
                    save_path.mkdir(parents=True, exist_ok=True)
                    print(f"\n데이터 수집 시작: {save_path}")
                    # 저장 시작 시간 기록
                    save_start_time = time.time()
                    save_frame_count = 0
                    last_frequency_print_time = save_start_time
                    is_recording = True
                    data_buffer = []  # 버퍼 초기화
                    print_color("데이터를 메모리 버퍼에 저장 중... (q를 누르면 파일로 저장)", color="green")
                elif state == "save":
                    # 메모리 버퍼에 데이터 추가 (파일 저장 없음)
                    if is_recording:
                        # obs와 action을 복사해서 버퍼에 추가
                        obs_copy = {}
                        for k, v in obs.items():
                            if isinstance(v, np.ndarray):
                                obs_copy[k] = v.copy()
                            else:
                                obs_copy[k] = v
                        action_copy = action.copy() if isinstance(action, np.ndarray) else action
                        
                        data_buffer.append((dt, obs_copy, action_copy))
                        save_frame_count += 1
                        
                        # 주기적으로 frequency 출력
                        current_time = time.time()
                        if save_start_time is not None:
                            elapsed_time = current_time - save_start_time
                            if elapsed_time > 0:
                                current_frequency = save_frame_count / elapsed_time
                                
                                # 주기적으로 frequency 출력 (5초마다)
                                if (last_frequency_print_time is None or 
                                    current_time - last_frequency_print_time >= frequency_print_interval):
                                    print_color(
                                        f"\n[Frequency] 버퍼에 저장된 프레임: {save_frame_count}, "
                                        f"경과 시간: {elapsed_time:.2f}초, "
                                        f"현재 frequency: {current_frequency:.2f} Hz",
                                        color="cyan"
                                    )
                                    last_frequency_print_time = current_time
                elif state == "normal":
                    # 저장 종료: 버퍼에 있는 모든 데이터를 파일로 저장
                    if is_recording and len(data_buffer) > 0:
                        assert save_path is not None, "something went wrong"
                        print_color(
                            f"\n버퍼에 저장된 {len(data_buffer)}개 프레임을 파일로 저장 중...",
                            color="yellow"
                        )
                        
                        # 파일 저장 (백그라운드 스레드 사용 가능하지만 일단 동기로)
                        import threading
                        from queue import Queue
                        
                        save_queue = Queue()
                        save_complete = threading.Event()
                        
                        def save_worker():
                            """백그라운드에서 파일 저장"""
                            assert save_path is not None
                            saved = 0
                            for dt, obs, action in data_buffer:
                                save_frame(save_path, dt, obs, action)
                                saved += 1
                                if saved % 100 == 0:
                                    print(f"  저장 중... {saved}/{len(data_buffer)}")
                            save_complete.set()
                        
                        # 백그라운드 스레드에서 저장
                        save_thread = threading.Thread(target=save_worker, daemon=False)
                        save_thread.start()
                        
                        # 저장 완료 대기 (최대 60초)
                        save_complete.wait(timeout=60.0)
                        
                        if save_complete.is_set():
                            print_color(
                                f"저장 완료: {len(data_buffer)}개 프레임",
                                color="green"
                            )
                        else:
                            print_color(
                                "저장이 완료되지 않았습니다 (타임아웃)",
                                color="red"
                            )
                        
                        # 최종 frequency 출력
                        if save_start_time is not None:
                            elapsed_time = time.time() - save_start_time
                            if elapsed_time > 0:
                                final_frequency = save_frame_count / elapsed_time
                                print_color(
                                    f"\n[최종 Frequency] 총 저장된 프레임: {save_frame_count}, "
                                    f"총 경과 시간: {elapsed_time:.2f}초, "
                                    f"평균 frequency: {final_frequency:.2f} Hz",
                                    color="green",
                                    attrs=("bold",)
                                )
                                print_color(
                                    f"설정된 control rate: {args.hz} Hz",
                                    color="yellow"
                                )
                    
                    # 상태 초기화
                    is_recording = False
                    data_buffer = []
                    save_path = None
                    save_start_time = None
                    save_frame_count = 0
                    last_frequency_print_time = None
                else:
                    raise ValueError(f"Invalid state {state}")
            obs = env.step(action)
    except KeyboardInterrupt:
        print_color("\n프로그램 종료 중...", color="yellow")
        # 저장 중이었다면 버퍼에 있는 데이터 저장
        if is_recording and len(data_buffer) > 0 and save_path is not None:
            print_color(
                f"\n버퍼에 저장된 {len(data_buffer)}개 프레임을 파일로 저장 중...",
                color="yellow"
            )
            for dt, obs, action in data_buffer:
                save_frame(save_path, dt, obs, action)
            print_color(f"저장 완료: {len(data_buffer)}개 프레임", color="green")
            
            # 최종 frequency 출력
            if save_start_time is not None:
                elapsed_time = time.time() - save_start_time
                if elapsed_time > 0:
                    final_frequency = save_frame_count / elapsed_time
                    print_color(
                        f"\n[최종 Frequency] 총 저장된 프레임: {save_frame_count}, "
                        f"총 경과 시간: {elapsed_time:.2f}초, "
                        f"평균 frequency: {final_frequency:.2f} Hz",
                        color="green",
                        attrs=("bold",)
                    )
                    print_color(
                        f"설정된 control rate: {args.hz} Hz",
                        color="yellow"
                    )
    finally:
        # 웹캠 리소스 해제
        if args.use_webcam and not args.mock and 'camera_clients' in locals():
            for name, camera in camera_clients.items():
                if hasattr(camera, 'release'):
                    try:
                        camera.release()
                        print_color(f"웹캠 카메라 리소스 해제: {name}", color="green")
                    except Exception as e:
                        print_color(f"웹캠 카메라 리소스 해제 실패: {e}", color="red")
        


if __name__ == "__main__":
    main(tyro.cli(Args))
