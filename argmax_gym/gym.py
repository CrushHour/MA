import logging
import os
import sys
import threading
import time
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import zmq

from .runner import GymRunner, TcpRemoteGymRunner

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "messages"))

import capnp  # noqa: F401, E402
import gym_capnp  # noqa: E402


class Gym:
    context = zmq.Context()

    def __init__(
            self,
            env,
            batch_size,
            bazel_output_dir: Union[str, Path] = Path("/root/.cache/bazel/_bazel_root"),
            bazel_remote_cache: str = "http://bazel-remote.argmax.ai",
            logs_path: Union[str, Path] = Path("/tmp/argmax_gym_logs"),
            env_args: str = "",
            prefix: str = "vwml"
    ):
        self.env = env
        self.batch_size = batch_size
        self.uuid = uuid.uuid4()
        self.bazel_output_dir = bazel_output_dir
        if isinstance(self.bazel_output_dir, str):
            self.bazel_output_dir = Path(self.bazel_output_dir)

        self.bazel_remote_cache = bazel_remote_cache
        self.logs_path = logs_path
        if isinstance(self.logs_path, str):
            self.logs_path = Path(self.logs_path)

        self.env_args = env_args
        self.prefix = prefix

        self.record = False
        self.monitored_envs = 3

        self.sockets = []

        # Results buffer
        self._observations = [None for _ in range(self.batch_size)]
        self._rewards = np.zeros((self.batch_size,), dtype=np.float32)
        self._dones = np.zeros((self.batch_size,), dtype=np.bool)
        self._images = [None for _ in range(self.batch_size)]

        self.gym_runner = self._create_gym_runner(self.env, self.batch_size)
        self.gym_runner.start()
        self._setup_gym_watchdog()

        self._wait_until_envs_report_readiness()

        self._init_communication_channel()

    def __del__(self):
        self.close()

    def close(self):
        self.watchdog_active = False
        if hasattr(self, "gym_runner"):
            self.gym_runner.stop()

    def _setup_gym_watchdog(self):
        self.watchdog_active = True
        self.watchdog = threading.Thread(target=self.continuous_assert_envs_running, args=())
        self.watchdog.daemon = True
        self.watchdog.start()

    def _wait_until_envs_report_readiness(self):
        logging.info("Waiting for environments to report readiness...")

        for i in range(self.batch_size):
            try:
                socket = self.context.socket(zmq.REQ)
                socket.setsockopt(zmq.RCVTIMEO, 30 * 60000)
                socket.connect(self._communication_endpoint(i))
                socket.send_string("init")
                message = socket.recv()
                if message != b"init_confirmed":
                    raise ConnectionError("Connection couldn't be established with all environments")
                socket.close()
            except zmq.Again:
                raise ConnectionError("Connection couldn't be established with all environments")

        logging.info("All environments reported successful start-up.")

    def _init_communication_channel(self):
        logging.info("Initialize normal communication channel..")
        # Initialize connection for normal communication
        for i in range(self.batch_size):
            socket = self.context.socket(zmq.REQ)
            socket.connect(self._communication_endpoint(i))
            socket.setsockopt(zmq.RCVTIMEO, -1)
            self.sockets.append(socket)

        logging.info("Environment setup complete.")

    def continuous_assert_envs_running(self):
        while self.watchdog_active:
            time.sleep(5.)
            self.assert_envs_running()

    def assert_envs_running(self):
        is_running, idx = self.gym_runner.is_running()
        if not is_running and self.watchdog_active:
            # stop remaining envs, if any
            self.gym_runner.stop()
            # show logs of the failing env
            tail = self.gym_runner.get_recent_logs(idx)
            logging.error("".join(tail))

            raise RuntimeError("Simulation environment couldn't start/terminated")

    @abstractmethod
    def _create_gym_runner(self, env, batch_size) -> GymRunner:
        """Creation of a gym runner which spawns environments, either locally or remote."""
        pass

    @abstractmethod
    def _communication_endpoint(self, index) -> str:
        """Communication endpoint for environment communication."""
        pass

    def reset(self, record=False):
        self.record = record

        for i in range(self.batch_size):
            self._send_reset(i)

        for i in range(self.batch_size):
            (self._observations[i], self._rewards[i],
             self._dones[i], self._images[i]) = self._receive_observation(i)

        return self._observations, self._images

    def step(self, control: np.ndarray):
        batch_size = control.shape[0]
        assert batch_size <= self.batch_size, (f"Required batch size {batch_size} bigger "
                                               f"than instantiated envs {self.batch_size}")

        for i in range(batch_size):
            if not self._dones[i]:
                self._send_control(i, control[i])

        for i in range(batch_size):
            if not self._dones[i]:
                (self._observations[i], self._rewards[i],
                 self._dones[i], self._images[i]) = self._receive_observation(i)

        return self._observations, self._rewards, self._dones, self._images

    def end_episode(self, control: np.ndarray):
        """Send a control message with done=True to indicate the end of an episode."""
        batch_size = control.shape[0]
        assert batch_size <= self.batch_size, (f"Required batch size {batch_size} bigger "
                                               f"than instantiated envs {self.batch_size}")

        for i in range(batch_size):
            self._send_control(i, control[i], done=True)

        for i in range(batch_size):
            self._receive_observation(i)

    def _receive_observation(self, i):
        enc_observation_msg = self.sockets[i].recv()
        return_msg = gym_capnp.GymReturnProto.from_bytes(enc_observation_msg)

        observation = np.array(return_msg.observation, dtype=np.float32)
        reward = np.float(return_msg.reward)
        done = np.bool(return_msg.done)
        image = None

        if i < self.monitored_envs:
            if hasattr(return_msg, "image") and hasattr(return_msg, "imageBuffer"):
                image = np.array(return_msg.imageBuffer, dtype=np.uint8)
                image = image.reshape(
                    return_msg.image.rows, return_msg.image.cols, return_msg.image.channels)

        return observation, reward, done, image

    def _send_reset(self, env_id):
        action_msg = gym_capnp.GymControlProto.new_message()
        action_msg.reset = True
        action_msg.render = self.record and env_id < self.monitored_envs

        self.sockets[env_id].send(action_msg.to_bytes())

    def _send_control(self, env_id, control, done=False):
        action_msg = gym_capnp.GymControlProto.new_message()

        control_list = action_msg.init("action", len(control))
        action_msg.done = done
        for i in range(len(control)):
            control_list[i] = float(control[i])

        action_msg.reset = False
        action_msg.render = self.record and env_id < self.monitored_envs

        self.sockets[env_id].send(action_msg.to_bytes())


class TcpRemoteGym(Gym):
    def __init__(
            self,
            env,
            hosts,
            target_device,
            bazel_output_dir: Union[str, Path] = Path("/root/.cache/bazel/_bazel_root"),
            bazel_remote_cache: str = "http://bazel-remote.argmax.ai",
            logs_path: Union[str, Path] = Path("/tmp/argmax_gym_logs"),
            port=55000,
            env_args: str = "",
            prefix: str = "vwml"
    ):
        self.hosts = hosts
        self.target_device = target_device
        self.port = port
        super().__init__(
            env,
            batch_size=len(self.hosts),
            bazel_output_dir=bazel_output_dir,
            bazel_remote_cache=bazel_remote_cache,
            logs_path=logs_path,
            env_args=env_args,
            prefix=prefix)

    def _create_gym_runner(self, env, batch_size):
        return TcpRemoteGymRunner(
            env,
            hosts=self.hosts,
            target_device=self.target_device,
            port=self.port,
            bazel_output_dir=self.bazel_output_dir,
            bazel_remote_cache=self.bazel_remote_cache,
            logs_path=self.logs_path,
            env_args=self.env_args,
            prefix=self.prefix)

    def _communication_endpoint(self, index):
        return f"tcp://{self.hosts[index]}:{self.port + index}"
