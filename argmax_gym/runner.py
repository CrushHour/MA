import glob
import logging
import os
import signal
import subprocess
import sys
from abc import abstractmethod, ABC
from collections import deque
from pathlib import Path
from typing import Tuple, List, Union

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "messages"))

import capnp  # noqa: F401, E402


# deploy_path = "argmax_gym/envs"
deploy_path = "/root/deploy"

class GymRunner(ABC):
    """

    Parameters
    ----------
    env
        Environment to run. Corresponds to a bazel target/Isaac app name.
    batch_size
        Number of environment instances.
    websight_port
        Port where Isaac's websight can be reached.
    bazel_output_dir
        Set bazel's output directory for caching.
    env_args
        An argument string that is passed to the environment execution.
    """

    def __init__(
            self,
            env,
            batch_size: int = 1,
            websight_port: int = 3000,
            bazel_output_dir: Union[str, Path] = Path("/root/.cache/bazel/_bazel_root"),
            bazel_remote_cache: str = "http://bazel-remote.argmax.ai",
            logs_path: Union[str, Path] = Path("/tmp/argmax_gym_logs"),
            env_args: str = "",
            prefix: str = "vwml"
    ):
        self.env = env
        self.batch_size = batch_size
        self.bazel_output_dir = bazel_output_dir
        if isinstance(self.bazel_output_dir, str):
            self.bazel_output_dir = Path(self.bazel_output_dir)

        self.bazel_remote_cache = bazel_remote_cache
        self.bazel_remote_cache_arg = ""
        if self.bazel_remote_cache:
            self.bazel_remote_cache_arg = f"--remote_cache={self.bazel_remote_cache}"

        self.logs_path = logs_path
        if isinstance(self.logs_path, str):
            self.logs_path = Path(self.logs_path)

        self.env_args = env_args
        self.prefix = prefix

        # Remove old logs in any
        old_logs = glob.glob(f"{self.logs_path}/*")
        for f in old_logs:
            os.remove(f)

        self.logs_path.mkdir(parents=True, exist_ok=True)

        self.websight_port = int(websight_port)

        self.simulation_processes = []
        self.watchdog = None

    def __del__(self):
        self.stop()

    def is_running(self) -> Tuple[bool, int]:
        """Check if all environments are still running.

        Returns
        -------
        Tuple[bool, int]
            First value indicates whether everything is still running.
            Second value indicates the environments id of the batch that failed. If multiple
            environments failed since the last check, only gives back the first one.
        """
        for i, p in enumerate(self.simulation_processes):
            poll = p.poll()
            if poll is not None:
                return False, i

        return True, -1

    def start(self):
        """Start all environments"""
        for i in range(self.batch_size):
            self._start_simulation(i)

    def stop(self):
        """Stop all environments"""
        for process in self.simulation_processes:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            except ProcessLookupError:
                # Process is already dead
                pass

    def _start_simulation(self, idx):
        cmd = self._get_start_simulation_command(idx)
        logging.info(cmd)
        process = subprocess.Popen(
            cmd,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(self.logs_path / f"{idx}.log", "w"),
            stderr=subprocess.STDOUT)

        self.simulation_processes.append(process)

    @abstractmethod
    def _get_start_simulation_command(self, idx) -> str:
        """Command to start a single instance of the simulation."""
        pass

    def get_recent_logs(self, idx, lines=100):
        """Get the last n log lines for the specified environment.

        Parameters
        ----------
        idx
            Id of the environment inside the batch.
        lines
            How many lines of the log to retrieve (from the end).

        Returns
        -------
        List
            The last n log lines.
        """
        with open(self.logs_path / f"{idx}.log", "r") as fin:
            return deque(fin, lines)


class TcpRemoteGymRunner(GymRunner):
    """A batch runner instantiating containing n env instances.

    Starts environments locally and communicates via TCP.

    Parameters
    ----------
    env
        Environment id that corresponds to an Isaac application name.
    hosts
        List of hosts where the environment should be spawned. Spawns exactly 1 instance
        for each host, i.e. number of hosts is equal to the batch size.
    websight_port: int
        Port of the Isaac's websight application.
    bazel_output_dir
        Path defining where to write/read bazel outputs.
    bazel_remote_cache
        Address of the remote cache.
    port
        Starting port for tcp communication.
    """

    def __init__(
            self,
            env,
            hosts: List[str],
            target_device: str = "x86_64",
            port: int = 55000,
            websight_port: int = 3000,
            bazel_output_dir: Union[str, Path] = Path("/root/.cache/bazel/_bazel_root"),
            bazel_remote_cache: str = "http://bazel-remote.argmax.ai",
            logs_path: Union[str, Path] = Path("/tmp/argmax_gym_logs"),
            env_args: str = "",
            prefix: str = "vwml"
    ):
        super().__init__(
            env,
            batch_size=len(hosts),
            websight_port=websight_port,
            bazel_output_dir=bazel_output_dir,
            bazel_remote_cache=bazel_remote_cache,
            logs_path=logs_path,
            env_args=env_args,
            prefix=prefix)

        self.remote_user = "root"
        self.hosts = hosts
        self.target_device = target_device
        self.port = int(port)

    def _get_start_simulation_command(self, idx) -> str:
        return (f'ssh -t -t {self.remote_user}@{self.hosts[idx]} '
                f'"cd {deploy_path}/{self.env}-pkg/ && '
                f'./run ./apps/{self.prefix}/argmax_gym/envs/{self.env}.py '
                f'--port {self.port + idx} '
                f'--websight-port {self.websight_port} '
                f'{self.env_args}"')

    def is_running(self) -> Tuple[bool, int]:
        """Check if all environments are still running.

        Returns
        -------
        Tuple[bool, int]
            First value indicates whether everything is still running.
            Second value indicates the environments id of the batch that failed. If multiple
            environments failed since the last check, only gives back the first one.
        """
        for i, p in enumerate(self.simulation_processes):
            poll = p.poll()
            if poll is not None:
                return False, i

        for i in range(len(self.simulation_processes)):
            if not self._check_remote_pid(i):
                return False, i

        return True, -1

    def _check_remote_pid(self, i):
        command = [
            "ssh",
            f"{self.remote_user}@{self.hosts[i]}",
            f'pgrep -u {self.remote_user} -f "/bin/bash ./run ./apps/{self.prefix}/argmax_gym/envs/{self.env}.py"',
        ]
        try:
            subprocess.check_output(command)
            # logging.info(output)
            # pid = int(output.split(b"\n")[0])
            return True
        except subprocess.CalledProcessError:
            logging.error("Cannot find remote gym process, terminate")

        return False

    def stop(self):
        """Stop all environments"""

        for idx, process in enumerate(self.simulation_processes):
            try:
                # get PID
                command = [
                    "ssh",
                    "-t",
                    "-t",
                    f"{self.remote_user}@{self.hosts[idx]}",
                    f'pgrep -u {self.remote_user} -f "apps/{self.prefix}/argmax_gym/envs/{self.env}.py"',
                ]
                output = subprocess.check_output(command)
                pid = [str(p) for p in output.split(b"\n")]

                # Stop remote process
                command = [
                    "ssh",
                    "-t",
                    "-t",
                    f"{self.remote_user}@{self.hosts[idx]}",
                    f'kill -INT {" ".join(pid)}',
                ]
                subprocess.call(command)
            except Exception as ex:
                logging.info(ex)

            # Stop ssh command
            os.killpg(os.getpgid(process.pid), signal.SIGINT)

    def start(self):
        for i in range(self.batch_size):
            self._start_simulation(i)

