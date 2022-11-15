import numpy as np
import mujoco
import mujoco_viewer
import pystache
import yaml
import math
import collections
import pytorch_kinematics as pk
import dm_control
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import base


class FingerModel:
    def __init__(self, template_file, parameters_file):
        self.template_file = template_file
        self.parameters_file = parameters_file
        self.render_template()

    def render_template(self, override=None):
        with open(self.parameters_file, "r") as stream:
            try:
                parameters = yaml.safe_load(stream)
                if override is not None:
                    parameters.update(override)
            except yaml.YAMLError as exc:
                print(exc)
                exit(0)

        with open(self.template_file, "r") as stream:
            self.model = pystache.render(stream.read(), parameters)


class MujocoFingerModel(FingerModel):
    def __init__(self, template_file, parameters_file):
        self.template_file = template_file
        self.parameters_file = parameters_file
        self.render_template(override={"complete": True})

    def run(self):
        mjc_model = mujoco.MjModel.from_xml_string(self.model)
        data = mujoco.MjData(mjc_model)

        # create the viewer object
        viewer = mujoco_viewer.MujocoViewer(mjc_model, data)

        # simulate and render
        for _ in range(100000):
            mujoco.mj_step(mjc_model, data)
            viewer.render()

        # close
        viewer.close()


class Physics(dm_control.mujoco.Physics):
    @property
    def bodies(self):
        return [
            "tracker-dau",
            "pip-tracker",
            # "tracker-dau-mcp",
        ]

    def orientations(self):
        return self.named.data.xmat[self.bodies]

    def positions(self):
        return self.named.data.xpos[self.bodies]


class TendonTask(base.Task):
    def __init__(self, random=None):
        self._sparse = False
        super().__init__(random=random)

    def initialize_episode(self, physics):
        super().initialize_episode(physics)

    def get_observation(self, physics):
        def matrix_from_observation(orientation, position):
            return np.vstack(
                [
                    np.hstack(
                        [
                            orientation.reshape((3, 3)),
                            np.transpose([position]),
                        ],
                    ),
                    np.array([0, 0, 0, 1]),
                ]
            )

        orientation = np.array(
            [
                matrix_from_observation(physics.orientations()[0], physics.positions()[0]),
                matrix_from_observation(physics.orientations()[1], physics.positions()[1]),
            ]
        )

        obs = collections.OrderedDict()
        obs["angle"] = physics.position()
        obs["orientations"] = orientation
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        return 0.0


class DMSuiteFingerModel(MujocoFingerModel):
    def build_env(self):
        physics = Physics.from_xml_string(self.model, common.ASSETS)
        task = TendonTask()

        return control.Environment(physics, task, time_limit=10)


class PyTorchFingerModel(FingerModel):
    def __init__(self, template_file, parameters_file):
        super().__init__(template_file, parameters_file)
        self.build_chain()

    def build_chain(self):
        # load robot description from URDF/MJCF and specify end effector link
        self.parsed_model = pk.mjcf_parser.from_xml_string(self.model)
        self.parameters = pk.parameters_from_mjcf(self.parsed_model)

    def forward_kinematic(self, joint_states):
        bodies = [
            "tracker-dau",
            "pip-tracker",
            # "tracker-dau-mcp",
        ]

        matrices = []
        for b in bodies:
            chain = pk.build_serial_chain_from_mjcf(
                None, b, self.parameters, model=self.parsed_model
            )
            m = chain.forward_kinematics(joint_states, end_only=True)
            m = m.get_matrix().reshape((joint_states.shape[0], -1))
            matrices.append(m)
        return matrices
