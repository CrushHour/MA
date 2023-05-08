import numpy as np
import mujoco
import mujoco_viewer
import pystache
import yaml
import math
import collections
import pytorch_kinematics as pk

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
                parameters.update(
                    {
                        "mesh_dir": "meshes_secondhand",
                    }
                )
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

        self.mjc_model = mujoco.MjModel.from_xml_string(self.model) # type: ignore
        self.data = mujoco.MjData(self.mjc_model) # type: ignore

        with open("./finger_control.xml", "w") as text_file:
            text_file.write(self.model)

        self.control_model = mujoco.MjModel.from_xml_string(self.model) # type: ignore
        self.control_data = mujoco.MjData(self.control_model) # type: ignore
    
    def run(self):
        return

if __name__ == "__main__":
    model = MujocoFingerModel("./mujocu/my_tendom_finger_template_simple.xml", "./mujocu/generated_parameters.yaml")
    model.run()

