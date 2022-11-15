from model import MujocoFingerModel

model = MujocoFingerModel("tendon_finger.template.xml", "generated_parameters.yaml")
model.run()
