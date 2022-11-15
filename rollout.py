# %%
import numpy as np
from rich.progress import track
from model import DMSuiteFingerModel
from dm_control.suite.wrappers import pixels
from PIL import Image
from numpy.linalg import inv

save_images = False

model = DMSuiteFingerModel("tendon_finger.template.xml", "generated_parameters.yaml")
env = model.build_env()

if save_images:
    env = pixels.Wrapper(
        env,
        pixels_only=False,
        render_kwargs={
            "width": 480,
            "height": 480,
            "camera_id": 0,
        },
    )

T = 1000
N = 1
n_obs = 32
n_angles = 3
observations = np.zeros((T, N, n_obs))
angles = np.zeros((T, N, n_angles))


for e in track(range(N), description="Simulating..."):
    ret = env.reset()
    transform = [inv(m) for m in ret.observation["orientations"]]

    observations[0, e] = np.concatenate(
        [np.reshape(m @ t, (16,)) for m, t in zip(ret.observation["orientations"], transform)]
    )
    angles[0, e] = ret.observation["angle"]
    for i in range(1, T):
        action = (
            (np.array([np.sin(i / 100.0), np.sin(i / 100.0), np.sin(i / 100.0)]) / 2.0 + 0.5)
            * np.pi
            / 4
        )
        ret = env.step(action)
        observations[i, e] = np.concatenate(
            [np.reshape(m @ t, (16,)) for m, t in zip(ret.observation["orientations"], transform)]
        )
        angles[i, e] = ret.observation["angle"]

        if save_images:
            im = Image.fromarray(ret.observation["pixels"])
            im.save(f"images/{e:04d}_{i:04d}.png")

# %%

np.save("rollout.npy", observations)
np.save("rollout_angles.npy", angles)


# %%