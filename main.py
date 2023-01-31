# %% Init
# packages
from argmax_gym import TcpRemoteGym
import numpy as np
import json
from datasplit import ObservationHandler
from definitions import action_off, action_amp, action_phase
import cv2 as cv
import time
from datetime import datetime
from rich.progress import track
from rich.status import Status

# def init
"""
ZUWEISUNG
Strecker 1 zeigerfinger: 4
Strecker 2 zeigerfinger: 6

Beuger 1 zeigerfinger: 3
Beuger 2 zeigerfinger: 2

Strecker 1 Daumen: 7
Strecker 2 Daumen: 5

Daumen Abspreitzer: 0
Daumen Beuger: 1

Id
ZF PP : 1007
ZF DP : 1008

DAUMEN DP : 1009
DAUMEN MC : 1010

FORCETORQUE: 1011
"""

safe_path = '/home/robotlab/Documents/GitHub/MA_Schote/MA/Data/test_01_31'

json_out = {
    'time': [],
    'action': [],
}

obs_handler = ObservationHandler(num_rigid_bodies=6)

print('Conntecting to TcpRemote.')
env = TcpRemoteGym(
    "phoenix-v0",
    hosts=["141.39.191.228"],
    target_device="x86_64",
    prefix="phoenix"
)

print('init')
obs, image = env.reset(record=False)
print('reset')

start_time = datetime.now()
t_max = 20 * 1000 #ms
t = 0
dt = 1000 / 25 #ms

# def motoren

# def videoaufzeichnung
cap = cv.VideoCapture(0)
video_list = []

if not cap.isOpened():
    print("Cannot open camera")
    exit()

def videocapture():
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    # Our operations on the frame come here
    
    return frame

def millis(init_time, now):
    '''Vergangene Zeit seit init_time in ms'''
    #now = datetime.now() # Format: datetime.datetime(2022, 12, 4, 14, 20, 12, 623357)
    dt = now - init_time # Format: datetime.timedelta(seconds=1, microseconds=1470)
    dt_in_ms = dt.seconds * 1000 + dt.microseconds / 1000
    return dt_in_ms

#%% main
if __name__ == '__main__':
    print('starting')
    try:
        
        while t <= t_max:
            t = millis(start_time, datetime.now())
        
            frame = videocapture()
            video_list.append(frame)

            #Bewegungsbefehl
            cos_in = 0.001 * t - action_phase
            
            action = 2 * action_off + 3 *action_amp * np.cos(cos_in)
            obs, reward, done, image = env.step(action)
            
            #Bewegung Speichern
            append_obs = obs[0].tolist()

            if obs_handler(append_obs):
                # add to json_out:
                json_out['time'].append(t)
                json_out['action'].append(action.tolist())
        

        # Close Test
        print('Finished Testrun.')
        # dump json
        json_out['observation'] = obs_handler.output_dict
        tt = time.localtime()
        current_time = time.strftime("%Y_%m_%d_%H_%M_%S", tt)

        with open(f'{safe_path}/{current_time}.json', 'w') as f:
            json.dump(json_out, f, indent=2)

        #save video
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        print(frame_width,',', frame_height)
        out = cv.VideoWriter(f'{safe_path}/Video/{current_time}.avi',cv.VideoWriter_fourcc(*'DIVX'), frameSize=(frame_width,frame_height), fps=25)
        for i in range(len(video_list)):
            frame = video_list[i]
            out.write(frame)
        # release camera
        cap.release()
        cv.destroyAllWindows()
        f.close()
        print('Data saved and closed!')



    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        env.close()
        with Status("Stopping environment", spinner="bouncingBar") as status:
            status.update(
                status="Stopping environment",
                spinner_style="white",
            )
            while env.gym_runner.is_running()[0]:
                time.sleep(0.1)
        print("[green]Environment stopped.")    
# %%