# %%
import numpy as np
# split the observation vector into their matching parts


class Sensor(object):
    """The General Sensor Object"""

    def __init__(self):
        self._state_vector = []
        self._length = len(self._state_vector)

    def __call__(self, obs):
        if len(obs) == self._length:
            self._state_vector = obs
            self._assign_state_vector()
        else:
            self._length_mismatch()

    def _assign_state_vector(self):
        print('Empty Implementation!')

    def _length_mismatch(self):
        print(f'Length Mismatch ({self.__class__.__name__})')

    def _to_json(self, aslist=False):
        """return a json compat dict with the attributes within"""
        attribute_dict = {}

        for attribute in dir(self):
            if ('_' in attribute) == False:
                attribute_dict[attribute] = [] if aslist else getattr(
                    self, attribute)

        return attribute_dict

    def __repr__(self):
        """Print all attributes"""
        myreturnstr = f'{self.__class__.__name__}:\n'
        for attribute in dir(self):
            if ('_' in attribute) == False:
                myreturnstr = f'{myreturnstr}\n{attribute}: {getattr(self, attribute)}'
        myreturnstr = f'{myreturnstr}\n'
        return myreturnstr


class MotorSensor(Sensor):
    """The Motor class"""

    def __init__(self):
        super().__init__()

        self.power = 0
        self.vel = 0
        self.torque = 0
        self._state_vector = [0, 0, 0]
        self._length = len(self._state_vector)

    def _assign_state_vector(self):
        self.power = self._state_vector[0]
        self.vel = self._state_vector[1]
        self.torque = self._state_vector[2]


class AnalogSensor(Sensor):
    """The Analog Force Sensor with F , dF"""

    def __init__(self):
        super().__init__()
        self.force = 0
        self.dforce = 0
        self._state_vector = [0, 0]
        self._length = len(self._state_vector)

    def _assign_state_vector(self):
        self.force = self._state_vector[0]
        self.dforce = self._state_vector[1]


class ForceTorqueSensor(Sensor):
    """The Force Torque Sensor"""

    def __init__(self):
        super().__init__()
        self.fx = 0
        self.fy = 0
        self.fz = 0

        self.mx = 0
        self.my = 0
        self.mz = 0

        

        self._state_vector = [0, 0, 0, 0, 0, 0]
        self._length = len(self._state_vector)

    def _assign_state_vector(self):
        self.fx = self._state_vector[0]
        self.fy = self._state_vector[1]
        self.fz = self._state_vector[2]

        self.mx = self._state_vector[3]
        self.my = self._state_vector[4]
        self.mz = self._state_vector[5]




class RigidBody(Sensor):
    """The Rigid Body as an object from the OptiTrack System"""

    def __init__(self):
        """init the rigid body object"""
        super().__init__()

        # position
        self.x = 0
        self.y = 0
        self.z = 0

        # quaternion
        self.qw = 1
        self.qx = 0
        self.qy = 0
        self.qz = 0

        # velocity translation
        self.vx = 0
        self.vy = 0
        self.vz = 0

        # velocity angles
        self.valpha = 0
        self.vbeta = 0
        self.vgamma = 0

        self._update_state_vector()
        self._length = len(self._state_vector)

        # also init the tmat
        self._tmat = np.eye(4)

    def _update_state_vector(self):
        self._state_vector = [
            self.x,
            self.y,
            self.z,

            # quaternion
            self.qw,
            self.qx,
            self.qy,
            self.qz,

            # velocity translation
            self.vx,
            self.vy,
            self.vz,

            # velocity angles
            self.valpha,
            self.vbeta,
            self.vgamma,
        ]

    def _assign_state_vector(self):
        """assign the individual elements from the state vector"""
        self.x = self._state_vector[0]
        self.y = self._state_vector[1]
        self.z = self._state_vector[2]

        # MAYBE ITS THE OTHER WAY ROUND...?! -> Optitrack export is in qx, qy, qz, qw order!
        self.qw = self._state_vector[3]
        self.qx = self._state_vector[4]
        self.qy = self._state_vector[5]
        self.qz = self._state_vector[6]

        self.vx = self._state_vector[7]
        self.vy = self._state_vector[8]
        self.vz = self._state_vector[9]

        self.valpha = self._state_vector[10]
        self.vbeta = self._state_vector[11]
        self.vgamma = self._state_vector[12]

        # self._calculate_transformation_matrix()

    def _quat_to_rot(self):
        """add quat to rot to computation graph"""
        qw, qx, qy, qz = self.qw, self.qx, self.qy. self.qz

        matrix = np.zeros(3, 3)

        matrix[0, 0] = 1. - 2. * qy ** 2 - 2. * qz ** 2
        matrix[1, 1] = 1. - 2. * qx ** 2 - 2. * qz ** 2
        matrix[2, 2] = 1. - 2. * qx ** 2 - 2. * qy ** 2

        matrix[0, 1] = 2. * qx * qy - 2. * qz * qw
        matrix[1, 0] = 2. * qx * qy + 2. * qz * qw

        matrix[0, 2] = 2. * qx * qz + 2 * qy * qw
        matrix[2, 0] = 2. * qx * qz - 2 * qy * qw

        matrix[1, 2] = 2. * qy * qz - 2. * qx * qw
        matrix[2, 1] = 2. * qy * qz + 2. * qx * qw

        return matrix

    def _calculate_transformation_matrix(self):
        """calculate the current transformation matrix"""
        rotmat = self._quat_to_rot()
        vec = np.array([self.x, self.y, self.z])
        self._tmat[:3, :3] = rotmat
        self._tmat[:3, 3] = vec


class ObservationHandler(object):
    """Handle the Observation and split data to the Sensors"""

    def __init__(self, num_motors=8, num_analog_sens=9, num_ft_sensors=1, num_rigid_bodies=5
                 ):

        self.num_motors = num_motors
        self.motors = [
            MotorSensor() for _ in range(num_motors)
        ]

        self.num_analog_sens = num_analog_sens
        self.analog_sensors = [
            AnalogSensor() for _ in range(num_analog_sens)
        ]

        self.num_ft_sensors = num_ft_sensors
        self.ft_sensors = [
            ForceTorqueSensor() for _ in range(num_ft_sensors)
        ]

        self.num_rigid_bodies = num_rigid_bodies
        self.rigid_bodies = [
            RigidBody() for _ in range(num_rigid_bodies)
        ]

        self.get_state_vector_length()
        self.create_empty_dict()

    def get_state_vector_length(self):
        """get all the lengths of the states"""

        self.name_list = [
            'motors', 'analogs', 'force_torques', 'rigid_bodies'
        ]
        self.sensor_list = [
            self.motors, self.analog_sensors, self.ft_sensors, self.rigid_bodies
        ]

        overall_length = 0
        for (name, sensors) in zip(self.name_list, self.sensor_list):
            loc_length = 0
            for sensor in sensors:
                loc_length += sensor._length
            setattr(self, f'{name}_length', loc_length)
            overall_length += loc_length

        self._length = overall_length

    def __call__(self, obs):
        """update the observation handler"""
        if len(obs) == self._length:
            self.assign_observation(obs)
            self.append_to_output_dict()
            return True

        else:
            print(len(obs))
            print(self._length)
            print('Length mismatch!')
            return False

    def assign_observation(self, obs):
        """write the data to the sensors"""
        cur_index = 0

        # iterate trough the different sensor types
        for sensors in self.sensor_list:

            # iterate trough the sensors
            for sensor in sensors:
                cur_len = sensor._length
                # update the sensor values
                sensor(obs[cur_index:cur_index+cur_len])
                cur_index += cur_len

    def create_empty_dict(self):
        """create an empty output dictionary"""
        self.output_dict = {}
        for name, sensors in zip(self.name_list, self.sensor_list):
            self.output_dict[name] = [
                sensor._to_json(aslist=True) for sensor in sensors
            ]

    def append_to_output_dict(self):
        """write the current state to the output dictionary by appending the values"""
        for name, sensors in zip(self.name_list, self.sensor_list):

            for idx, sensor in enumerate(sensors):
                loc_dict = sensor._to_json()

                for key in loc_dict.keys():
                    self.output_dict[name][idx][key].append(loc_dict[key])


# %% some tests
if __name__ == '__main__':

    ftsens = ForceTorqueSensor()
    ftsens([1, 2, 3, 4, 5, 6])
    ftsens
    # %%
    ftsens._to_json()
    # %%
    obs = ObservationHandler()
    obs.get_state_vector_length()
    # %%
    obs.motors_length
    # %%

    a = [1, 2, 3, 4, 5, 6]
    a[0:4]
    # %%
    a[4:6]
    # %%

    # %%
    obs.output_dict
    # %%
    obs(np.zeros(98).tolist())
    # %%
    obs.output_dict
    # %%
