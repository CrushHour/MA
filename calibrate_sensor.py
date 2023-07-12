# %%
from sklearn.linear_model import LinearRegression
import numpy as np
import transformation_functions as tf
import json
import matplotlib.pyplot as plt

def arrange_sensor_data(data):
    """Arrange the sensor data into a list of lists, where each sublist contains the force measurements for one sensor"""
    sorted_data = []
    sensor_data = data['observation']['analogs']
    for i in range(4):
        sorted_data.append(sensor_data[i]['force'])
        sorted_data.append(sensor_data[i]['dforce'])

    return sorted_data

def find_calibration_range(force_measurements):
    """
    data: a list of tuples where each tuple contains two elements: a timestamp and a force measurement
    multiplier: the number of standard deviations to consider as the threshold for a significant change in force measurement
    """
    multiplier = 2

    threshold = np.std(force_measurements) * multiplier

    press_periods = []
    prev_value = force_measurements[0]  # initial force measurement
    press_start = None

    for i, value in enumerate(force_measurements):
        if abs(value - prev_value) > threshold:
            if press_start is None:  # start of a new press
                press_start = i
            else:  # end of the current press
                press_periods.append((press_start, i))
                press_start = None  # reset press_start for the next press

            prev_value = value  # update prev_value with the current force measurement

    # If the last press hasn't ended yet, we add it to the list as well
    if press_start is not None:
        press_periods.append((press_start, press_start + 1))

    return [press_periods[0][0],press_periods[-1][-1]]   

def calibrate_sensor(calibrated_values, uncalibrated_values):
    # Make sure that the lists are numpy arrays, as the linear regression model expects this format
    calibrated_values = np.array(calibrated_values).reshape((-1, 1))
    uncalibrated_values = np.array(uncalibrated_values).reshape((-1, 1))

    # Create a new linear regression model
    model = LinearRegression()

    # Fit the model to the uncalibrated values
    model.fit(uncalibrated_values, calibrated_values)

    # The slope (scale) and intercept (offset) from the linear regression model give the calibration parameters
    scale = model.coef_[0]
    offset = model.intercept_

    return scale[0], offset[0] # type: ignore

def apply_calibration(uncalibrated_values, sensor_id, calibration_file='calibration_parameters_long.json'):
    # Load the calibration parameters from the JSON file
    with open(calibration_file, 'r') as f:
        calibration_params = json.load(f)

    # Get the scale and offset for the given sensor ID
    scale = calibration_params['scale'][sensor_id]
    offset = calibration_params['offset'][sensor_id]

    # Make sure that the list is a numpy array for vectorized operations
    uncalibrated_values = np.array(uncalibrated_values)

    # Apply the scale and offset to the uncalibrated values
    calibrated_values = scale * uncalibrated_values + offset
    
    #calibration from FT-Sensor to Newton [N] (for documentation look in MA)
    calibrated_values = calibrated_values / 8.998278583527435 #[Units/N]
    
    return calibrated_values

def filter_array(arr, t=0.8, decimal_places=7):
    """Filter the array using the given threshold t and round the filtered values to the specified number of decimal places"""
    filtered_arr = []
    loc_value = arr[0]
    filtered_arr.append(round(loc_value, decimal_places))
    for value in arr[1:]:
        loc_value = t * loc_value + (1 - t) * value
        filtered_arr.append(round(loc_value, decimal_places))
    for i in range(len(arr)-1, 0, -1):
        loc_value = t * loc_value + (1 - t) * filtered_arr[i]
        filtered_arr[i] = round(loc_value, decimal_places)
    return filtered_arr


if __name__ == '__main__':
    save_parameters = True
    use_filter = False

    #path = 'Data\\test_01_31\\2023_07_05_19_04_12.json' # Messung, dreimal auf jeden Sensor gedrückt, begonnen bei Motor 0, dann chronologisch weiter
    path = ["Data\\calibration\\old\\Motor_0_Sensor_0_2023_07_06_18_17_07.json",
            "Data\\calibration\\old\\Motor_1_Sensor_5_2023_07_06_18_18_20.json",
            "Data\\calibration\\old\\Motor_2_Sensor_1_2023_07_06_18_19_38.json",
            "Data\\calibration\\old\\Motor_3_2023_07_06_18_21_03.json",
            "Data\\calibration\\old\\Motor_4_Sensor_2.json",
            "Data\\calibration\\old\\Motor_5_Sensor_7_2023_07_06_18_12_24.json",
            "Data\\calibration\\old\\Motor_6_Sensor_3_2023_07_06_18_14_00.json",
            "Data\\calibration\\old\\Motor_7_Sensor_8_2023_07_06_18_15_15.json"
           ]
        
    scale = []
    offset = []
    add_on = [1500,1750]
    #range = [[129,213]]

    for i in range(8):
        
        data = tf.get_json(path[i])

        sensor_data = arrange_sensor_data(data)
        
        uncalibrated_values = np.array(sensor_data[i])

        if use_filter:
            uncalibrated_values = filter_array(uncalibrated_values, t=0.8)
            uncalibrated_values = tf.interpolate_1d(np.array(uncalibrated_values))

        force_torque =  data['observation']['force_torques'][0]['fz']
        time = data['time']

        iscale, ioffset = calibrate_sensor(force_torque, uncalibrated_values)

        calibratet_value = iscale * uncalibrated_values + ioffset

        scale.append(iscale)
        offset.append(ioffset) 

        plt.plot(time, force_torque, label='Force-Torque Sensor')
        plt.plot(time, uncalibrated_values, label='Uncalibrated')
        plt.plot(time, calibratet_value, label='Calibrated')
        plt.legend()
        plt.title(f'Sensor {i}')
        plt.show()

    # Save the calibration parameters to a json file
    calibration_parameters = {'scale': scale, 'offset': offset}
    if save_parameters:
        with open('calibration_parameters_long.json', 'w') as f:
            json.dump(calibration_parameters, f, indent=4)
    
    #show calibrated values
    time = data['time'] # type: ignore
    for i in range(8):
        print(f'Sensor {i}: {scale[i]} * sensor_value + {offset[i]} = calibrated_value')
        calibratet_value = apply_calibration(sensor_data[i], i,'calibration_parameters_long.json') # type: ignore
        plt.plot(time, calibratet_value, label=f'Motor Analog {i}')
    plt.legend()
    plt.show()

#%%
