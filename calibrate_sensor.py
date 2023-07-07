# %%
from sklearn.linear_model import LinearRegression
import numpy as np
import transformation_functions as tf
import json
import matplotlib.pyplot as plt

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

    add_on = [1500,1750]

    if sensor_id > 3:
        calibrated_values = np.array(calibrated_values) - np.mean(calibrated_values[add_on[0]:add_on[1]])

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
    save_parameters = False

    #path = 'Data\\test_01_31\\2023_07_05_19_04_12.json' # Messung, dreimal auf jeden Sensor gedrückt, begonnen bei Motor 0, dann chronologisch weiter
    path = ["Data\\calibration\\Motor_0_Sensor_0_2023_07_06_18_17_07.json",
            "Data\\calibration\\Motor_2_Sensor_1_2023_07_06_18_19_38.json",
            "Data\\calibration\\Motor_4_Sensor_2.json",
            "Data\\calibration\\Motor_6_Sensor_3_2023_07_06_18_14_00.json",
            'Data\\test_01_31\\2023_07_05_19_04_12.json',
            "Data\\calibration\\Motor_1_Sensor_5_2023_07_06_18_18_20.json",
            "Data\\calibration\\Motor_3_Sensor_6_nicht sicher.json",
            "Data\\calibration\\Motor_5_Sensor_7_2023_07_06_18_12_24.json",
            "Data\\calibration\\Motor_7_Sensor_8_2023_07_06_18_15_15.json"
           ]
    
    
    scale = []
    offset = []
    add_on = [1500,1750]
    #range = [[129,213]]
    j = 0
    k = 0

    for i in range(0,9):
        #data = tf.get_json(path[i])
        data = tf.get_json(path)

        sensor_data = data['observation']['analogs']
        #press_period = find_calibration_range(sensor_data[i]['force'])
        #print(press_period)
        #calibrated_values =  data['observation']['force_torques'][0]['fz'][press_period[0]:press_period[1]] + data['observation']['force_torques'][0]['fz'][add_on[0]:add_on[1]]
        #uncalibrated_values = sensor_data[i]['force'][press_period[0]:press_period[1]] + sensor_data[i]['force'][add_on[0]:add_on[1]]
        #time = data['time'][press_period[0]:press_period[1]] + data['time'][add_on[0]:add_on[1]]

        if i % 2 == 0:
            uncalibrated_values = sensor_data[k]['force']
            k += 1
        else:            
            uncalibrated_values = sensor_data[j]['dforce']
            print('dforce')
            #uncalibrated_values = filter_array(uncalibrated_values, t=0.8)
            #uncalibrated_values = tf.interpolate_1d(np.array(uncalibrated_values))
            j += 1

        calibrated_values =  data['observation']['force_torques'][0]['fz']
        time = data['time']

        iscale, ioffset = calibrate_sensor(calibrated_values, uncalibrated_values)

        scale.append(iscale)
        offset.append(ioffset)


        plt.plot(time, calibrated_values, label='Calibrated')
        plt.plot(time, uncalibrated_values, label='Uncalibrated')
        plt.legend()
        plt.title(f'Sensor {i} dforce')
        plt.show()

    # Save the calibration parameters to a json file
    calibration_parameters = {'scale': scale, 'offset': offset}
    if save_parameters:
        with open('calibration_parameters.json', 'w') as f:
            json.dump(calibration_parameters, f, indent=4)
    
    time = data['time'] # type: ignore
    for i in [0,1,2,3,5,6,7,8]:
        print(f'Sensor {i}: {scale[i]} * sensor_value + {offset[i]} = calibrated_value')
        calibratet_value = scale[i] * np.array(sensor_data[i]['force']) + offset[i] # type: ignore
        if i > 3:
            calibratet_value = np.array(calibratet_value) - np.mean(calibratet_value[add_on[0]:add_on[1]])
        plt.plot(time, calibratet_value, label=f'Sensor {i}')
    plt.legend()
    plt.show()

#%%
