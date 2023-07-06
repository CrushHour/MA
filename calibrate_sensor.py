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

def apply_calibration(uncalibrated_values, sensor_id, calibration_file='calibration_parameters.json'):
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


if __name__ == '__main__':
    path = 'Data\\test_01_31\\2023_07_05_19_04_12.json' # Messung, dreimal auf jeden Sensor gedrückt, begonnen bei Motor 0, dann chronologisch weiter
    #path = 'Data\\test_01_31\\2023_07_05_19_02_38.json' 
    data = tf.get_json(path)
    sensor_data = data['observation']['analogs']
    
    scale = []
    offset = []
    add_on = [1500,1750]
    #range = [[129,213]]
    for i in range(9):
        press_period = find_calibration_range(sensor_data[i]['force'])
        calibrated_values =  data['observation']['force_torques'][0]['fz'][press_period[0]:press_period[1]] + data['observation']['force_torques'][0]['fz'][add_on[0]:add_on[1]]
        uncalibrated_values = sensor_data[i]['force'][press_period[0]:press_period[1]] + sensor_data[i]['force'][add_on[0]:add_on[1]]

        time = data['time'][press_period[0]:press_period[1]] + data['time'][add_on[0]:add_on[1]]

        iscale, ioffset = calibrate_sensor(calibrated_values, uncalibrated_values)

        scale.append(iscale)
        offset.append(ioffset)

        print(press_period)

        plt.plot(time, calibrated_values, label='Calibrated')
        plt.plot(time, uncalibrated_values, label='Uncalibrated')
        plt.legend()
        plt.title(f'Sensor {i}')
        plt.show()

    # Save the calibration parameters to a json file
    calibration_parameters = {'scale': scale, 'offset': offset}
    with open('calibration_parameters.json', 'w') as f:
        json.dump(calibration_parameters, f, indent=4)
    
    time = data['time']
    for i in [0,1,2,3,5,6,7,8]:
        print(f'Sensor {i}: {scale[i]} * sensor_value + {offset[i]} = calibrated_value')
        calibratet_value = scale[i] * np.array(sensor_data[i]['force']) + offset[i]
        if i > 3:
            calibratet_value = np.array(calibratet_value) - np.mean(calibratet_value[add_on[0]:add_on[1]])
        plt.plot(time, calibratet_value, label=f'Sensor {i}')
    plt.legend()
    plt.show()