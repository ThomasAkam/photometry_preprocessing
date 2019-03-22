# Function for opening pyPhotometry data files in Python.
# Copyright (c) Thomas Akam 2018.  Licenced under the GNU General Public License v3.

import json
import numpy as np
from scipy.signal import butter, filtfilt

def import_ppd(file_path, low_pass=20, high_pass=0.01):
    '''Function to import pyPhotometry binary data files into Python. The high_pass 
    and low_pass arguments determine the frequency in Hz of highpass and lowpass 
    filtering applied to the filtered analog signals. To disable highpass or lowpass
    filtering set the respective argument to None.  Returns a dictionary with the 
    following items:
        'subject_ID'    - Subject ID
        'date_time'     - Recording start date and time (ISO 8601 format string)
        'mode'          - Acquisition mode
        'sampling_rate' - Sampling rate (Hz)
        'LED_current'   - Current for LEDs 1 and 2 (mA)
        'version'       - Version number of pyPhotometry
        'analog_1'      - Raw analog signal 1 (volts)
        'analog_2'      - Raw analog signal 2 (volts)
        'analog_1_filt' - Filtered analog signal 1 (volts)
        'analog_2_filt' - Filtered analog signal 2 (volts)
        'digital_1'     - Digital signal 1
        'digital_2'     - Digital signal 2
        'pulse_times_1' - Times of rising edges on digital input 1 (ms).
        'pulse_times_2' - Times of rising edges on digital input 2 (ms).
        'time'          - Time of each sample relative to start of recording (ms)
    '''
    with open(file_path, 'rb') as f:
        header_size = int.from_bytes(f.read(2), 'little')
        data_header = f.read(header_size)
        data = np.frombuffer(f.read(), dtype=np.dtype('<u2'))
    # Extract header information
    header_dict = json.loads(data_header)
    volts_per_division = header_dict['volts_per_division']
    sampling_rate = header_dict['sampling_rate']
    # Extract signals.
    analog  = data >> 1                     # Analog signal is most significant 15 bits.
    digital = ((data & 1) == 1).astype(int) # Digital signal is least significant bit.
    # Alternating samples are signals 1 and 2.
    analog_1 = analog[ ::2] * volts_per_division[0]
    analog_2 = analog[1::2] * volts_per_division[1]
    digital_1 = digital[ ::2]
    digital_2 = digital[1::2]
    time = np.arange(analog_1.shape[0])*1000/sampling_rate # Time relative to start of recording (ms).
    # Filter signals with specified high and low pass frequencies (Hz).
    if low_pass and high_pass:
        b, a = butter(2, np.array([high_pass, low_pass])/(0.5*sampling_rate), 'bandpass')
    elif low_pass:
        b, a = butter(2, low_pass/(0.5*sampling_rate), 'low')
    elif high_pass:
        b, a = butter(2, high_pass/(0.5*sampling_rate), 'high')
    if low_pass or high_pass:
        analog_1_filt = filtfilt(b, a, analog_1)
        analog_2_filt = filtfilt(b, a, analog_2)
    else:
        analog_1_filt = analog_2_filt = None
    # Extract rising edges for digital inputs.
    pulse_times_1 = (1+np.where(np.diff(digital_1) == 1)[0])*1000/sampling_rate
    pulse_times_2 = (1+np.where(np.diff(digital_2) == 1)[0])*1000/sampling_rate
    # Return signals + header information as a dictionary.
    data_dict = {'analog_1'      : analog_1,
                 'analog_2'      : analog_2,
                 'analog_1_filt' : analog_1_filt,
                 'analog_2_filt' : analog_2_filt,
                 'digital_1'     : digital_1,
                 'digital_2'     : digital_2,
                 'pulse_times_1' : pulse_times_1,
                 'pulse_times_2' : pulse_times_2,
                 'time'          : time}
    data_dict.update(header_dict)
    return data_dict