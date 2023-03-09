import numpy as np
from scipy.fft import fft

def period(data, frame_time) -> float:

    # Perform the FFT on the data
    fft_data = fft(data)

    # Find the power spectrum of the FFT data
    power_spectrum = np.abs(fft_data)**2

    # Find the frequency corresponding to the maximum power in the spectrum
    max_power_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
    frequency = max_power_idx / len(data)

    # Calculate the period of the data in seconds
    period_frames = 1 / frequency
    period_time = period_frames * frame_time

    return period_time


