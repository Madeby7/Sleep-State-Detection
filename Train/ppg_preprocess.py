import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import biosppy
from scipy.fftpack import fft
import pandas as pd
import biosppy
from biosppy.signals import ppg
from biosppy.signals.tools import filter_signal
from scipy.signal import find_peaks, butter, filtfilt
import neurokit2 as nk
import plotly.graph_objs as go
from scipy.signal import find_peaks

SAMPLING_RATE = 25
LOWCUT = 0.2
HIGHCUT = 5.0
SYNT_HEART_RATE = 70
SYNT_DURATION = 20
N_FFT=512
NOISE_LEVEL = 0.3

# CLASSIFICATION WINDOW AND STEP
SEGMENT_LENGTH_S = 30  # 20 seconds per segment
STEP_SIZE_S = 15        # 5 seconds step size

def generate_synthetic_ppg(duration, sampling_rate, heart_rate):
    ppg_signal = nk.ppg_simulate(duration, sampling_rate, heart_rate, random_state=42)
    return ppg_signal

def add_noise(signal, noise_level):
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

def process_ppg_with_biosppy(signal, sampling_rate):
    ppg_obj = biosppy.signals.ppg.ppg(signal, sampling_rate=sampling_rate, show=False)
    heart_rate = np.mean(ppg_obj['heart_rate'])
    return ppg_obj, heart_rate

def process_ppg(signal, sampling_rate):
    signals, info = nk.ppg_process(signal, sampling_rate=sampling_rate)
    return signals, info

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
def butter_highpass(cutoff, fs, order=5):
    """
    Create a highpass filter.
    
    Args:
    cutoff (float): The cutoff frequency for the highpass filter.
    fs (float): The sampling frequency of the signal.
    order (int): The order of the filter. Higher values mean a sharper cutoff.
    
    Returns:
    b, a: Filter coefficients.
    """
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = butter(order, high, btype='highpass')
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    """
    Apply a highpass filter to the data.
    
    Args:
    data (array-like): The input signal.
    cutoff (float): The cutoff frequency for the highpass filter.
    fs (float): The sampling frequency of the signal.
    order (int): The order of the filter. Higher values mean a sharper cutoff.
    
    Returns:
    y: The filtered signal.
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def insert_discontinuities(df, time_col, threshold):
    df = df.copy()
    # Calculate the time differences
    time_diff = df[time_col].diff().abs()
    # Identify where the time difference exceeds the threshold
    discontinuity_indices = time_diff[time_diff > threshold].index

    # Insert None values
    for index in discontinuity_indices:
        df = pd.concat([df.iloc[:index], pd.DataFrame({time_col: [None], 'ledGreen_filtered': [None], 'ledGreen': [None]}), df.iloc[index:]]).reset_index(drop=True)
    
    return df

def calculate_snr(signal, sampling_rate, signal_band, noise_band, window='hann', n_fft=None):
    
    # USE PEAKS. + OR - OF THE PEAKS
    N = len(signal)
    if n_fft is None:
        n_fft = N

    # Apply window function
    if window == 'hann':
        window_func = np.hanning(N)
    elif window == 'hamming':
        window_func = np.hamming(N)
    elif window == 'blackman':
        window_func = np.blackman(N)
    else:
        window_func = np.ones(N)
    
    # Normalize the windowed signal to maintain power
    windowed_signal = signal * window_func
    windowed_signal = windowed_signal / np.sqrt(np.mean(window_func**2))

    # FFT and Power Spectral Density (PSD)
    fft_result = np.fft.fft(windowed_signal, n=n_fft)
    fft_result = fft_result[:n_fft // 2]  # Consider only positive frequencies
    psd = (np.abs(fft_result)**2) / (N * np.sum(window_func**2))

    # Calculate power in the signal and noise bands
    signal_power = np.sum(psd[signal_band])
    noise_power = np.sum(psd[noise_band])

    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr)
    return snr_db

def calculate_snr_around_peaks(signal, sampling_rate, signal_band, noise_band, window='hann', n_fft=None, peak_window=50):
    """
    Calculate SNR for PPG signal focusing on regions around the detected peaks.

    Parameters
    ----------
    signal : array-like
        Raw PPG signal.
    sampling_rate : float
        Sampling frequency of the PPG signal.
    signal_band : array-like
        The indices of the signal frequencies for SNR calculation.
    noise_band : array-like
        The indices of the noise frequencies for SNR calculation.
    window : str
        Window function to apply (default is 'hann').
    n_fft : int
        Number of FFT points (default is length of the signal).
    peak_window : int
        The number of samples to include before and after each peak for the SNR calculation.

    Returns
    -------
    snr_db : float
        The signal-to-noise ratio in decibels (dB).
        Returns 0 if no peaks are detected.
    """
    
    # Step 1: Detect peaks using NeuroKit2
    #peaks, info = nk.ppg_peaks(signal, sampling_rate=sampling_rate, method="bishop", show=False)
    #peak_indices = np.where(peaks == 1)[0]  # Get indices of detected peaks

    # Step 1: Detect peaks using SciPy
    peak_indices, _ = find_peaks(signal, distance=sampling_rate//2, prominence=0.5)  # Customize params as needed

    if len(peak_indices) == 0:
        # No peaks detected, return SNR of 0
        return 0

    # Step 2: Extract windows around each detected peak
    signal_windows = []
    for peak in peak_indices:
        start_idx = max(0, peak - peak_window)  # Start `peak_window` samples before the peak
        end_idx = min(len(signal), peak + peak_window)  # End `peak_window` samples after the peak
        signal_windows.extend(signal[start_idx:end_idx])  # Extract the signal around the peak

    signal_windows = np.array(signal_windows)

    # Step 3: Apply the window function
    N = len(signal_windows)
    if n_fft is None:
        n_fft = N

    if window == 'hann':
        window_func = np.hanning(N)
    elif window == 'hamming':
        window_func = np.hamming(N)
    elif window == 'blackman':
        window_func = np.blackman(N)
    else:
        window_func = np.ones(N)

    # Normalize the windowed signal to maintain power
    windowed_signal = signal_windows * window_func
    windowed_signal = windowed_signal / np.sqrt(np.mean(window_func**2))

    # Step 4: FFT and Power Spectral Density (PSD)
    fft_result = np.fft.fft(windowed_signal, n=n_fft)
    fft_result = fft_result[:n_fft // 2]  # Consider only positive frequencies
    psd = (np.abs(fft_result)**2) / (N * np.sum(window_func**2))

    # Step 5: Calculate power in the signal and noise bands
    signal_power = np.sum(psd[signal_band])
    noise_power = np.sum(psd[noise_band])

    # Avoid division by zero in the case of no noise
    if noise_power == 0:
        return float('inf')  # Infinite SNR if no noise

    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr)

    return snr_db

def determine_snr_threshold(synthetic_signal, sampling_rate, noise_level, window='hann', n_fft=None):
    noisy_signal = add_noise(synthetic_signal, noise_level)
    if n_fft is None:
        n_fft = len(noisy_signal)
    
    frequencies = np.fft.fftfreq(n_fft, 1/sampling_rate)[:n_fft // 2]
    # Define signal and noise bands
    signal_band = (frequencies >= LOWCUT) & (frequencies <= HIGHCUT)  # Typical PPG signal range
    noise_band = (frequencies < LOWCUT) | (frequencies > HIGHCUT)     # Example noise band
    
    snr_db = calculate_snr(noisy_signal, sampling_rate, signal_band, noise_band, window=window, n_fft=n_fft)
    return snr_db

# Segment the actual PPG signal
def segment_signal(signal, segment_length):
    return [signal[i:i + segment_length] for i in range(0, len(signal), segment_length)]

# Evaluate actual PPG signal segments
def evaluate_ppg_signal(actual_signal, sampling_rate, snr_threshold, segment_length):
    segments = segment_signal(actual_signal, segment_length)
    segment_results = []
    for segment in segments:
        if len(segment) < segment_length:
            continue  # Skip segments that are too short
        snr = calculate_snr(segment, sampling_rate)
        quality = 'good' if snr >= snr_threshold else 'bad'
        segment_results.append((snr, quality))
    return segment_results



def get_good_bad_segments(df,snr_threshold,sampling_rate,n_fft=512, window_duration=20):
    actual_ppg = df['ledGreen_filtered'].values

    # Segment the actual PPG signals
    segment_length = window_duration * sampling_rate  # Segment length in samples (10 seconds)
    segments = [actual_ppg[i:i + segment_length] for i in range(0, len(actual_ppg), segment_length)]

    # Define signal and noise frequency bands
    frequencies = np.fft.fftfreq(n_fft, 1/sampling_rate)[:n_fft // 2]
    signal_band = (frequencies >= LOWCUT) & (frequencies <= HIGHCUT)  # PPG signal band
    noise_band = (frequencies < LOWCUT) | (frequencies > HIGHCUT)     # Noise band

    # Evaluate each segment and classify them based on SNR
    good_bad_segments = []
    for i, segment in enumerate(segments):
        if len(segment) < segment_length:
            continue  # Skip segments that are too short
        #print(len(segment))
        #segment_results = biosppy.signals.ppg.ppg(signal=segment, sampling_rate=25, show=False)
        #segment = segment_results['filtered']
        snr = calculate_snr(segment, sampling_rate, signal_band, noise_band,'hann',n_fft)
        #print(snr)
        quality = 'good' if snr >= snr_threshold else 'bad'
        #print(quality)
        start_idx = i * segment_length
        end_idx = start_idx + len(segment)
        good_bad_segments.append((quality, start_idx, end_idx))

# Function to segment signal into overlapping windows
def overlapping_windows(signal, segment_length, step_size):
    return [signal[i:i + segment_length] for i in range(0, len(signal) - segment_length + 1, step_size)]

# Function to aggregate scores from overlapping segments
def aggregate_classifications(classifications, signal_length, segment_length, step_size):
    classification = np.zeros(signal_length)
    counts = np.zeros(signal_length)

    for quality, start_idx, end_idx in classifications:
        value = 1 if quality == 'good' else -1
        classification[start_idx:end_idx] += value
        counts[start_idx:end_idx] += 1

    # Aggregate scores without normalizing
    return classification

def remove_segments_based_on_score(df_original, aggregated_score, threshold):
    # Identify the indices of bad segments based on the aggregate score
    bad_indices = np.where(aggregated_score < threshold)[0]
    
    # Create ranges of continuous bad segments
    bad_ranges = []
    if len(bad_indices) > 0:
        start_idx = bad_indices[0]
        for i in range(1, len(bad_indices)):
            if bad_indices[i] != bad_indices[i-1] + 1:
                bad_ranges.append((start_idx, bad_indices[i-1]))
                start_idx = bad_indices[i]
        bad_ranges.append((start_idx, bad_indices[-1]))

    # Remove the bad segments from the original DataFrame
    for start_idx, end_idx in bad_ranges:
        df_original = df_original.drop(index=range(start_idx, end_idx+1))

    return df_original.reset_index(drop=True)

def align_sensors_on_time(df, sensor_columns):
    """Aligns all sensor data by interpolating to common timestamps."""
    # Ensure 'unixTimes' is numeric
    df['unixTimes'] = pd.to_numeric(df['unixTimes'], errors='coerce')
    
    # Set Unix times as index
    df.set_index('unixTimes', inplace=True)
    
    # Interpolate missing values to align all sensor data
    #df[sensor_columns] = df[sensor_columns].interpolate(method='linear', axis=0)
    
    # Drop any remaining rows with NaN values after interpolation
    #df.dropna(subset=sensor_columns, inplace=True)
    
    return df.reset_index()

def plot_fft(ppg_signal, fs, window='hann', n_fft=None, title='FFT of PPG Signal'):
    """
    Plot the FFT of a PPG signal using Plotly.

    Parameters:
    - ppg_signal: np.array, the PPG signal data
    - fs: float, the sampling rate of the signal in Hz
    - window: str, the type of window function to apply ('hann', 'hamming', 'blackman', etc.)
    - n_fft: int, number of points for FFT computation (None uses the length of the signal)
    - title: str, the title of the plot
    """
    N = len(ppg_signal)
    if n_fft is None:
        n_fft = N

    # Apply window function
    if window == 'hann':
        window_func = np.hanning(N)
    elif window == 'hamming':
        window_func = np.hamming(N)
    elif window == 'blackman':
        window_func = np.blackman(N)
    else:
        window_func = np.ones(N)
    
    # Normalize the windowed signal
    windowed_signal = ppg_signal * window_func
    windowed_signal = windowed_signal / np.sqrt(np.mean(window_func**2))

    # Compute the FFT
    fft_result = np.fft.fft(windowed_signal, n=n_fft)
    fft_magnitude = np.abs(fft_result)[:n_fft // 2]  # Take the positive frequencies
    fft_real = np.real(fft_result)[:n_fft // 2]  # Take the positive frequencies
    fft_imag = np.imag(fft_result)[:n_fft // 2]  # Take the positive frequencies

    frequencies = np.fft.fftfreq(n_fft, 1/fs)[:n_fft // 2]

    # Exclude zero and negative frequencies
    positive_freqs = frequencies > 0
    fft_magnitude = fft_magnitude[positive_freqs]
    fft_real = fft_real[positive_freqs]
    fft_imag = fft_imag[positive_freqs]
    frequencies = frequencies[positive_freqs]
    
    normalized_magnitude = fft_magnitude / np.max(fft_magnitude)
    
    fig = go.Figure()

    # Plot the magnitude of the FFT
    fig.add_trace(go.Scatter(x=frequencies, y=fft_magnitude, mode='lines', name='FFT Magnitude'))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude',
        hovermode='x unified'
    )

    # Show the plot
    fig.show()
    
    # Plot the FFT magnitude spectrum using Plotly
    fig = go.Figure()

    # Plot the magnitude of the FFT
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=fft_magnitude,
        mode='lines',
        name='FFT Magnitude',
        hovertemplate='Frequency: %{x:.2f} Hz<br>Magnitude: %{y:.2e}<extra></extra>'
    ))


def calculate_hrv_from_heart_rate(ppg_obj):
    # Get the heart rate (in bpm)
    heart_rate = ppg_obj['heart_rate']
    
    # Convert heart rate to RR intervals (in seconds)
    rr_intervals = 60 / heart_rate  # RR intervals in seconds
    
    # Calculate SDNN: Standard deviation of RR intervals
    sdnn = np.std(rr_intervals) * 1000  # Convert to milliseconds
    
    # Calculate RMSSD: Square root of the mean of the squared differences of RR intervals
    rr_diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diff**2)) * 1000  # Convert to milliseconds
    
    return sdnn, rmssd
def is_ppg_signal(signal, sampling_rate, min_peak_count=2, freq_range=(0.5, 5), show=False):
    """
    Function to identify if a given signal is likely a PPG signal and compute simple HRV metrics.
    
    Parameters:
    - signal: The input signal.
    - sampling_rate: The sampling rate of the signal.
    - min_peak_count: Minimum number of peaks to identify a periodic signal.
    - freq_range: Frequency range for typical PPG signals (default is 0.5-5 Hz).
    - show: Whether to display the signal with detected peaks.
    
    Returns:
    - is_ppg: Boolean indicating if the signal is likely a PPG signal.
    - details: Dictionary containing the analysis details including simple HRV metrics.
    """
    details = {}
    
    try:
        # Detect peaks in the signal using neurokit2's PPG peak detector
        #peaks, info = nk.ppg_peaks(signal, sampling_rate=sampling_rate, method="bishop", show=show)
        
        # Step 1: Detect peaks using SciPy
        peaks, _ = find_peaks(signal, distance=sampling_rate//2, prominence=0.5)  # Customize params as needed

    except Exception as e:
        details["error"] = f"Peak detection failed: {e}"
        return False, details

    # Convert the peaks to a NumPy array and calculate the number of detected peaks
    #peak_count = np.sum(peaks.to_numpy())
    peak_count = len(peaks)
    details["peak_count"] = peak_count
    
    # If the number of peaks is less than the threshold, it's unlikely to be a PPG signal
    if peak_count < min_peak_count:
        details["error"] = "Too few peaks detected"
        return False, details
    
    try:
        ppg_obj = biosppy.signals.ppg.ppg(signal, sampling_rate=sampling_rate, show=show)
        mean_heart_rate = np.mean(ppg_obj['heart_rate'])

        # Calculate HRV metrics (SDNN, RMSSD) using heart rate
        sdnn, rmssd = calculate_hrv_from_heart_rate(ppg_obj)
        
        # Simple HRV calculations
        hrv_metrics = {}
        hrv_metrics['SDNN'] = sdnn  # SDNN: Standard deviation of RR intervals
        hrv_metrics['RMSSD'] = rmssd  # RMSSD: Root mean square of successive differences
        hrv_metrics['mean_hr'] = mean_heart_rate
        details["hrv_metrics"] = hrv_metrics

        # Check if the calculated heart rate or HRV metrics are out of range
        if mean_heart_rate < 40 or mean_heart_rate > 120:
            return False, details
        if rmssd < 20 or rmssd > 200:
            return False, details

    except Exception as e:
        details["error"] = f"Error in biosppy.signals.ppg.ppg: {e}"
        return False, details
    
    return True, details


def filter_good_bad_segments(df_original, config,seed=42,verbos=False):
    np.random.seed(seed)
    
    SAMPLING_RATE = config['frequencies']['ppg']
    SEGMENT_LENGTH_S = config['segmentation']['segment_length_seconds']
    STEP_SIZE_S = config['segmentation']['step_size_seconds']

    LOWCUT = config['filters']['ppg']['lowcut']
    HIGHCUT = config['filters']['ppg']['highcut']
    
    SYNT_HEART_RATE = config['synthetic_data']['heart_rate']
    SYNT_DURATION = config['synthetic_data']['duration']
    N_FFT = config['synthetic_data']['n_fft']
    NOISE_LEVEL = config['synthetic_data']['noise_level']


    df = df_original.dropna(subset=['ledIR', 'ledRed', 'ledGreen', 'sleep_label'])
    
    led_green = df['ledGreen'].values
    if LOWCUT is not None:
        if HIGHCUT is not None:
            df['ledGreen_filtered'] = bandpass_filter(led_green, LOWCUT, HIGHCUT, SAMPLING_RATE)
        else:
            df['ledGreen_filtered'] = highpass_filter(led_green, LOWCUT, SAMPLING_RATE, order=2)
    else:
        df['ledGreen_filtered'] = led_green  # No filtering if LOWCUT is Non
    
    # SYNTHETIC PPG
    synthetic_ppg = generate_synthetic_ppg(SYNT_DURATION, SAMPLING_RATE, SYNT_HEART_RATE)
    if LOWCUT!=None and HIGHCUT!=None:
        synthetic_ppg=bandpass_filter(synthetic_ppg, LOWCUT, HIGHCUT, SAMPLING_RATE)
    if LOWCUT!=None and HIGHCUT==None:
        synthetic_ppg=highpass_filter(synthetic_ppg, LOWCUT, SAMPLING_RATE, order=2)
        
    #SNR THRESHOLD
    snr_threshold = determine_snr_threshold(synthetic_ppg, SAMPLING_RATE, NOISE_LEVEL,'hann',N_FFT)
    if verbos:
        print("snr_threshold: ",snr_threshold)
    
    # CLASSIFICATION SEGMENT
    segment_length = SEGMENT_LENGTH_S * SAMPLING_RATE  # Segment length in samples
    step_size = STEP_SIZE_S * SAMPLING_RATE       
    segments = overlapping_windows(df['ledGreen_filtered'].values, segment_length, step_size)
    if verbos:
        print("Overlapping segments created")
        print(len(segments))
    
    good_bad_segments_overlap = []
    frequencies = np.fft.fftfreq(N_FFT, 1/SAMPLING_RATE)[:N_FFT // 2]
    signal_band = (frequencies >= LOWCUT) & (frequencies <= HIGHCUT)  # PPG signal band
    noise_band = (frequencies < LOWCUT) | (frequencies > HIGHCUT)     # Noise band
    for i, segment in enumerate(segments):
        if len(segment) < segment_length:
            continue  # Skip segments that are too short
        #snr = calculate_snr(segment, SAMPLING_RATE, signal_band, noise_band, 'hann', N_FFT)
        snr = calculate_snr_around_peaks(segment, SAMPLING_RATE, signal_band, noise_band, 'hann', N_FFT)
        is_ppg, details = is_ppg_signal(segment, sampling_rate=SAMPLING_RATE)
        
        quality = 'good' if snr >= snr_threshold and is_ppg  else 'bad'
        start_idx = i * step_size
        end_idx = start_idx + segment_length
        good_bad_segments_overlap.append((quality, start_idx, end_idx))
        
    if verbos:
        print("segments Classified into good or bad")
    
     # Aggregate the classifications for each signal value
    aggregated_score = aggregate_classifications(good_bad_segments_overlap, 
                                                 len(df['ledGreen_filtered']), 
                                                 segment_length, step_size)
    

    # Define a threshold for bad segments
    threshold = 0  

    # Remove bad segments from the original DataFrame based on the aggregate score
    df_filtered_original = remove_segments_based_on_score(df_original, aggregated_score, threshold)

    sensor_columns_ppg = config['features']['ppg']
    sensor_columns_gyro = config['features']['gyroscope']
    sensor_columns_acc = config['features']['accelerometer']
    
    # Align all sensors by Unix times
    df_aligned = align_sensors_on_time(df_filtered_original, sensor_columns_ppg + sensor_columns_gyro + sensor_columns_acc)
    return df_aligned


if __name__ == '__main__':
    np.random.seed(42)
    
    #READ ACTUAL PPG
    df_original = pd.read_csv('data/train/california-00011978-right-sync.csv')
    #df_original = pd.read_csv('data/adhd_train/adhd-KKI_004-left-sync.csv')
    df_original['sleep_label'] = df_original['sleep_stage'].apply(lambda x: 0 if x == 'WK' else 1)  # 0: awake, 1: sleep
    df = df_original.copy()
    
    df = df.dropna(subset=['ledIR', 'ledRed', 'ledGreen', 'sleep_label'])
    df['ledGreen_filtered']=bandpass_filter(df['ledGreen'].dropna(), LOWCUT, HIGHCUT, SAMPLING_RATE)

    # SYNTHETIC PPG
    synthetic_ppg = generate_synthetic_ppg(SYNT_DURATION, SAMPLING_RATE, SYNT_HEART_RATE)
    synthetic_ppg=bandpass_filter(synthetic_ppg, LOWCUT, HIGHCUT, SAMPLING_RATE)
    
    #SNR THRESHOLD
    snr_threshold = determine_snr_threshold(synthetic_ppg, SAMPLING_RATE, NOISE_LEVEL,'hann',N_FFT)
    print("snr_threshold: ",snr_threshold)
    

    # CLASSIFICATION SEGMENT
    segment_length = SEGMENT_LENGTH_S * SAMPLING_RATE  # Segment length in samples
    step_size = STEP_SIZE_S * SAMPLING_RATE       
    segments = overlapping_windows(df['ledGreen_filtered'].values, segment_length, step_size)
    
    good_bad_segments_overlap = []
    frequencies = np.fft.fftfreq(N_FFT, 1/SAMPLING_RATE)[:N_FFT // 2]
    signal_band = (frequencies >= LOWCUT) & (frequencies <= HIGHCUT)  # PPG signal band
    noise_band = (frequencies < LOWCUT) | (frequencies > HIGHCUT)     # Noise band
    for i, segment in enumerate(segments):
        if len(segment) < segment_length:
            continue  # Skip segments that are too short
        #snr = calculate_snr(segment, SAMPLING_RATE, signal_band, noise_band, 'hann', N_FFT)
        snr = calculate_snr_around_peaks(segment, SAMPLING_RATE, signal_band, noise_band, 'hann', N_FFT)
        print(snr)
        quality = 'good' if snr >= snr_threshold else 'bad'
        start_idx = i * step_size
        end_idx = start_idx + segment_length
        good_bad_segments_overlap.append((quality, start_idx, end_idx))

   
    # Aggregate the classifications for each signal value
    aggregated_score = aggregate_classifications(good_bad_segments_overlap, 
                                                 len(df['ledGreen_filtered']), 
                                                 segment_length, step_size)


    # Define a threshold for bad segments
    threshold = 0  

    # Remove bad segments from the original DataFrame based on the aggregate score
    df_filtered_original = remove_segments_based_on_score(df_original, aggregated_score, threshold)

    print("Original DataFrame shape:", df_original.shape)
    print("Filtered DataFrame shape:", df_filtered_original.shape)
    
    print(df_original.dropna(subset=['ledIR', 'ledRed', 'ledGreen', 'sleep_label']).shape)
    print(df_original.dropna(subset=['gyroscopeX', 'gyroscopeY', 'gyroscopeZ', 'sleep_label']).shape)
    print(df_original.dropna(subset=['accelerometerX', 'gyroscopeY', 'accelerometerY', 'sleep_label']).shape)
    print(df_original.dropna(subset=['tempObject']).shape)
    
    print(df_filtered_original.dropna(subset=['ledIR', 'ledRed', 'ledGreen', 'sleep_label']).shape)
    print(df_filtered_original.dropna(subset=['gyroscopeX', 'gyroscopeY', 'gyroscopeZ', 'sleep_label']).shape)
    print(df_filtered_original.dropna(subset=['accelerometerX', 'gyroscopeY', 'accelerometerY', 'sleep_label']).shape)
    print(df_filtered_original.dropna(subset=['tempObject']).shape)
    
    sensor_columns_ppg = ['ledIR', 'ledRed', 'ledGreen']
    sensor_columns_gyro = ['gyroscopeX', 'gyroscopeY', 'gyroscopeZ']
    sensor_columns_acc = ['accelerometerX', 'accelerometerY', 'accelerometerZ']

    # Align all sensors by Unix times
#     df_aligned = align_sensors_on_time(df_filtered_original, sensor_columns_ppg + sensor_columns_gyro + sensor_columns_acc)

#     print(df_aligned.dropna(subset=['ledIR', 'ledRed', 'ledGreen', 'sleep_label']).shape)
#     print(df_aligned.dropna(subset=['gyroscopeX', 'gyroscopeY', 'gyroscopeZ', 'sleep_label']).shape)
#     print(df_aligned.dropna(subset=['accelerometerX', 'gyroscopeY', 'accelerometerY', 'sleep_label']).shape)
    
#     good_mask = aggregated_score >= 0
#     df_filtered = df[good_mask].reset_index(drop=True)
    
#     print("Original DataFrame shape:", df.shape)
#     print("Filtered DataFrame shape:", df_filtered.shape)
    
    
    
    
    
    