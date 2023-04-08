import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pydub import AudioSegment

st.title("MP3 Song Spectrogram Beat Visualizer")

# File upload
audio_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

if audio_file:
    # Convert the MP3 file to a WAV file
    audio = AudioSegment.from_file(audio_file)
    audio.export("temp.wav", format="wav")

    # Open the WAV file
    with open("temp.wav", "rb") as f:
        data = f.read()

    # Get the audio file parameters
    num_channels = audio.channels
    sample_rate = audio.frame_rate
    sample_width = audio.sample_width
    num_frames = len(audio.get_array_of_samples())

    # Convert the audio data to a numpy array
    audio_array = np.frombuffer(data, dtype=np.int16)

    # Reshape the numpy array to a 2D array of shape (num_channels, num_frames)
    audio_array = np.reshape(audio_array, (num_frames, num_channels))

    # Compute the spectrogram
    window_size = int(0.01 * sample_rate)  # Window size of 10ms
    window = np.hanning(window_size)
    hop_size = int(0.01 * sample_rate)  # Hop size of 10ms
    num_fft_points = 2 * window_size
    spectrogram = np.zeros((num_frames // hop_size, num_fft_points // 2 + 1))
    for i in range(0, num_frames - window_size, hop_size):
        # Apply window to the audio data
        windowed_audio = window * audio_array[i:i+window_size, :]

        # Compute the magnitude spectrum using FFT
        magnitude_spectrum = np.abs(np.fft.rfft(windowed_audio, n=num_fft_points, axis=0))

        # Convert to decibels
        magnitude_spectrum_db = 20 * np.log10(magnitude_spectrum + np.finfo(float).eps)

        # Store the spectrogram
        spectrogram[i // hop_size, :] = magnitude_spectrum_db[:, 0]

    # Compute the beat
    mean_spectrogram = np.mean(spectrogram, axis=1)
    diff_spectrogram = np.diff(mean_spectrogram)
    threshold = np.max(diff_spectrogram) * 0.75
    beat_frames = np.where(diff_spectrogram > threshold)[0]
    beat_times = beat_frames * hop_size / sample_rate

    # Plot the spectrogram and beat
    fig, ax = plt.subplots()
    ax.imshow(spectrogram.T, aspect='auto', origin='lower')
    ax.plot(beat_frames, np.ones_like(beat_frames) * (num_fft_points // 2 + 1), 'r.')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram with Beat")

    # Show the plot
    st.pyplot(fig)

    # Print the beat times
    st.write(f"Beat times (s): {beat_times}")
