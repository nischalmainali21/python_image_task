import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

SUPPORTED_FORMATS = ['aiff',
                     'au',
                     'avr',
                     'caf',
                     'flac',
                     'htk',
                     'svx',
                     'mat4',
                     'mat5',
                     'mpc2k',
                     'mp3',
                     'ogg',
                     'paf',
                     'pvf',
                     'raw',
                     'rf64',
                     'sd2',
                     'sds',
                     'ircam',
                     'voc',
                     'w64',
                     'wav',
                     'nist',
                     'wavex',
                     'wve',
                     'xi']


def load_audio_files(directory: str, file_paths: list[str]) -> list[tuple[np.ndarray, int]]:
    """Loads audio files from the specified directory.

    Args:
        directory (str): The parent directory with audio files.
        file_paths (list): List of audio file paths relative to the directory.

    Returns:
        list: list of tuples with each tuple data of the loaded audio file.
    """
    loaded_sounds = []
    for file_path in file_paths:
        full_path = os.path.join(directory, file_path)
        if os.path.exists(full_path):
            file_format = os.path.splitext(full_path)[1][1:]
            if file_format.lower() in SUPPORTED_FORMATS:
                try:
                    y, sr = librosa.load(full_path)
                    loaded_sounds.append((y, sr))
                except Exception as e:
                    print(e)
            else:
                print(f'Unsupported file format for {full_path}. Skipping...')
        else:
            print(f'File {full_path} does not exist. Skipping...')
    return loaded_sounds


def plot_spectogram_hz(sound_names, raw_sounds):
    """Plots spectogram of sounds in Hz scale.

    Args:
        sound_names (list): List of sound names.
        raw_sounds (list): list of tuples containing raw sound array and its sampling rate.
    """
    # check if proper data,
    for sound_name, (y, sr) in zip(sound_names, raw_sounds):
        if librosa.util.valid_audio(y):
            plt.figure(figsize=(10, 4))
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            plt.subplot(1, 1, 1)
            librosa.display.specshow(S_db, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram in Hz scale of {sound_name}')
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    parent_directory = "./audio_data/Actor_01"
    sound_file_paths = ['03-01-01-01-01-01-01.wav',
                        '03-01-01-01-01-02-01.wav', '03-01-01-01-02-01-01.wav']

    loaded_sounds = load_audio_files(parent_directory, sound_file_paths)

    sound_names = ["First Voiceline", "Second Voiceline", "Third Voiceline"]
    plot_spectogram_hz(sound_names, loaded_sounds)
    # print(loaded_sounds)
