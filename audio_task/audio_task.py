try:
    import sys
    import argparse
    import librosa
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from tqdm import tqdm
    # pip install PyQt5
    matplotlib.use('qtagg')
except ImportError as impErr:
    print(f"[Error]: Required packages not installed\n{impErr}")
    sys.exit(1)


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


class AudioDataNotValid(BaseException):
    """
    Exception raised for invalid audio data.
    """
    pass


def load_audio_files(directory: str, file_paths: list[str]) -> list[tuple[np.ndarray, int]]:
    """
    Loads audio files from the specified directory.

    Parameters
    ----------
    directory : str
        The parent directory with audio files.
    file_paths : list[str]
        List of audio file paths relative to the directory.

    Returns
    -------
    list[tuple[np.ndarray, int]]
        List of tuples with each tuple containing arrya and sampling rate of the loaded audio file.
    """
    loaded_sounds = []
    for file_path in tqdm(file_paths, desc='Loading Files'):
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


def plot_spectogram_hz(sound_names: list[str], raw_sounds: list[tuple[np.ndarray, int]]):
    """
    Plots spectogram of sounds in Hz scale.

    Parameters
    ----------
    sound_names : list[str]
        List of sound names.
    raw_sounds : list[tuple[np.ndarray, int]]
        List of tuples containing raw sound array and its sampling rate.

    """
    for sound_name, (y, _) in tqdm(zip(sound_names, raw_sounds), desc='Generating Spectogram'):
        if librosa.util.valid_audio(y):
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(
                S_db, x_axis='time', y_axis='log', ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title=f'Spectrogram of {sound_name}')
            plt.tight_layout()
            fig.canvas.manager.set_window_title(
                f'[Log(Hz) Scale] {sound_name}')
        else:
            raise AudioDataNotValid(f"Invalid audio data for {sound_name}")
    plt.show()


def plot_spectogram_note_scale(sound_names: list[str], raw_sounds: list[tuple[np.ndarray, int]]):
    """
    Plots spectogram of sounds in log scale with pitches marked.

    Parameters
    ----------
    sound_names : list[str]
        List of sound names.
    raw_sounds : list[tuple[np.ndarray, int]]
        List of tuples containing raw sound array and its sampling rate.
    """
    for sound_name, (y, _) in zip(sound_names, raw_sounds):
        if librosa.util.valid_audio(y):
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(
                S_db, x_axis='time', y_axis='fft_note', ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title=f'Spectrogram of {sound_name}')
            plt.tight_layout()
            fig.canvas.manager.set_window_title(f'[Note Scale] {sound_name}')
        else:
            raise AudioDataNotValid(f"Invalid audio data for {sound_name}")
    plt.show()


def main():
    # parent_directory = "./audio_data/Actor_01"
    # sound_file_paths = ['03-01-01-01-01-01-01.wav',
    #                     '03-01-01-01-01-02-01.wav', '03-01-01-01-02-01-01.wav']

    parser = argparse.ArgumentParser(
        description="Load and plot spectrograms of audio files.")
    parser.add_argument("parent_directory", type=str,
                        help="The parent directory with audio files.")
    parser.add_argument("sound_file_paths", type=str, nargs='+',
                        help="List of audio file paths relative to the directory.")
    args = parser.parse_args()

    parent_directory = args.parent_directory
    sound_file_paths = args.sound_file_paths
    sound_names = [f'Sound {i+1}' for i in range(len(sound_file_paths))]

    sound_names = [f'Sound {i}' for i in sound_file_paths]

    loaded_sounds = load_audio_files(parent_directory, sound_file_paths)

    while True:
        print("\nChoose an option:")
        print("1. Plot spectrogram in Hz scale")
        print("2. Plot spectrogram in note scale")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")
        match choice:
            case '1':
                plot_spectogram_hz(sound_names, loaded_sounds)
            case '2':
                plot_spectogram_note_scale(sound_names, loaded_sounds)
            case '3':
                print('Exiting the program')
                break
            case _:
                print("Invalid Choice")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt. Exiting...")
        sys.exit(0)
