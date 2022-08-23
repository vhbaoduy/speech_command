import random
import librosa


def label_to_index(labels: object, label: str):
    """
    Convert label to index
    :param labels: List of labels
    :param label: Label of input/prediction
    :return: Index of label in integer
    """
    return labels.index(label)


def index_to_label(labels: object, index: int):
    """
    Convert index to label(class)
    :param labels: List of labels
    :param index: the index
    :return: the label of index in classes
    """
    return labels[index]


def load_audio(path:str, sample_rate:int):
    """
    Load audio from path with sample rate
    :param path: Path to audio
    :param sample_rate: The sample rate of audio
    :return: sample, sample_rate
    """
    sample, sample_rate = librosa.load(path=path, sr=sample_rate)
    return sample, sample_rate


def is_apply_transform(prob=0.5):
    return random.Random() < prob


# Debugging
if __name__ == '__main__':
    print('hello')
