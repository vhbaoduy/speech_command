import os.path
import random
import librosa
import numpy as np


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


def load_audio(path: str, sample_rate: int):
    """
    Load audio from path with sample rate
    :param path: Path to audio
    :param sample_rate: The sample rate of audio
    :return: sample, sample_rate
    """
    sample, sample_rate = librosa.load(path=path, sr=sample_rate)
    # if len(sample) < sample_rate:
    #     sample = librosa.util.fix_length(sample, size=sample_rate)
    return sample, sample_rate


def is_apply_transform(prob=0.5):
    return random.random() < prob


def load_features(path: str, n=None):
    if os.path.isdir(path):
        vocab_name = path.split('\\')[-1]
        features = []
        for file_name in os.listdir(path):
            if file_name.endswith('.npy'):
                feat = np.load(os.path.join(path, file_name))
                features.append(feat)
                if n is not None and len(features) == n:
                    break
        return vocab_name, np.array(features), len(features)

    return None, None, None


def compute_variance(features):
    mean = np.mean(features, axis=0)
    norm = features - mean
    variance = np.mean(norm ** 2)
    radius = np.max(np.linalg.norm(norm, axis=0))
    return variance, radius


def visualize(feature_path, classes, n_class=5, n_samples=100):
    random_choice = []
    while len(random_choice) < n_class:
        choice = random.choice(classes)
        if choice not in random_choice:
            random_choice.append(choice)


