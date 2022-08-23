import os
import numpy as np
from torch.utils.data import Dataset

import utils

CLASSES = 'bed, bird, cat, dog, down, eight, five, four, go, happy, house, left, marvin, nine, ' \
          'no, off, on, one, right, seven, sheila, six, stop, three, tree, two, up, wow, yes, zero'.split(', ')


class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir, df, sample_rate, transform, classes=CLASSES):
        self.root_dir = root_dir
        self.classes = classes
        self.df = df
        self.sample_rate = sample_rate
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        path = os.path.join(self.root_dir, sample['file_name'])
        label = utils.label_to_index(self.classes, sample['vocab'])
        sample, sample_rate = utils.load_audio(path, self.sample_rate)
        data = {
            'samples': sample,
            'sample_rate': sample_rate,
            'target': label
        }
        return data, label

    def make_weights_for_balanced_classes(self):
        df_count = self.df.groupby(self.df['vocab'])['vocab'].count()
        N = sum(df_count)
        weight = N / np.array(df_count.loc[self.df['vocab']])
        return weight


class BackgroundNoiseDataset(Dataset):
    def __init__(self, path, transform, sample_rate, sample_length=1):
        noise_files = [file for file in os.listdir(path) if file.endswith('.wav')]
        samples = []
        for f in noise_files:
            noise_path = os.path.join(path, f)
            sample, sample_rate = utils.load_audio(noise_path, sample_rate)
            samples.append(sample)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r * c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.classes = CLASSES
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data

if __name__ == '__main__':
    print("helloword")
