import pandas as pd
import os
import matplotlib.pyplot as plt


class DataPreparing(object):
    def __init__(self, dataset_path, output_path, ratio_split):
        self.dataset_path = dataset_path
        self.ratio_split = ratio_split
        self.output_path = output_path
        self.train = None
        self.valid = None
        self.classes = []
        self.n_classes = 0
        self.speakers = []
        self.n_speakers = 0

    def create_dataframe(self):
        # audio_path = os.path.join(self.dataset_path, 'audio')
        # Insert the name of class
        for folder in os.listdir(self.dataset_path):
            if folder != '_background_noise_':
                self.classes.append(folder)
        self.n_classes = len(self.classes)
        train = {
            "file_name": [],
            "speaker": [],
            "vocab": []
        }
        valid = {
            "file_name": [],
            "speaker": [],
            "vocab": []
        }
        for name in self.classes:
            data_path = os.path.join(self.dataset_path, name)
            files = os.listdir(data_path)
            idx_split = int((1-self.ratio_split)*len(files))
            for i, file in enumerate(files):
                if file.endswith(".wav"):
                    parsing = file.split("_")
                    speaker = parsing[0]
                    file_name = name + "/" + file
                    if i < idx_split:
                        train["speaker"].append(speaker)
                        train["vocab"].append(name)
                        train["file_name"].append(file_name)
                    else:
                        valid["speaker"].append(speaker)
                        valid["vocab"].append(name)
                        valid["file_name"].append(file_name)

                    if speaker not in self.speakers:
                        self.speakers.append(speaker)

        self.n_speakers = len(self.speakers)
        # classes = sorted(self.classes)
        # s = ', '.join([str(x) for x in classes])
        # print(s)
        # Create dataframe
        # Store dataframe
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        df_train = pd.DataFrame(train)
        df_train.to_csv(os.path.join(self.output_path, 'train.csv'), index=False)
        self.train = df_train
        df_valid = pd.DataFrame(valid)
        df_valid.to_csv(os.path.join(self.output_path, 'valid.csv'), index=False)
        self.valid = df_valid

    def print_dataset_statistics(self):
        print("#" * 5, "DATASET STATISTICS", "#" * 5)
        print('The number of classes: {}'.format(self.n_classes))
        print('The number of speakers: {}'.format(self.n_speakers))
        print('The number of rows in dataframe: {} (Train {} Valid {})'.format(len(self.train) + len(self.valid),
                                                                               len(self.train),
                                                                               len(self.valid)))
        print("#" * 50)

    def visualize_dataset_statistics(self):
        df = pd.concat([self.train, self.valid], ignore_index=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))
        class_counter = df.groupby(['vocab'])['vocab'].count()
        ax1.bar(class_counter.index.values.tolist(), class_counter, width=0.3)
        ax1.set_title('Vocab bar chart')
        ax1.set_ylabel('Count')

        speaker_counter = df.groupby(['speaker'])['speaker'].count().reset_index(name='counts')
        ax2.boxplot(speaker_counter['counts'])
        ax2.set_title('Speaker Boxplot')
        ax2.set_xlabel('Speaker')
        ax2.set_ylabel('Count')

        fig.savefig(os.path.join(self.output_path, 'dataset-statistic.png'))
        # plt.show()
