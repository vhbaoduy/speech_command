from datasets import BackgroundNoiseDataset
from .transform_wav import *
from .transform_stft import *
from torchvision.transforms import Compose


def build_transform(conf, mode='train'):
    """
    Build transform for train and valid dataset
    :param conf: Configuration from config file
    :param mode: 'train', 'valid'
    :return: data augmentation, background noise, feature transform
    """
    if mode == 'train':
        data_aug_transform = Compose(
            [ChangeAmplitude(),
             ChangeSpeedAndPitchAudio(),
             FixAudioLength(),
             ToSTFT(),
             StretchAudioOnSTFT(),
             TimeshiftAudioOnSTFT(),
             FixSTFTDimension()
             ])
        if conf.background_noise:
            bg_dataset = BackgroundNoiseDataset(conf.background_noise_path, data_aug_transform, conf.sample_rate)
            add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
        train_feature_transform = Compose(
            [ToMelSpectrogramFromSTFT(n_mels=conf.mel_spectrogram),
             DeleteSTFT(),
             ToTensor('mel_spectrogram', 'input')])
        return Compose([data_aug_transform, add_bg_noise, train_feature_transform])

    if mode == 'valid':
        valid_transform = Compose([FixAudioLength(),
                                   ToMelSpectrogram(n_mels=conf.mel_spectrogram),
                                   ToTensor('mel_spectrogram', 'input')])
        return valid_transform

    return None
