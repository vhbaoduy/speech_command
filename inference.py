import numpy as np

import utils
from transforms import *
from datasets import *
from configs import *
from models import *

from torchvision.transforms import Compose
from torch.utils.data import WeightedRandomSampler, DataLoader
import torch
from tensorboardX import SummaryWriter
import argparse
import pandas as pd
import os
from tqdm import tqdm
import time


def main():
    parser = argparse.ArgumentParser(description='Inference phase',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, default='./configs/configs.yaml')

    # Parse agr and load config
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    device = 'cpu'
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        device = 'cuda'

    conf = Configuration(args.config_file)
    df = pd.read_csv(os.path.join(conf.output_path, conf.inference_file))
    transform = build_transform(conf, mode='valid')

    inference_dataset = SpeechCommandsDataset(conf.dataset_path,
                                              df,
                                              conf.sample_rate,
                                              transform)
    dataloader = DataLoader(inference_dataset,
                            batch_size=conf.batch_size,
                            shuffle=False,
                            num_workers=conf.num_workers,
                            pin_memory=use_gpu)

    model = CifarResNeXt(nlabels=len(CLASSES), in_channels=1)
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    try:
        checkpoint = torch.load(os.path.join(conf.checkpoint_path, conf.checkpoint_name))
        model.load_state_dict(checkpoint['state_dict'])
    except:
        pass

    if not os.path.exists(conf.inference_path):
        os.mkdir(conf.inference_path)
    pbar = tqdm(dataloader)

    with torch.no_grad():
        for batch in pbar:
            inputs = batch['input']
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch['target']
            if use_gpu:
                inputs = inputs.to(device)
                targets = targets.to(device)

            # forward
            outputs, feats = model(inputs)
            prediction = outputs.data.max(1, keepdim=True)[1]

            # Convert gpu to cpu
            prediction = prediction.cpu().numpy().ravel()
            targets = targets.cpu().numpy().ravel()
            feats = feats.cpu().numpy()

            for i in range(len(batch['path'])):
                file_name = batch['path'][i]
                name_class = utils.index_to_label(CLASSES, targets[i])
                folder = os.path.join(conf.inference_path, name_class)
                if not os.path.exists(folder):
                    os.mkdir(folder)
                audio_name = file_name.split('/')[1].split('.')[0]
                # print(feats[i])
                # print(feats[i].shape)

                if conf.label_choice:
                    if targets[i] == prediction[i]:
                        np.save(os.path.join(folder, audio_name + ".npy"), feats[i])
                else:
                    np.save(os.path.join(folder, audio_name + ".npy"), feats[i])



if __name__ == '__main__':
    main()
