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

start_timestamp = int(time.time() * 1000)
start_epoch = 0
best_accuracy = 0
best_loss = 1e100
global_step = 0


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
             FixAudioLength(), ToSTFT(),
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
        valid_feature_transform = Compose([ToMelSpectrogram(n_mels=conf.mel_spectrogram),
                                           ToTensor('mel_spectrogram', 'input')])
        return Compose([FixAudioLength(), valid_feature_transform])

    return None


def get_lr(opt):
    return opt.param_groups[0]['lr']

def main():
    global best_accuracy, global_step, start_epoch
    parser = argparse.ArgumentParser(description='Train model for speech command',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, default='./configs/configs.yaml')

    use_gpu = torch.cuda.is_available()
    device = 'cpu'
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        device = 'gpu'

    # Parse agr and load config
    args = parser.parse_args()
    conf = Configuration(args.config_file)

    # Load dataframe
    df_train = pd.read_csv(os.path.join(conf.output_path, 'train.csv'))
    df_valid = pd.read_csv(os.path.join(conf.output_path, 'valid.csv'))

    # Build transform
    train_transform = build_transform(conf, mode='train')
    valid_transform = build_transform(conf, mode='valid')

    train_dataset = SpeechCommandsDataset(conf.dataset_path,
                                          df_train,
                                          conf.sample_rate,
                                          train_transform)

    valid_dataset = SpeechCommandsDataset(conf.dataset_path,
                                          df_valid,
                                          conf.sample_rate,
                                          valid_transform)

    weights = train_dataset.make_weights_for_balanced_classes()
    sampler = WeightedRandomSampler(weights, len(weights))

    # Data loader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=conf.batch_size,
                                  sampler=sampler,
                                  num_workers=conf.num_workers,
                                  pin_memory=use_gpu)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=conf.batch_size,
                                  shuffle=False,
                                  num_workers=conf.num_workers,
                                  pin_memory=use_gpu)

    model = CifarResNeXt(nlabels=len(CLASSES), in_channels=1)
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss()

    if conf.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=conf.lr,
                                    momentum=0.9,
                                    weight_decay=conf.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=conf.lr,
                                     weight_decay=conf.weight_decay)

    if conf.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=conf.lr_scheduler_patience,
                                                                  factor=conf.lr_scheduler_gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_scheduler_step_size,
                                                       gamma=conf.lr_scheduler_gamma, last_epoch=start_epoch - 1)

    if conf.resume:
        print("Resuming a checkpoint '%s'" % conf.checkpoint_name)
        checkpoint = torch.load(os.path.join(conf.checkpoint_path, conf.checkpoint_name))
        model.load_state_dict(checkpoint['state_dict'])
        model.float()
        optimizer.load_state_dict(checkpoint['optimizer'])

        best_accuracy = checkpoint.get('accuracy', best_accuracy)
        best_loss = checkpoint.get('loss', best_loss)
        start_epoch = checkpoint.get('epoch', start_epoch)
        global_step = checkpoint.get('step', global_step)

        del checkpoint  # reduce

    name = 'resnext_%s_%s' % (conf.optim, conf.batch_size)
    writer = SummaryWriter(comment=('_speech_commands_') + name)


    def train(epoch):
        global global_step
        phase = 'train'
        print(f'Epoch {epoch} - lr {get_lr(optimizer)}')
        writer.add_scalar('%s/learning_rate' % phase, get_lr(optimizer), epoch)

        model.train()
        running_loss = 0.0
        iter = 0
        correct = 0
        total = 0

        pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)
        for batch in pbar:
            inputs = batch['input']
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch['target']

            inputs = torch.autograd.Variable(inputs, requires_grad=True)
            targets = torch.autograd.Variable(targets, requires_grad=False)

            if use_gpu:
                inputs = inputs.to(device)
                targets = inputs.to(device)

            preds, _ = model(inputs)

            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter += 1
            global_step += 1
            running_loss += loss.item()
            predicted = preds.data.max(1, keepdim=True)[1]

            correct += predicted.eq(targets.data.view_as(predicted)).sum()
            total += targets.size(0)

            pbar.set_postfix({
                'loss': "%.05f" % (running_loss / iter),
                'acc': "%.02f%%" % (100 * correct / total)
            })

        accuracy = correct / total
        epoch_loss = running_loss / iter
        writer.add_scalar('%s/accuracy' % phase, 100 * accuracy, epoch)
        writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)


    def valid(epoch):
        global best_accuracy, best_loss, global_step

        phase = 'valid'
        model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        iter = 0
        correct = 0
        total = 0

        pbar = tqdm(valid_dataloader, unit="audios", unit_scale=valid_dataloader.batch_size)
        for batch in pbar:
            inputs = batch['input']
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch['target']

            inputs = torch.autograd.Variable(inputs, volatile=True)
            targets = torch.autograd.Variable(targets, requires_grad=False)

            if use_gpu:
                inputs = inputs.to(device)
                targets = targets.to(device)

            # forward
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            # statistics
            iter += 1
            global_step += 1
            running_loss += loss.data[0]
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()
            total += targets.size(0)

            writer.add_scalar('%s/loss' % phase, loss.data[0], global_step)

            # update the progress bar
            pbar.set_postfix({
                'loss': "%.05f" % (running_loss / iter),
                'acc': "%.02f%%" % (100 * correct / total)
            })

        accuracy = correct / total
        epoch_loss = running_loss / iter
        writer.add_scalar('%s/accuracy' % phase, 100 * accuracy, epoch)
        writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

        save_checkpoint = {
            'epoch': epoch,
            'step': global_step,
            'state_dict': model.state_dict(),
            'loss': epoch_loss,
            'accuracy': accuracy,
            'optimizer': optimizer.state_dict(),
        }

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(save_checkpoint, 'checkpoints/best-loss-speech-commands-checkpoint-%s.pth' % name)
            torch.save(model, '%d-%s-best-loss.pth' % (start_timestamp, name))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(save_checkpoint, 'checkpoints/best-acc-speech-commands-checkpoint-%s.pth' % name)
            torch.save(model, '%d-%s-best-acc.pth' % (start_timestamp, name))

        torch.save(save_checkpoint, 'checkpoints/last-speech-commands-checkpoint.pth')
        del checkpoint  # reduce memory
        return epoch_loss


    print("Training %s ...")
    since = time.time()
    for epoch in range(start_epoch, conf.max_epochs):
        if conf.lr_scheduler == 'step':
            lr_scheduler.step()

        train(epoch)
        epoch_loss = valid(epoch)

        if conf.lr_scheduler == 'plateau':
            lr_scheduler.step(metrics=epoch_loss)

        time_elapsed = time.time() - since
        time_str = 'Total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60,
                                                                         time_elapsed % 60)
        print("%s, Best accuracy: %.02f%%, best loss %f" % (time_str, 100 * best_accuracy, best_loss))
    print("finished")

if __name__ == '__main__':
    main()