from transforms import *
from datasets import *
from configs import *
from models import *


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


def get_lr(opt):
    return opt.param_groups[0]['lr']


def main():
    global best_accuracy, global_step, start_epoch, start_timestamp, best_loss
    parser = argparse.ArgumentParser(description='Train model for speech command',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, default='./configs/configs.yaml')

    use_gpu = torch.cuda.is_available()
    device = 'cpu'
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        device = 'cuda'

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

    # Create model
    model = CifarResNeXt(nlabels=len(CLASSES), in_channels=1)
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # Init criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Init optimizer
    if conf.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=conf.lr,
                                    momentum=0.9,
                                    weight_decay=conf.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=conf.lr,
                                     weight_decay=conf.weight_decay)

    # Init learning rate scheduler
    if conf.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=conf.lr_scheduler_patience,
                                                                  factor=conf.lr_scheduler_gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_scheduler_step_size,
                                                       gamma=conf.lr_scheduler_gamma, last_epoch=start_epoch - 1)

    # Resuming mode
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

    # Create checkpoint path
    if not os.path.exists(conf.checkpoint_path):
        os.mkdir(conf.checkpoint_path)

    # Init name model and board
    name = 'resnext_%s_%s' % (conf.optimizer, conf.batch_size)
    writer = SummaryWriter(comment=('_speech_commands_') + name)

    def train(epoch):
        global global_step
        phase = 'train'
        print(f'Epoch {epoch} - lr {get_lr(optimizer)}')
        writer.add_scalar('%s/learning_rate' % phase, get_lr(optimizer), epoch)

        model.train()
        running_loss = 0.0
        it = 0
        correct = 0
        total = 0

        pbar = tqdm(train_dataloader)
        for batch in pbar:
            inputs = batch['input']
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch['target']

            inputs = torch.autograd.Variable(inputs, requires_grad=True)
            targets = torch.autograd.Variable(targets, requires_grad=False)

            if use_gpu:
                inputs = inputs.to(device)
                targets = targets.to(device)

            preds, _ = model(inputs)

            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            it += 1
            global_step += 1
            running_loss += loss.item()
            predicted = preds.data.max(1, keepdim=True)[1]

            correct += predicted.eq(targets.data.view_as(predicted)).sum()
            total += targets.size(0)

            pbar.set_postfix({
                'loss': "%.05f" % (running_loss / it),
                'acc': "%.02f%%" % (100 * correct / total)
            })

        accuracy = correct / total
        epoch_loss = running_loss / it
        writer.add_scalar('%s/accuracy' % phase, 100 * accuracy, epoch)
        writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    def valid(epoch):
        global best_accuracy, best_loss, global_step

        phase = 'valid'
        model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        it = 0
        correct = 0
        total = 0

        pbar = tqdm(valid_dataloader)
        with torch.no_grad():
            for batch in pbar:
                inputs = batch['input']
                inputs = torch.unsqueeze(inputs, 1)
                targets = batch['target']

                if use_gpu:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                # forward
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)

                # statistics
                it += 1
                global_step += 1
                running_loss += loss.item()
                pred = outputs.data.max(1, keepdim=True)[1]
                correct += pred.eq(targets.data.view_as(pred)).sum()
                total += targets.size(0)

                writer.add_scalar('%s/loss' % phase, loss.item(), global_step)

                # update the progress bar
                pbar.set_postfix({
                    'loss': "%.05f" % (running_loss / it),
                    'acc': "%.02f%%" % (100 * correct / total)
                })

        accuracy = correct / total
        epoch_loss = running_loss / it
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
            torch.save(save_checkpoint,
                       conf.checkpoint_path + '/' + 'best-loss-speech-commands-checkpoint-%s.pth' % name)
            torch.save(model, conf.checkpoint_path + '/' + 'best-loss.pth' % (start_timestamp, name))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(save_checkpoint,
                       conf.checkpoint_path + '/' + 'best-acc-speech-commands-checkpoint-%s.pth' % name)
            torch.save(model, conf.checkpoint_path + '/' + 'best-acc.pth' % (start_timestamp, name))

        torch.save(save_checkpoint, conf.checkpoint_path + '/' + 'last-speech-commands-checkpoint.pth')
        del checkpoint  # reduce memory
        return epoch_loss

    print("Training ...")
    since = time.time()
    for epoch in range(start_epoch, conf.max_epochs):
        if conf.lr_scheduler == 'step':
            lr_scheduler.step()

        train(epoch)
        epoch_loss = valid(epoch)

        if conf.lr_scheduler == 'plateau':
            lr_scheduler.step(metrics=epoch_loss)

        time_elapsed = time.time() - since
        time_str = 'Total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600,
                                                                         time_elapsed % 3600 // 60,
                                                                         time_elapsed % 60)
        print("%s, Best accuracy: %.02f%%, best loss %f" % (time_str, 100 * best_accuracy, best_loss))
    print("Finished")


if __name__ == '__main__':
    main()
