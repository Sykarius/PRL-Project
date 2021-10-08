import torch
from models import Model3
import torch.nn as nn
from tqdm import tqdm
import argparse
from torch.optim import Adam
import os
import logging
import pickle

# Training models that take stacked input

parser = argparse.ArgumentParser(
    description=""" python train_nostack.py --epochs | -e [num (def 20)] --batchs | -b [num (def 32)] --learning_rate | -lr [float (def 0.0001)] --ckpt | -c [str]""")
parser.add_argument('--epochs', '-e', type=int,
                    default=20, help='No of epochs')
parser.add_argument('--batchs', '-b', type=int,
                    default=32, help='Batch size')
parser.add_argument('--learning_rate', '-lr', type=float,
                    default=0.0001, help='Learning Rate')
parser.add_argument('--ckpt', '-c', type=str,
                    default=None, help='Checkpoint Name')


def training(model, opt, params):

    batch_size = params['batch_size']
    epochs = params['epochs']
    global idx

    loss_func = nn.CrossEntropyLoss(
        torch.tensor([0.2, 2.5, 1., 1.]).to(device))

    frame_dir = './frames/nostack/'
    game_dirs = [frame_dir + x + '/' for x in os.listdir(frame_dir)]

    game_frames = dict()
    game_action = dict()
    for g in game_dirs:
        game_frames[g] = []
        game_action[g] = []
        for f in os.listdir(g):
            game_frames[g].append(g+f)
            game_action[g].append(int(f.rstrip('.pkl').split('_')[1]))

    loss_run = 0
    with tqdm(range(epochs), desc='Training', unit='epochs') as erange:

        for e in erange:

            erange.set_postfix({'Epoch': e+1, 'Game': 0, 'loss': loss_run})
            logger.info(f'Start Epoch {e}')
            for g in game_dirs:

                end = len(game_frames[g])
                i = 0
                loss_run = 0
                hidden = model.init_hidden(device)
                bno = 1

                while i+batch_size < end:

                    curb = []
                    opt.zero_grad()

                    erange.set_postfix(
                        {'Epoch': e+1, 'Game': g[-2:], 'Batch': bno, 'loss': loss_run})

                    for f in game_frames[g][i: i+batch_size]:
                        with open(f, 'rb') as f:
                            curb.append(pickle.load(f).unsqueeze(0))

                    batch_frames = torch.cat(curb, dim=0).type(
                        torch.FloatTensor).to(device)
                    actions = torch.tensor(
                        game_action[g][i: i+batch_size]).to(device)
                    pred_actions, hidden = model(batch_frames, hidden)

                    loss = loss_func(pred_actions, actions)
                    loss.backward()

                    opt.step()
                    hidden = (hidden[0].detach(), hidden[1].detach())  # TBTT

                    with torch.no_grad():
                        loss_run += (loss.to(device_cpu).item() - loss_run)/bno

                    i += batch_size
                    bno += 1

            if (e+1) % 10 == 0:

                saved_dict = {'model': model.state_dict(),
                              'opt': opt.state_dict()}

                torch.save(saved_dict, ckpt_path + 'ckpt_' + str(idx) + '.pt')
                logger.info(f'Checkpointing ckpt_{str(idx)}')
                idx += 1

            logger.info(f'Completed Epoch {e}')


if __name__ == '__main__':

    args = parser.parse_args()

    model_name = 'model3'
    epochs = args.epochs
    batch_size = args.batchs
    lr = args.learning_rate
    ckpt = args.ckpt

    ckpt_path = 'saved/' + model_name + '/'

    ckpt_ids = sorted([x.replace('.pt', '').split('_')[1]
                       for x in os.listdir(ckpt_path)])

    if not len(ckpt_ids):
        idx = 0
    else:
        idx = int(ckpt_ids[-1])+1

    logger = logging.getLogger('training-nostack')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
    file_handler = logging.FileHandler(
        os.path.join('logs/', 'training.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f'Training Session for {model_name}')

    device_cpu = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')

    else:
        device = device_cpu

    params = dict()
    params['batch_size'] = batch_size
    params['epochs'] = epochs

    model = Model3().to(device)

    opt = Adam(model.parameters(), lr)

    if ckpt is not None and os.path.exists(ckpt_path + ckpt + '.pt'):
        with open(ckpt_path + ckpt + '.pt', 'rb') as f:
            saved_dict = torch.load(f)

        model.load_state_dict(saved_dict['model'])
        opt.load_state_dict(saved_dict['opt'])
        logger.info(f'Loaded {ckpt}')

    # Start Training
    training(model, opt, params)
    logger.info('Training session complete')

    saved_dict = {'model': model.state_dict(), 'opt': opt.state_dict()}

    torch.save(saved_dict, ckpt_path + 'ckpt_' + str(idx) + '.pt')
    logger.info(f'Saved to ckpt ckpt_{str(idx)}')
