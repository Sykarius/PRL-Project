import torch
from models import Model1, Model2
from datasets import ImitationData
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torch.optim import Adam
import os
import logging

# Training models that take stacked input

parser = argparse.ArgumentParser(
    description=""" python train_stack.py --name | -n [model1 | model2] --epochs | -e [num (def 20)] --batchs | -b [num (def 32)] --learning_rate | -lr [float (def 0.0001)] --ckpt | -c [str]""")
parser.add_argument('--name', '-n', type=str, required=True,
                    help='Name of the model to be trained')
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

    dataset = ImitationData('./frames/stack/')
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    loss_func = nn.CrossEntropyLoss()
    loss_run = 0
    global idx

    with tqdm(range(epochs), desc='Training', unit='epochs') as erange:

        for e in erange:

            erange.set_postfix({'Epoch': e+1, 'loss': loss_run})
            bno = 1
            loss_run = 0
            logger.info(f'Start Epoch {e}')
            for states, actions in dataloader:

                opt.zero_grad()

                states = states.to(device)
                actions[actions == 2] -= 1
                actions[actions == 3] -= 1
                actions = actions.type(torch.LongTensor)
                actions = actions.to(device)
                pred_actions = model(states)

                loss = loss_func(pred_actions, actions)
                loss.backward()

                opt.step()

                with torch.no_grad():
                    loss_run += (loss.to(device_cpu).item() - loss_run)/bno

                erange.set_postfix(
                    {'Epoch': e+1, 'Batch': bno, 'loss': loss_run})

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

    model_name = args.name
    assert model_name in ['model1', 'model2']
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

    logger = logging.getLogger('training-stack')
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

    if model_name == 'model1':
        model = Model1().to(device)
    else:
        model = Model2().to(device)

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
