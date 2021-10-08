import gym
from torchvision import transforms
from datasets import ChannelFirst, FrameStack
import argparse
import torch
import numpy as np
import os
from models import Model1, Model2

parser = argparse.ArgumentParser(
    description=""" python test_stack.py --name | -n [model1 | model2] --max_steps | -s [num (def 20000)] --ckpt | -c [str]""")
parser.add_argument('--name', '-n', type=str, required=True,
                    help='Name of the model to be tested')
parser.add_argument('--max_steps', '-s', type=int,
                    default=20000, help='Max Steps in the simulation')
parser.add_argument('--ckpt', '-c', type=str,
                    default=None, help='Checkpoint Name', required=True)


def play(model, max_steps):

    env = gym.make('BreakoutNoFrameskip-v4')
    env = gym.wrappers.Monitor(
        env, video_dir, force=True, video_callable=lambda _: True)
    s = preprocess(np.array(env.reset())).type(
        torch.FloatTensor).to(device)
    a = 1
    for i in range(max_steps):
        if i % 100 == 0:
            a = 1
        else:
            out = model(s)
            a = torch.argmax(out, axis=1).to(device_cpu).item()
            if a == 1 or a == 2:
                a += 1
        ns, _, done, _ = env.step(a)
        ns = preprocess(np.array(ns)).type(
            torch.FloatTensor).to(device)
        s = ns

        if done:
            break
    else:
        env.stats_recorder.save_complete()
        env.stats_recorder.done = True


if __name__ == '__main__':

    args = parser.parse_args()
    model_name = args.name
    assert model_name in ['model1', 'model2']
    ckpt = args.ckpt
    max_steps = args.max_steps

    ckpt_path = './saved/' + model_name + '/' + ckpt + '.pt'
    video_dir = './agent/' + model_name + '/'

    device_cpu = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    else:
        device = device_cpu

    if model_name == 'model1':
        model = Model1().to(device)
    else:
        model = Model2().to(device)

    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            saved_dict = torch.load(f)

        model.load_state_dict(saved_dict['model'])
        model.eval()

    else:
        raise ValueError('Checkpoint does not exists')

    preprocess = transforms.Compose([
        ChannelFirst(),
        transforms.Resize((84, 84)),
        transforms.Grayscale(),
        FrameStack(4)
    ])

    print('Starting test run')

    play(model, max_steps)

    video_id = 0
    for f in os.listdir(video_dir):

        if f.startswith('test') and f.endswith('.mp4'):
            video_id = max(video_id, int(f.replace('.mp4', '').split('_')[1]))

    video_id += 1

    for f in os.listdir(video_dir):
        if f.startswith('open') and f.endswith('.mp4'):
            os.rename(video_dir + f, video_dir +
                      'test_' + str(video_id) + '.mp4')
        elif f.startswith('open') and f.endswith('.meta.json'):
            os.rename(video_dir + f, video_dir +
                      'test_' + str(video_id) + '.meta.json')

    print('Completed')
