from typing import List
from environments.environment_abstract import Environment, State
from utils import env_utils
from argparse import ArgumentParser
import pickle
import os
import time
import numpy as np

from multiprocessing import Queue, Process


def scrambles_to_states(env: Environment,  datapath: str, scramblepath: str):
        scrambles = np.fromfile(scramblepath, dtype=np.uint8)
        scrambles = scrambles.reshape(-1, 50)
        num_states = scrambles.shape[0]
        start_time = time.time()
        print("Generating data for %s" % datapath)
        states = env.generate_goal_states(num_states,np_format=True)
        num_back_steps = np.sum(scrambles != 255 ,axis=1).tolist()
        for i in range(50):
            moves = scrambles[:, i]
            active = moves != 255
            if not np.any(active):
                break
            for move in np.unique(moves[active]):
                mask = moves == move
                states[mask], _ = env.move_np(
                    states[mask],
                    int(move)
                )

        data_gen_time = time.time() - start_time

        # save data
        start_time = time.time()

        data = dict()
        data['states'] = states
        data['num_back_steps'] = num_back_steps

        pickle.dump(data, open(datapath, "wb"), protocol=-1)

        save_time = time.time() - start_time

        print("%s - Data Gen Time: %s, Save Time: %s" % (datapath, data_gen_time, save_time))


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="Environment")
    parser.add_argument('--data_file', type=str, required=True, help="Directory to save files")
    parser.add_argument('--scramble_file', type=str, required=True, help="Directory of scramble file to read from")

    args = parser.parse_args()

    env: Environment = env_utils.get_environment(args.env)

    if not os.path.exists(args.data_file):
        os.makedirs(args.data_file)
    if not os.path.exists(args.scramble_file):
        print("Scramble file not found")
        return
    
    scrambles_to_states(env, args.data_file, args.scramble_file)

if __name__ == "__main__":
    main()
