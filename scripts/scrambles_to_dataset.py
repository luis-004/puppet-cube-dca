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
        states = env.generate_goal_states(num_states)
        num_back_steps = np.sum(scrambles != 255, axis=1).tolist()
        for move_idx in range(50):
            for action in np.unique(scrambles[:, move_idx]):
                if action == 255: continue
                indices = np.where(scrambles[:, move_idx] == action)[0]
                if len(indices) == 0: continue
                selected_states = [states[i] for i in indices]
                next_states, _ = env.next_state(selected_states, int(action))
                for i, next_state in zip(indices, next_states):
                    states[i] = next_state

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
    parser.add_argument('--data_file', type=str, required=True, help="File to save files")
    parser.add_argument('--scramble_file', type=str, required=True, help="File of scramble file to read from")

    args = parser.parse_args()

    env: Environment = env_utils.get_environment(args.env)

    if not os.path.exists(args.scramble_file):
        print("Scramble file not found")
        return
    
    scrambles_to_states(env, args.data_file, args.scramble_file)

if __name__ == "__main__":
    main()
