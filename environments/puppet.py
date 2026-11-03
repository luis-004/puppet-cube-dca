from typing import List, Dict, Tuple, Union
import numpy as np
from torch import nn
from random import randrange

from utils.pytorch_models import ResnetModel
from .environment_abstract import Environment, State


class PuppetState(State):
    __slots__ = ['pieces', 'hash']

    def __init__(self, pieces: np.ndarray):
        self.pieces: np.ndarray = pieces
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.pieces.tobytes())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.pieces, other.pieces)


class Puppet(Environment):
    moves: List[str] = ["%s%i" % (f, n) for f in ['R', 'U', 'F'] for n in [1,-1,2]]
    moves_rev: List[str] = ["%s%i" % (f, n) for f in ['R', 'U', 'F'] for n in [-1,1,2]]

    def __init__(self):
        super().__init__()
        self.dtype = np.uint8

        self.goal_pieces = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=self.dtype)

        self.rotate_corners: Dict[str, np.ndarray] = {
            "R1": np.array([17, 14, 8, 9, 4, 5, 6, 3, 0, 15, 16, 11, 12, 13, 10, 7, 1, 2, 18, 19, 20], dtype=self.dtype),
            "U1": np.array([12, 1, 2, 0, 10, 11, 6, 19, 8, 9, 7, 17, 18, 13, 5, 15, 16, 14, 3, 4, 20], dtype=self.dtype),
            "F1": np.array([1, 13, 2, 3, 4, 7, 12, 8, 20, 9, 10, 11, 14, 19, 15, 6, 16, 17, 18, 0, 5], dtype=self.dtype),
            "R-1": np.array([8, 16, 17, 7, 4, 5, 6, 15, 2, 3, 14, 11, 12, 13, 1, 9, 10, 0, 18, 19, 20], dtype=self.dtype),
            "U-1": np.array([3, 1, 2, 18, 19, 14, 6, 10, 8, 9, 4, 5, 0, 13, 17, 15, 16, 11, 12, 7, 20], dtype=self.dtype),
            "F-1": np.array([19, 0, 2, 3, 4, 20, 15, 5, 7, 9, 10, 11, 6, 1, 12, 14, 16, 17, 18, 13, 8], dtype=self.dtype),
            "R2": np.array([2, 10, 0, 15, 4, 5, 6, 9, 17, 7, 1, 11, 12, 13, 16, 3, 14, 8, 18, 19, 20], dtype=self.dtype),
            "U2": np.array([18, 1, 2, 12, 7, 17, 6, 4, 8, 9, 19, 14, 3, 13, 11, 15, 16, 5, 0, 10, 20], dtype=self.dtype),
            "F2": np.array([13, 19, 2, 3, 4, 8, 14, 20, 5, 9, 10, 11, 15, 0, 6, 12, 16, 17, 18, 1, 7], dtype=self.dtype),
        }

        self.rotate_edges: Dict[str, np.ndarray] = {
            "R1": np.array([0, 11, 7, 3, 4, 5, 1, 15, 8, 9, 2, 16, 12, 13, 14, 10, 6, 17], dtype=self.dtype),
            "U1": np.array([3, 1, 9, 17, 4, 5, 6, 7, 2, 12, 10, 0, 8, 13, 14, 15, 16, 11], dtype=self.dtype),
            "F1": np.array([10, 5, 2, 3, 0, 13, 6, 7, 8, 1, 14, 11, 12, 9, 4, 15, 16, 17], dtype=self.dtype),
            "R-1": np.array([0, 6, 10, 3, 4, 5, 16, 2, 8, 9, 15, 1, 12, 13, 14, 7, 11, 17], dtype=self.dtype),
            "U-1": np.array([11, 1, 8, 0, 4, 5, 6, 7, 12, 2, 10, 17, 9, 13, 14, 15, 16, 3], dtype=self.dtype),
            "F-1": np.array([4, 9, 2, 3, 14, 1, 6, 7, 8, 13, 0, 11, 12, 5, 10, 15, 16, 17], dtype=self.dtype),
            "R2": np.array([0, 16, 15, 3, 4, 5, 11, 10, 8, 9, 7, 6, 12, 13, 14, 2, 1, 17], dtype=self.dtype),
            "U2": np.array([17, 1, 12, 11, 4, 5, 6, 7, 9, 8, 10, 3, 2, 13, 14, 15, 16, 0], dtype=self.dtype),
            "F2": np.array([14, 13, 2, 3, 10, 9, 6, 7, 8, 5, 4, 11, 12, 1, 0, 15, 16, 17], dtype=self.dtype),
        }

        self.position = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6], dtype=self.dtype)
        self.orientation = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2], dtype=self.dtype)
        self.pow3 = np.array([0, 1, 3, 9, 27, 81, 243], dtype=self.dtype)

        self.possible_shapes = np.fromfile('shape_valid', dtype=np.uint8)
        self.shift = np.array([0, 0, 1, 0, 0, 2, 0, 0, 3], dtype=self.dtype)

        self.max_entry = 20 # highest value an entry can have
        self.num_corners = 7
        self.from6choose3 = self._compute_from6choose3()
        self.state_len = len(self.goal_pieces)

        self.moves_rev_ints = np.array([1, 0, 2, 4, 3, 5, 7, 6, 8], dtype=self.dtype)

    def next_state(self, states: List[PuppetState], action: int) -> Tuple[List[PuppetState], List[float]]:
        states_np = np.stack([x.pieces for x in states], axis=0)
        states_next_np, transition_costs = self._move_np(states_np, action)

        states_next: List[PuppetState] = [PuppetState(x) for x in list(states_next_np)]

        return states_next, transition_costs

    def prev_state(self, states: List[PuppetState], action: int) -> List[PuppetState]:
        move: str = self.moves[action]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[PuppetState], np.ndarray]:
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_pieces.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[PuppetState] = [PuppetState(self.goal_pieces.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[PuppetState]) -> np.ndarray:
        states_np = np.stack([state.pieces for state in states], axis=0)
        is_equal = np.equal(states_np, np.expand_dims(self.goal_pieces, 0))

        return np.all(is_equal, axis=1)

    def state_to_nnet_input(self, states: List[PuppetState]) -> List[np.ndarray]:
        states_np = np.stack([state.pieces for state in states], axis=0)

        representation_np: np.ndarray = states_np.astype(self.dtype)

        return [representation_np]

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        nnet = ResnetModel(self.state_len, self.max_entry + 1, 5000, 1000, 4, 1, True)

        return nnet

    # modified function which avoids consecutive moves on the same face
    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[PuppetState], List[int]]:
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        num_env_moves: int = self.get_num_moves()

        moves_face = np.arange(num_env_moves) // 3

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        states_np: np.ndarray = self.generate_goal_states(num_states, np_format=True)
        scramble_nums: np.ndarray = np.random.choice(scrambs, num_states)

        max_len: int = max(int(scramble_nums.max()), 1)
        history = np.full((num_states, max_len), -1, dtype=np.int64)
        blocked = np.zeros((num_states, max_len, num_env_moves), dtype=bool)
        depth = np.zeros(num_states, dtype=np.int64)

        active = depth < scramble_nums
        while np.any(active):
            idxs = np.where(active)[0]
            d = depth[idxs]

            last_move = history[idxs, np.maximum(d - 1, 0)]
            last_face = np.where(d > 0, moves_face[last_move], -1)

            valid = ~blocked[idxs, d] # (n, num_moves)
            valid &= moves_face[None, :] != last_face[:, None]
            has_valid = valid.any(axis=1)

            # dead end
            bt = idxs[~has_valid]
            if len(bt) > 0:
                m_prev = history[bt, depth[bt] - 1]
                for m in np.unique(m_prev):
                    sel = bt[m_prev == m]
                    states_np[sel], _ = self._move_np(states_np[sel], int(self.moves_rev_ints[m]))
                blocked[bt, depth[bt]] = False
                depth[bt] -= 1
                blocked[bt, depth[bt], m_prev] = True # block the move for layers below
                history[bt, depth[bt]] = -1

            # choose a random move
            fw = idxs[has_valid]
            if len(fw) > 0:
                valid_fw = valid[has_valid]
                scores = np.random.rand(*valid_fw.shape)
                scores[~valid_fw] = -1.0
                chosen = scores.argmax(axis=1)

                legal = np.zeros(len(fw), dtype=bool)
                for m in np.unique(chosen):
                    sel = chosen == m
                    sub = fw[sel]
                    old = states_np[sub].copy()
                    new, tc = self._move_np(old.copy(), int(m))
                    ok = np.array(tc) == 1.0
                    new[~ok] = old[~ok]
                    states_np[sub] = new
                    legal[sel] = ok

                good, bad = fw[legal], fw[~legal]
                blocked[bad, depth[bad], chosen[~legal]] = True
                history[good, depth[good]] = chosen[legal]
                depth[good] += 1

            active = depth < scramble_nums

        states: List[PuppetState] = [PuppetState(x) for x in list(states_np)]
        return states, scramble_nums.tolist()

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()

        states_exp: List[List[State]] = [[] for _ in range(len(states))]

        tc: np.ndarray = np.empty([num_states, num_env_moves])

        # numpy states
        states_np: np.ndarray = np.stack([state.pieces for state in states])

        # for each move, get next states, transition costs, and if solved
        move_idx: int

        for move_idx in range(num_env_moves):
            # next state
            states_next_np: np.ndarray
            tc_move: List[float]
            states_next_np, tc_move = self._move_np(states_np, move_idx)

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(PuppetState(states_next_np[idx]))

        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l
    
    def _move_np(self, states_np: np.ndarray, action: int):
        action_str: str = self.moves[action]

        states_next_np: np.ndarray = states_np.copy()

        # move the pieces first
        states_next_np[:, :self.num_corners] = self.rotate_corners[action_str][states_np[:, :self.num_corners]]
        states_next_np[:, self.num_corners:] = self.rotate_edges[action_str][states_np[:, self.num_corners:]]

        shape_keys = self._compute_shape_keys(states_next_np[:, :self.num_corners])

        # check if the resulting shapes are valid
        legal = ((self.possible_shapes[shape_keys] >> self.shift[action]) & 1).astype(bool)

        # only return valid states
        states_next_np[~legal] = states_np[~legal]

        # an invalid turn's cost is set to infinity
        transition_costs = np.where(legal, 1.0, np.inf).tolist()

        return states_next_np, transition_costs  
    
    def _compute_shape_keys(self, corners: np.ndarray):
        pos = self.position[corners]

        i = np.arange(self.num_corners, dtype=np.uint8)
        permutation = np.bitwise_or.reduce((i & 1) << pos, axis=1)

        orientation_key = np.sum(self.orientation[corners].astype(np.uint16) * self.pow3[pos].astype(np.uint16), axis=1, dtype=np.uint16)

        return (np.uint32(orientation_key) * np.uint32(140)
            + np.uint32(self.from6choose3[self.position[corners[:, 0]], permutation]))

    def _compute_from6choose3(self):
        table = np.full((self.num_corners, 128), 255, dtype=np.uint8)
        idx = 0
        for i in range(self.num_corners):
            for j in range(i + 1, self.num_corners):
                for k in range(j + 1, self.num_corners):
                    permutation = (1 << i) | (1 << j) | (1 << k)
                    for zeroCor in range(self.num_corners):
                        if zeroCor != i and zeroCor != j and zeroCor != k:
                            table[zeroCor, permutation] = idx
                            idx += 1
        return table
