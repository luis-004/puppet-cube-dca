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
    moves: List[Tuple[str, np.uint8]] = [(f, n) for f in ['R', 'U', 'F'] for n in [1,-1,2]]
    moves_rev: List[Tuple[str, np.uint8]] = [(f, n) for f in ['R', 'U', 'F'] for n in [-1,1,2]]

    def __init__(self):
        super().__init__()
        self.dtype = np.uint8

        """
		state: 
		(from upper to lower layers, starting UFR, going left, back, down; 0 : constant(always the same piece/ not reachable / trivially redundant))
		(maybe, other cubies are redundant too)
		{
		c, c, c, c, c,  c, c, c, c, c,  c, c, c, c, c,  c, c, c, c, c,  c, c, c, c, c,		
		c, c, c, c, c,  c, C, e, C, c,  c, e, 0, e, c,  c, C, e, C, c,  c, c, c, c, c,
		c, c, c, c, c,  c, e, 0, e, c,  c, 0, 0, 0, c,  c, e, 0, 0, c,  c, c, c, c, c,
		c, c, c, c, c,  c, C, e, C, c,  c, e, 0, 0, c,  c, C, 0, C, 0,  c, c, c, 0, 0,
		c, c, c, c, c,  c, c, c, c, c,  c, c, c, c, c,  c, c, c, 0, 0,  c, c, c, 0, 0,
		}
		c: corner cubie [1,7]; e: edge piece [1,9], + 10 if flipped; 0: empty
        {
            0,   1,   2,   3,   4,    5,   6,   7,   8,   9,    10,  11,  12,  13,  14,   15,  16,  17,  18,  19,   20,  21,  22,  23,  24,
            25,  26,  27,  28,  29,   30,  31,  32,  33,  34,   35,  36,  37,  38,  39,   40,  41,  42,  43,  44,   45,  46,  47,  48,  49,
            50,  51,  52,  53,  54,   55,  56,  57,  58,  59,   60,  61,  62,  63,  64,   65,  66,  67,  68,  69,   70,  71,  72,  73,  74,
            75,  76,  77,  78,  79,   80,  81,  82,  83,  84,   85,  86,  87,  88,  89,   90,  91,  92,  93,  94,   95,  96,  97,  98,  99,
            100, 101, 102, 103, 104,  105, 106, 107, 108, 109,  110, 111, 112, 113, 114,  115, 116, 117, 118, 119,  120, 121, 122, 123, 124
        }
		"""
        # solved state
        '''self.goal_pieces: np.ndarray = np.array([
            7, 7, 6, 6, 0,  7, 7, 6, 6, 0,      3, 3, 2, 2, 0,        3, 3, 2, 2, 0,          0, 0, 0, 0, 0,
            7, 7, 6, 6, 0,  7, 7, 9, 6, 0,	    3, 8, 255, 12, 0,	  3, 3, 14, 2, 0,         0, 0, 0, 0, 0,
            5, 5, 4, 4, 0,  5, 10, 255, 15, 0,  1, 255, 255, 255, 0,  1, 11, 255, 255, 0,     0, 0, 0, 0, 0,
            5, 5, 4, 4, 0,  5, 5, 13, 4, 0,     1, 16, 255, 255, 0,   1, 1, 255, 255, 0,      0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,  0, 0, 0, 0, 0,      0, 0, 0, 0, 0,        0, 0, 0, 0, 0,          0, 0, 0, 0, 0
        ], dtype=self.dtype)
        7 7 6 6 0
        7 7 6 6 0
        3 3 2 2 0
        3 3 2 2 0
        0 0 0 0 0

        7 7 6 6 0
        7 7 2 6 0
        3 1 0 5 0
        3 3 7 2 0
        0 0 0 0 0

        5 5 4 4 0
        5 3 0 8 0
        1 0 0 0 0
        1 4 0 0 0
        0 0 0 0 0

        5 5 4 4 0
        5 5 6 4 0
        1 9 0 0 0
        1 1 0 0 0
        0 0 0 0 0

        0 0 0 0 0
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 0 0

        '''

        self.goal_pieces: np.ndarray = np.array([
            7, 7, 6, 6, 0,      7, 7, 6, 6, 0,      3, 3, 2, 2, 0,      3, 3, 2, 2, 0,      0, 0, 0, 0, 0,
            7, 7, 6, 6, 0,      7, 7, 2, 6, 0,	    3, 1, 0, 5, 0,      3, 3, 7, 2, 0,      0, 0, 0, 0, 0,
            5, 5, 4, 4, 0,      5, 3, 0, 8, 0,      1, 0, 0, 0, 0,      1, 4, 0, 0, 0,      0, 0, 0, 0, 0,
            5, 5, 4, 4, 0,      5, 5, 6, 4, 0,      1, 9, 0, 0, 0,      1, 1, 0, 0, 0,      0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,      0, 0, 0, 0, 0,      0, 0, 0, 0, 0,      0, 0, 0, 0, 0,      0, 0, 0, 0, 0
        ], dtype=self.dtype)

        self.rotate_idxs: Dict[str, np.ndarray] = {
            "R": np.array([
                [31, 41, 91, 81],    [56, 36, 66, 86],
                [30, 40, 90, 80],    [35, 65, 85, 55],   
                [0, 20, 120, 100],   [5, 45, 115, 75],   [10, 70, 110, 50],  [15, 95, 105, 25],
                [1, 21, 121, 101],   [6, 46, 116, 76],   [11, 71, 111, 51],  [16, 96, 106, 26],
                [2, 22, 122, 102],   [7, 47, 117, 77],   [12, 72, 112, 52],  [17, 97, 107, 27]
            ], dtype=self.dtype),
            "U": np.array([
                [31, 33, 43, 41],   [32, 38, 42, 36],
                [6,  8,  18, 16],   [7,  13, 17, 11],   
                [0,  4,  24, 20],   [1,  9,  23, 15],   [2,  14, 22, 10],  [3,  19, 21,  5],
                [25, 29, 49, 45],   [26, 34, 48, 40],   [27, 39, 47, 35],  [28, 44, 46, 30],
                [50, 54, 74, 70],   [51, 59, 73, 65],   [52, 64, 72, 60],  [53, 69, 71, 55]
            ], dtype=self.dtype),
            "F": np.array([
                [31, 81, 83, 33],    [58, 32,  56,  82],
                [26, 76,  78,  28],  [51, 77,  53,  27],
                [0,  100, 104, 4],   [25, 101, 79,  3],   [50, 102, 54,  2],  [75, 103, 29,  1],
                [5,  105, 109, 9],   [30, 106, 84,  8],   [55, 107, 59,  7],  [80, 108, 34,  6],
                [10, 110, 114, 14],  [35, 111, 89,  13],  [60, 112, 64,  12], [85, 113, 39,  11]
            ], dtype=self.dtype),
        }
        self.blocking_idx= 8
        self.corner_idx = 0
        self.edge_idx = 1
        #self.edge_summand = 7
        self.max_entry = 19
        self.flip_summand = 10
        self.state_len = len(self.goal_pieces)

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

        representation_np: np.ndarray = (states_np / self.max_entry).astype(np.float32)

        return [representation_np]

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        nnet = ResnetModel(self.state_len, self.max_entry, 5000, 1000, 4, 1, True)

        return nnet

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[PuppetState], List[int]]:
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()

        # Get goal states
        states_np: np.ndarray = self.generate_goal_states(num_states, np_format=True)

        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)

        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        moves_lt = num_back_moves < scramble_nums

        while np.any(moves_lt):
            idxs: np.ndarray = np.where(moves_lt)[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)

            states_np[idxs], tc_move = self._move_np(states_np[idxs], move)

            legal_mask = np.array(tc_move) == 1.0
            num_back_moves[idxs[legal_mask]] += 1

            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

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
        move: int
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
    

    @staticmethod
    def _in_moving_corners(pieces, moving_corners):
        return np.any(pieces[..., None] == moving_corners[:, None, None, :],axis=-1)
    
    def _move_np(self, states_np: np.ndarray, action: int):
        face, rotation = self.moves[action]
        idxs: np.ndarray = self.rotate_idxs[face]

        #turning pieces which never block each other
        states_next_np: np.ndarray = states_np.copy()
        states_next_np[:, idxs[:self.blocking_idx]] = states_np[:, np.roll(idxs[:self.blocking_idx], rotation, axis=1)]

        # flipping edges on F move
        if face == 'F' and rotation != 2:
            states_next_np[:, idxs[self.edge_idx].flatten()] += (np.sign(self.flip_summand - states_next_np[:, idxs[self.edge_idx].flatten()].astype(int))).astype(self.dtype) * (self.flip_summand)

        # rotating critical pieces while looking for blockades
        piece_idxs = idxs[self.blocking_idx:]
        pieces = states_np[:, piece_idxs]
        new_pieces = states_np[:, np.roll(piece_idxs, rotation, axis=1)]
        moving_corners = states_np[:, idxs[self.corner_idx]]
        is_moving = self._in_moving_corners(new_pieces, moving_corners)
        target_is_free = self._in_moving_corners(pieces, moving_corners) | (pieces == 0)
        pieces = states_next_np[:, piece_idxs]
        pieces[target_is_free] = 0
        pieces[is_moving] = new_pieces[is_moving]
        states_next_np[:, piece_idxs] = pieces
        any_illegal = np.any(is_moving & ~target_is_free, axis=(1, 2))
        
        # handling double moves 
        if rotation == 2:
            any_illegal |= self._move_helper(states_np, face, 1) & self._move_helper(states_np, face, -1)

        transition_costs = np.where(any_illegal, 255.0, 1.0).tolist()
        states_next_np[any_illegal] = states_np[any_illegal]

        return states_next_np, transition_costs  
    
    def _move_helper(self, states_np: np.ndarray, face, rotation):
        idxs: np.ndarray = self.rotate_idxs[face]
        piece_idxs = idxs[self.blocking_idx:]
        pieces = states_np[:, piece_idxs]
        new_pieces = states_np[:, np.roll(piece_idxs, rotation, axis=1)]
        moving_corners = states_np[:, idxs[self.corner_idx]]
        is_moving = self._in_moving_corners(new_pieces, moving_corners)
        target_is_free = self._in_moving_corners(pieces, moving_corners) | (pieces == 0)
        return np.any(is_moving & ~target_is_free, axis=(1, 2))