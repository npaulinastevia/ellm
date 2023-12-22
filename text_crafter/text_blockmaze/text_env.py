import string

from text_crafter.text_blockmaze import constants
from wuji.problem.mdp.netease.blockmaze.maze import BaseMaze, Object, DeepMindColor as color, BaseEnv, \
    VonNeumannMotion
from skimage.draw import random_shapes
from transformers import AutoTokenizer
import numpy as np
from gym.spaces import Box, Discrete
def get_maze():
    size = (20, 20)
    max_shapes = 50
    min_shapes = max_shapes // 2
    max_size = 3
    seed = 2
    x, _ = random_shapes(size, max_shapes, min_shapes, max_size=max_size, multichannel=False, random_seed=seed)

    x[x == 255] = 0
    x[np.nonzero(x)] = 1

    # wall
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1

    return x


map = get_maze()
start_idx = [[10, 7]]
goal_idx = [[12, 12]]


class Maze(BaseMaze):
    @property
    def size(self):
        return map.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(map == 0), axis=1))
        obstacle = Object('obstacle', 85, color.obstacle, True, np.stack(np.where(map == 1), axis=1))
        agent = Object('agent', 170, color.agent, False, [])
        goal = Object('goal', 255, color.goal, False, [])
        return free, obstacle, agent, goal


class Env(BaseEnv):
    #def __init__(self):
     #   super().__init__()

    def __init__(self, action_space_type='easier', use_sbert=True, max_seq_len=100, **kwargs):
            super().__init__()
            self.action_space_type = action_space_type  # Easier or harder

            # Tokenizer to encode all strings
            self.use_sbert = use_sbert
            if use_sbert:
                self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L3-v2',
                                                               use_fast=True)

            # Obs configuration
            #view = self._view
            #item_rows = int(np.ceil(len(constants.items) / view[0]))
            #self._text_view = engine.EmbeddingView(self.world, [
             #   objects.Player, objects.Cow, objects.Zombie,
              #  objects.Skeleton, objects.Arrow, objects.Plant], [view[0], view[1] - item_rows])
            self._max_seq_len = max_seq_len
            self._vocab = self._get_action_vocab()

            self.maze = Maze()
            self.achievements = {name: 0 for name in constants.achievements}

            self.motions = VonNeumannMotion()
            self.visited=[]
            # self.bugs = [
            #     [1,1],[3,4],[7,5],[18,1],[11,12],[18,14],
            #     [12,6],[18,6],[11,14],[1,13],[3,13],[1,17],
            #     [2,18],[10,18],[17,18],[12,18],[15,17]
            # ]
            # self.bugs = np.logical_and(np.random.randint(0,2,[20,20]), np.logical_not(map))
            # self.bugs_cnt = np.count_nonzero(self.bugs)
            self.bug_idxs = [[0, 1], [3, 4], [1, 6], [7, 5], [6, 17], [5, 11], [7, 1], [0, 10], [16, 10], [18, 1], [4, 1],
                             [11, 12], [18, 14], [12, 6], [18, 6], [11, 14], [1, 13], [3, 13], [1, 17], [2, 18], [10, 18],
                             [15, 3], [17, 18], [12, 18], [15, 17]]
            self.bug_cnt = len(self.bug_idxs)

            #self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
            self.observation_space = Box(low=0, high=len(self.maze.objects), shape=(400,), dtype=np.uint8)
            self.action_space = Discrete(len(self.motions))
            self.actions_names=['move down','move left','move right','move up']

            self.context = dict(
                inputs=1,
                outputs=self.action_space.n
            )

    def step(self, action):
        motion = self.motions[action]

        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]

        # mark bug position
        bug = tuple(new_position) if new_position in self.bug_idxs else None

        # if bug is not None:
        #     print(bug)

        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        goal = self._is_goal(new_position)
        if bug is not None:
            if bug not in self.visited:
                self.achievements['detect_bug'] += 1
        if goal:
            self.achievements['reach_goal'] += 1
            reward = +10
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False
        info = {
            #'inventory': self.player.inventory.copy(),
            'achievements': self.achievements.copy(),
            #'discount': 1 - float(dead),
            #'semantic': self._sem_view(),
            'player_pos': new_position,
            'reward': reward ,
            #'health_reward': health_reward,
            'action_success': goal,
            #'eval_success': eval_success,
            'player_action': action,
            #'inventory': self.player.inventory.copy(),
            #'local_token': self._local_token_view.local_token_view(self.player),
            'large_obs': self.maze.to_value()
        }
        obs = {
            'obs': self.maze.to_value(),
            'text_obs' :  '',
            'inv_status': {},
            'success': info['action_success']
        }
        return self.tokenize_obs(obs), reward, done, info
        #return self.maze.to_value()[..., np.newaxis], reward, done, dict(bug=bug, valid=valid, goal=goal)

        #return self.maze.to_value().reshape(-1), reward, done, dict(bug=bug, valid=valid, goal=goal, current=current_position, render=self.maze.to_value()[..., np.newaxis])

    def reset(self):
        self.bug_item = set()
        self.maze.objects.agent.positions = start_idx
        self.maze.objects.goal.positions = goal_idx
        reward = 0
        dead = False
        obs = {
            'obs': self.maze.to_value(),
           'text_obs' :  "",
           'inv_status': {},
            'success': False
        }

        info = {
            'player_action': None,
            #'inventory': self.player.inventory.copy(),
            'achievements': self.achievements.copy(),
            #'discount': 1 - float(dead),
            #'semantic': self._sem_view(),
            #'local_token': self._local_token_view.local_token_view(self.player),
            'player_pos': start_idx[0],
            'reward': reward,
            'large_obs': self.maze.to_value()
        }
        return self.tokenize_obs(obs), info


        #return self.maze.to_value().reshape(-1)
    def tokenize_str(self, s):
        """Tokenize a string using the vocab index"""

        if self.use_sbert:  # Use SBERT tokenizer
            return np.array(self.tokenizer(s)['input_ids'])
        # Use the vocab index
        arr = np.zeros(self._max_seq_len, dtype=int)
        if " " in s:
            word_list = [w.strip(string.punctuation + ' ').lower() for w in s.split()]
            word_list = [w for w in word_list if len(w) > 0]
        else:
            word_list = [s.lower()]
        assert len(word_list) <= self._max_seq_len, f"word list length {len(word_list)} too long; increase max seq length: {self._max_seq_len}"

        for i, word in enumerate(word_list):
            if len(word) == 0:
                continue
            assert word in self._vocab, f"Invalid vocab word: |{word}|. {s}"
            arr[i] = self._vocab.index(word)
        return arr

    def pad_sbert(self, input_arr):
        """Pad array to max seq length"""
        arr = np.zeros(self._max_seq_len, dtype=int)
        if len(input_arr) > self._max_seq_len:
            input_arr = input_arr[:self._max_seq_len]
        arr[:len(input_arr)] = input_arr
        return arr
    def tokenize_obs(self, obs_dict):
        """
        Takes in obs dict and returns a dict where all strings are tokenized.
        """
        if self.use_sbert and isinstance(obs_dict['inv_status'], dict):
            inv_status = ""
            for k, v in obs_dict['inv_status'].items():
                if v != '.' and 'null' not in v:
                    inv_status += v + " "
            obs_dict['text_obs'] = obs_dict['text_obs'] + " " + inv_status

        new_obs = {}
        for k, v in obs_dict.items():
            # If the value is a dictionary of strings, concatenate them into a single string
            if isinstance(v, dict) and isinstance(list(v.values())[0], str):
                v = " ".join(v.values())
            # If the value is a string, tokenize it
            if isinstance(v, str):
                arr = self.tokenize_str(v)
                new_obs[k] = arr
            else:
                # Value is already tokenized (int, array, etc)
                new_obs[k] = v
        if self.use_sbert:
            new_obs['text_obs'] = self.pad_sbert(new_obs['text_obs'])
        return new_obs
    def _get_action_vocab(self):
        """Create a list of all possible vocab words."""
        # split string is the transformers library split token
        self.split_str = ' [SEP] '
        vocab = {self.split_str}
        vocab.update("you have in your inventory".split())
        vocab.update("you feel hurt hungry thirsty sleepy".split())
        vocab.update("you see".split())
        vocab.update("you are targeting".split())
        vocab.update('arrow player and'.split())
        #vocab.update(constants.materials)

        split_actions = [ac.split() for ac in constants.actions]
        split_actions = [item for sublist in split_actions for item in sublist]

        vocab.update(split_actions)
        #vocab.update(constants.walkable)
        #vocab.update(constants.items.keys())
        #vocab.update(constants.collect.keys())
        #vocab.update(constants.place.keys())
        #vocab.update(constants.make.keys())
        vocab.update(constants.achievements)

        vocab_list = ['null'] + sorted(list(vocab))
        return vocab_list

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out
    def get_visited_state(self, action):
        motion = self.motions[action]

        current_position = self.maze.objects.agent.positions[0]

        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]

        return new_position
    def get_image(self):
        return self.maze.to_rgb()