import numpy as np
import gymnasium as gym
from gymnasium import spaces

N_CHILDREN = 5      # youth in the CHW's caseload
MAX_WEEKS = 20      # one episode = 20 weeks
MAX_REFERRALS = 2   # health centre can treat only 2 youth at a time
# hidden risk levels: 0 = well, 1 = mild, 2 = moderate, 3 = crisis


class CHWEnv(gym.Env):
    """A community health worker (CHW) in Kigali supports 5 young people over 20 weeks.
    Each week she can do ONE action:
      0-4  : home visit + screening of youth i (reveals their real risk level)
      5-9  : refer youth i to the health centre
      10   : group session at school (small help for everyone with mild distress)
    """

    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(2 * N_CHILDREN + 1)
        # per youth: last screening score, weeks since last visit, referred flag
        # plus: current week. Everything scaled to 0..1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3 * N_CHILDREN + 1,), dtype=np.float32)
        self.render_mode = render_mode
        self.renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.risk = self.np_random.choice([0, 0, 1, 1, 2], size=N_CHILDREN)  # hidden true state
        self.last_score = np.zeros(N_CHILDREN)        # what the CHW knows
        self.weeks_since = np.full(N_CHILDREN, 10.0)  # nobody screened yet
        self.referred = np.zeros(N_CHILDREN)
        self.week = 0
        self.last_action = None
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.concatenate([
            self.last_score / 3.0,
            np.minimum(self.weeks_since, 10) / 10.0,
            self.referred,
            [self.week / MAX_WEEKS],
        ])
        return obs.astype(np.float32)

    def step(self, action):
        action = int(action)
        self.last_action = action
        reward = -0.1  # small cost for the CHW's time

        if action < N_CHILDREN:  # visit + screen
            i = action
            if self.risk[i] >= 2 and self.last_score[i] < 2:
                reward += 1.0  # found a youth at risk
            self.last_score[i] = self.risk[i]
            self.weeks_since[i] = 0
            if self.risk[i] == 1 and self.np_random.random() < 0.3:
                self.risk[i] = 0  # counselling during the visit helps mild cases

        elif action < 2 * N_CHILDREN:  # refer
            i = action - N_CHILDREN
            if self.referred[i] == 1:
                reward -= 1.0  # already referred, wasted action
            elif self.referred.sum() >= MAX_REFERRALS:
                reward -= 1.0  # health centre is full
            elif self.risk[i] >= 2:
                self.referred[i] = 1
                reward += 5.0  # correct referral
            else:
                reward -= 2.0  # unnecessary referral wastes clinic resources

        else:  # group session
            for i in range(N_CHILDREN):
                if self.risk[i] == 1 and self.np_random.random() < 0.2:
                    self.risk[i] = 0

        # --- the world moves on one week ---
        for i in range(N_CHILDREN):
            if self.referred[i] == 1:
                if self.np_random.random() < 0.5:
                    self.risk[i] = max(self.risk[i] - 1, 0)
                if self.risk[i] == 0:
                    self.referred[i] = 0  # treatment finished, clinic reports back
                    self.last_score[i] = 0
            elif self.np_random.random() < 0.1:
                self.risk[i] = min(self.risk[i] + 1, 3)  # distress can grow if unnoticed
            self.weeks_since[i] += 1

        crisis = np.sum((self.risk == 3) & (self.referred == 0))
        reward -= 1.0 * crisis  # every youth in crisis without care is penalised
        self.week += 1

        terminated = bool(crisis >= 3)  # caseload out of control
        if terminated:
            reward -= 10.0
        truncated = self.week >= MAX_WEEKS

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, {"crisis": int(crisis)}

    def render(self):
        if self.renderer is None:
            from environment.rendering import Renderer
            self.renderer = Renderer()
        self.renderer.draw(self)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()