import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class Cos(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
    }

    x_dim = 1
    y_dim = 1
    u_dim = 1

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Ns: Optional[np.ndarray]=None,
        No: Optional[np.ndarray]=None,
        render_mode: str=None,
        horizon: int= 1000,
        periodic: Optional[bool]=True,
    ):
        
        super().__init__()

        self.A = A.astype(np.float32)
        self.B = B.astype(np.float32)
        self.Ns = Ns.astype(np.float32) if Ns is not None else None
        self.No = No.astype(np.float32) if No is not None else None

        self._verify_parameters()
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.horizon = horizon
        self.periodic = periodic

        self.state_space = spaces.Box(
            low=np.array([-np.pi]),
            high=np.array([np.pi]),
            shape=(1, ),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, ),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, ),
            dtype=np.float32,
        )

    def _verify_parameters(self):
        assert self.A.shape == (self.x_dim, self.x_dim)
        assert self.B.shape == (self.x_dim, self.u_dim)
        if self.Ns is not None:
            assert self.Ns.shape == (self.x_dim, self.x_dim)
        if self.No is not None:
            assert self.No.shape == (self.y_dim, self.y_dim)

    def manifold(self, s: np.ndarray):
        assert s.shape[1] == self.x_dim
        e = np.cos(s)
        return e

    def _get_obs(self):
        obs = self.manifold(self._state)
        if self.No is not None:
            no = self.np_random.multivariate_normal(
                mean=np.zeros(self.observation_space.shape),
                cov=self.No,
            ).astype(np.float32).reshape(1, -1)
            obs = obs + no
        return obs

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        
        super().reset(seed=seed)
        options = options or {}
        initial_state = options.get("initial_state")
        target_state = options.get("target_state")
        
        if initial_state is not None:
            assert initial_state.shape == self.state_space.shape
            self._state = initial_state.astype(np.float32).reshape(1, -1)
        else:
            self._state = self.state_space.sample().reshape(1, -1)

        if target_state is not None:
            assert target_state.shape == self.state_space.shape
            self._target = target_state.astype(np.float32).reshape(1, -1)
        else:
            self._target = self.state_space.sample().reshape(1, -1)

        self._step = 0
        observation = self._get_obs().flatten()
        info = {
            "state": self._state.copy().flatten(),
            "target": self._target.copy().flatten(),
        }

        return observation, info
    

    def step(
        self,
        action: np.ndarray,
    ):
        assert action.shape == self.action_space.shape
        action = action.astype(np.float32).reshape(1, -1)
        action = np.clip(
            action, 
            a_min=self.action_space.low,
            a_max=self.action_space.high,
        )

        self._state = self._state @ self.A.T + action @ self.B.T
        if self.Ns is not None:
            ns = self.np_random.multivariate_normal(
                mean=np.zeros(self.state_space.shape),
                cov=self.Ns,
            ).astype(np.float32).reshape(1, -1)
            self._state = self._state + ns

        self._step += 1
        truncated = bool(self._step >= self.horizon)
        terminated = False
        reward = 0.0
        
        if self.periodic:
            rng = self.state_space.high - self.state_space.low
            self._state = ((self._state - self.state_space.low) % rng) + self.state_space.low

        else:
            # Check if the state is valid
            is_valid = (
                np.all(self.state_space.low < self._state.flatten()) and np.all(self._state.flatten() < self.state_space.high)
            )
            if not is_valid:
                terminated = True

        obs = self._get_obs().flatten()
        info = {
            "state": self._state.copy().flatten(),
            "target": self._target.copy().flatten(),
        }

        return obs, reward, terminated, truncated, info 

            
    def render(self):
        if self.render_mode != "rgb_array":
            return None
    
        # Current latent x (x-position), current observed y (may be noisy)
        x = float(self._state.reshape(-1)[0])
        y_cur = float(self._get_obs().reshape(-1)[0])
    
        # Target latent x*, target noise-free y*
        xt = float(self._target.reshape(-1)[0])
        y_tgt = float(self.manifold(self._target).reshape(-1)[0])
    
        # Curve y = cos(x) using manifold()
        lo = float(self.state_space.low.reshape(-1)[0])
        hi = float(self.state_space.high.reshape(-1)[0])
        xs = np.linspace(lo, hi, 1200, dtype=np.float32).reshape(-1, 1)
        ys = self.manifold(xs).reshape(-1)
    
        fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
    
        # Manifold curve
        ax.plot(xs.reshape(-1), ys, linewidth=3, alpha=0.6, color="lightblue")
    
        # Points with labels
        ax.scatter([x],  [y_cur], s=90,  c="black", marker="o", label="current", zorder=5)
        ax.scatter([xt], [y_tgt], s=120, c="red",   marker="X", label="target",  zorder=6)
    
        # Cosmetics
        ax.set_xlim(lo, hi)
        ax.set_ylim(-1.15, 1.15)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", framealpha=0.9)
    
        fig.tight_layout(pad=0.3)
    
        # Convert to RGB array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)
        return img