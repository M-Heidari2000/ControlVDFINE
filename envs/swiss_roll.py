import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class SwissRoll(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
    }

    x_dim = 2
    y_dim = 3
    u_dim = 2

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
            low=np.array([-2*np.pi, -2*np.pi]),
            high=np.array([2*np.pi, 2*np.pi]),
            shape=(2, ),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2, ),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, ),
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
        s = s.copy()
        s[:, 0] = s[:, 0] + 2 * np.pi
        x = s[:, 0] * np.cos(s[:, 0]) / 2
        y = s[:, 1]
        z = s[:, 0] * np.sin(s[:, 0]) / 2
        e = np.stack([x, y, z], axis=1)
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
    
        fig = plt.figure(figsize=(7.2, 6.2), dpi=150)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
    
        # --- Sample the latent grid and map via manifold() ---
        lo = self.state_space.low.astype(np.float32)
        hi = self.state_space.high.astype(np.float32)
        
        s0 = np.linspace(lo[0], hi[0], 220, dtype=np.float32)   # roll parameter
        s1 = np.linspace(lo[1], hi[1], 90,  dtype=np.float32)   # "height" parameter
        S0, S1 = np.meshgrid(s0, s1)
        
        s_samples = np.column_stack([S0.ravel(), S1.ravel()]).astype(np.float32)
        M = self.manifold(s_samples)
        
        X = M[:, 0].reshape(S0.shape)
        Y = M[:, 1].reshape(S0.shape)
        Z = M[:, 2].reshape(S0.shape)
        
        # --- Surface: uniform light-blue + faint mesh lines (like your screenshot) ---
        ax.plot_surface(
            X, Y, Z,
            color="lightsteelblue",      # uniform color
            alpha=0.22,                  # translucent
            shade=False,                 # important: avoids dark shading
            rstride=1, cstride=1,        # lets edges show as a faint grid
            linewidth=0.25,
            edgecolor=(0.20, 0.35, 0.80, 0.15),  # RGBA: faint bluish grid lines
            antialiased=True,
        )
        
        # --- Darker boundary edges (the “outline” in your screenshot) ---
        # edges at s1 = low/high (two long edges running along the roll)
        edge_lo = self.manifold(np.stack([s0, np.full_like(s0, lo[1])], axis=1).astype(np.float32))
        edge_hi = self.manifold(np.stack([s0, np.full_like(s0, hi[1])], axis=1).astype(np.float32))
        ax.plot(edge_lo[:, 0], edge_lo[:, 1], edge_lo[:, 2], color=(0.10, 0.25, 0.70, 0.9), linewidth=2.2)
        ax.plot(edge_hi[:, 0], edge_hi[:, 1], edge_hi[:, 2], color=(0.10, 0.25, 0.70, 0.9), linewidth=2.2)
        
        # (optional) edges at s0 = low/high (the “end caps”)
        cap_lo = self.manifold(np.stack([np.full_like(s1, lo[0]), s1], axis=1).astype(np.float32))
        cap_hi = self.manifold(np.stack([np.full_like(s1, hi[0]), s1], axis=1).astype(np.float32))
        ax.plot(cap_lo[:, 0], cap_lo[:, 1], cap_lo[:, 2], color=(0.10, 0.25, 0.70, 0.65), linewidth=1.8)
        ax.plot(cap_hi[:, 0], cap_hi[:, 1], cap_hi[:, 2], color=(0.10, 0.25, 0.70, 0.65), linewidth=1.8)

        # --- Points (labels only via legend) ---
        obs_cur = self._get_obs().reshape(-1)             # current (observed, maybe noisy)
        obs_tgt = self.manifold(self._target).reshape(-1) # target (noise-free)
    
        ax.scatter(obs_cur[0], obs_cur[1], obs_cur[2],
                   s=110, c="black", marker="o",
                   label="current", depthshade=False)
    
        ax.scatter(obs_tgt[0], obs_tgt[1], obs_tgt[2],
                   s=150, c="red", marker="X",
                   label="target", depthshade=False)
    
        # --- Limits ---
        pad = 0.08
        xmin, xmax = float(X.min()), float(X.max())
        ymin, ymax = float(Y.min()), float(Y.max())
        zmin, zmax = float(Z.min()), float(Z.max())
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    
        ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
        ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
        ax.set_zlim(zmin - pad * dz, zmax + pad * dz)
    
        ax.set_xlabel("y[0]")
        ax.set_ylabel("y[1]")
        ax.set_zlabel("y[2]")
        ax.grid(True, alpha=0.2)
    
        ax.view_init(elev=30, azim=50)
    
        ax.legend(loc="upper left", framealpha=0.9)
        fig.tight_layout(pad=0.3)
    
        # --- Convert to RGB array ---
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)
        return img
