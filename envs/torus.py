import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from scipy.stats import ortho_group


class Torus(gym.Env):
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
        radius1: float = 1.0,
        radius2: float = 4.0,
        rotated: bool = False,
        Ns: Optional[np.ndarray] = None,
        No: Optional[np.ndarray] = None,
        render_mode: str = None,
        horizon: int = 1000,
        periodic: Optional[bool] = True,
    ):

        super().__init__()

        self.A = A.astype(np.float32)
        self.B = B.astype(np.float32)
        self.radius1 = radius1
        self.radius2 = radius2
        self.Q_rot = ortho_group.rvs(3).astype(np.float32) if rotated else None
        self.Ns = Ns.astype(np.float32) if Ns is not None else None
        self.No = No.astype(np.float32) if No is not None else None

        self._verify_parameters()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.horizon = horizon
        self.periodic = periodic

        self.state_space = spaces.Box(
            low=np.array([-np.pi, -np.pi]),
            high=np.array([np.pi, np.pi]),
            shape=(2,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
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
        x = (self.radius2 + self.radius1 * np.sin(s[:, 0:1])) * np.cos(s[:, 1:2])
        y = (self.radius2 + self.radius1 * np.sin(s[:, 0:1])) * np.sin(s[:, 1:2])
        z = self.radius1 * np.cos(s[:, 0:1])
        e = np.hstack([x, y, z])
        if self.Q_rot is not None:
            e = e @ self.Q_rot.T
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

        fig = plt.figure(figsize=(7.0, 7.0), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        lo = self.state_space.low.astype(np.float32)
        hi = self.state_space.high.astype(np.float32)

        n_th, n_ph = 180, 220
        theta = np.linspace(lo[0], hi[0], n_th, endpoint=False, dtype=np.float32)
        phi   = np.linspace(lo[1], hi[1], n_ph, endpoint=False, dtype=np.float32)
        TH, PH = np.meshgrid(theta, phi)

        s_samples = np.column_stack([TH.ravel(), PH.ravel()]).astype(np.float32)
        M = self.manifold(s_samples)

        X = M[:, 0].reshape(TH.shape)
        Y = M[:, 1].reshape(TH.shape)
        Z = M[:, 2].reshape(TH.shape)

        ax.plot_surface(
            X, Y, Z,
            color="lightsteelblue",
            alpha=0.20,
            shade=False,
            rstride=1, cstride=1,
            linewidth=0.25,
            edgecolor=(0.10, 0.25, 0.70, 0.12),
            antialiased=True,
        )

        obs_cur = self._get_obs().reshape(-1)
        obs_tgt = self.manifold(self._target).reshape(-1)

        ax.scatter(obs_cur[0], obs_cur[1], obs_cur[2],
                   s=140, c="black", marker="o", label="current", depthshade=False)
        ax.scatter(obs_tgt[0], obs_tgt[1], obs_tgt[2],
                   s=180, c="red", marker="X", label="target", depthshade=False)

        xmin, xmax = float(X.min()), float(X.max())
        ymin, ymax = float(Y.min()), float(Y.max())
        zmin, zmax = float(Z.min()), float(Z.max())

        xmid, ymid, zmid = (xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
        half = 0.55 * max_range

        ax.set_xlim(xmid - half, xmid + half)
        ax.set_ylim(ymid - half, ymid + half)
        ax.set_zlim(zmid - half, zmid + half)
        ax.set_box_aspect((1, 1, 1))

        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        ax.grid(True, alpha=0.15)

        ax.set_xlabel("y[0]")
        ax.set_ylabel("y[1]")
        ax.set_zlabel("y[2]")
        ax.view_init(elev=28, azim=35)
        ax.legend(loc="upper left", framealpha=0.9)

        fig.tight_layout(pad=0.2)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)
        return img