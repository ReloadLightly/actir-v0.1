from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Tuple
import numpy as np


class Role(str, Enum):
    XGP = "extra_regional_great_power"
    RGP = "regional_great_power"
    MP = "middle_power"


class Action(str, Enum):
    ARM = "arm"
    HOLD = "hold"
    POSTURE_UP = "posture_up"
    POSTURE_DOWN = "posture_down"
    SIGNAL_HAWK = "signal_hawk"
    SIGNAL_DOVE = "signal_dove"
    DEESCALATE = "deescalate"


@dataclass
class AgentParams:
    role: Role
    capability: float
    projection: float
    intel_quality: float
    risk_aversion: float


@dataclass
class ActirConfig:
    # Cast
    n_agents: int = 6          # XGP, RGP, 4 MPs
    n_middle_powers: int = 4

    # Episode
    horizon: int = 50
    seed: int = 0

    # Assumption switches (to be wired in later)
    obs_noise: float = 0.25
    maritime_friction_multiplier: float = 1.2
    offense_defense_balance: str = "balanced"  # defense_dominant | balanced | offense_dominant
    commitment_reliability: float = 0.8        # 0..1


class ActirV01:
    """
    ACTIR v0.1: Indo-Pacific regional dynamics prototype

    - Agents: 1 XGP, 1 RGP, rest MPs
    - Partial observability: noisy signals (placeholder)
    - Dynamics: simple crisis escalation (placeholder)
    - Next upgrade: dyadic latent intent + spiral vs deterrence inference (Jervis core)
    """

    def __init__(self, cfg: ActirConfig):
        assert cfg.n_agents == cfg.n_middle_powers + 2
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.agent_ids = [f"a{i}" for i in range(cfg.n_agents)]
        self.roles = self._assign_roles(cfg.n_middle_powers)
        self.params = self._init_params()

        self.t = 0
        self.crisis_level = 0.0

        # Geography/friction (distance/water proxy)
        self.friction = self._init_friction()

        # Beliefs (placeholder for upcoming Jervis mechanism)
        self.belief_intent: Dict[Tuple[str, str], float] = {}
        self._reset_beliefs()

    def _assign_roles(self, n_mps: int) -> Dict[str, Role]:
        roles: Dict[str, Role] = {}
        roles[self.agent_ids[0]] = Role.XGP
        roles[self.agent_ids[1]] = Role.RGP
        for k in range(n_mps):
            roles[self.agent_ids[2 + k]] = Role.MP
        return roles

    def _init_params(self) -> Dict[str, AgentParams]:
        p: Dict[str, AgentParams] = {}
        for aid in self.agent_ids:
            r = self.roles[aid]
            if r == Role.XGP:
                p[aid] = AgentParams(r, capability=1.2, projection=1.1, intel_quality=0.9, risk_aversion=0.8)
            elif r == Role.RGP:
                p[aid] = AgentParams(r, capability=1.1, projection=0.9, intel_quality=0.8, risk_aversion=0.7)
            else:
                p[aid] = AgentParams(r, capability=0.7, projection=0.6, intel_quality=0.7, risk_aversion=0.95)
        return p

    def _init_friction(self) -> Dict[Tuple[str, str], float]:
        fr: Dict[Tuple[str, str], float] = {}
        for i in self.agent_ids:
            for j in self.agent_ids:
                if i == j:
                    fr[(i, j)] = 0.0
                else:
                    base = float(self.rng.uniform(0.6, 1.4))
                    if self.roles[i] == Role.XGP or self.roles[j] == Role.XGP:
                        base *= self.cfg.maritime_friction_multiplier
                    fr[(i, j)] = base
        return fr

    def _reset_beliefs(self) -> None:
        self.belief_intent.clear()
        for i in self.agent_ids:
            for j in self.agent_ids:
                if i == j:
                    continue
                self.belief_intent[(i, j)] = float(np.clip(self.rng.normal(0.5, 0.1), 0.05, 0.95))

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        self.crisis_level = 0.0
        self._reset_beliefs()
        return self._obs()

    def step(self, actions: Dict[str, Action]) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        self.t += 1

        hawk = sum(1 for a in actions.values() if a in (Action.ARM, Action.POSTURE_UP, Action.SIGNAL_HAWK))
        dove = sum(1 for a in actions.values() if a in (Action.POSTURE_DOWN, Action.SIGNAL_DOVE, Action.DEESCALATE))

        # Placeholder dynamics (we’ll replace with real SD feedback loop next)
        self.crisis_level = float(np.clip(self.crisis_level + 0.08 * hawk - 0.10 * dove, 0.0, 1.0))
        war = self.crisis_level > 0.95

        rewards: Dict[str, float] = {}
        for aid in self.agent_ids:
            rewards[aid] = 1.0 - 2.0 * self.crisis_level
            if war:
                rewards[aid] -= 50.0

        done = war or (self.t >= self.cfg.horizon)
        info = {"t": self.t, "crisis_level": self.crisis_level, "war": war}
        return self._obs(), rewards, done, info

    def _obs(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {}
        for i in self.agent_ids:
            base_std = self.cfg.obs_noise * (1.0 - self.params[i].intel_quality)
            signal = float(np.clip(self.crisis_level + self.rng.normal(0.0, base_std), 0.0, 1.0))
            obs[i] = {"t": self.t, "role": self.roles[i].value, "crisis_signal": signal}
        return obs


def demo() -> None:
    env = ActirV01(ActirConfig(seed=42))
    obs = env.reset()
    print("RESET:", obs)

    for _ in range(6):
        actions = {aid: Action.HOLD for aid in env.agent_ids}
        actions[env.agent_ids[1]] = Action.SIGNAL_HAWK   # RGP hawkish
        actions[env.agent_ids[2]] = Action.DEESCALATE    # MP tries to calm

        obs, rew, done, info = env.step(actions)
        print(info, {k: round(v, 2) for k, v in rew.items()})
        if done:
            break


if __name__ == "__main__":
    demo()

