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


class IntentType(str, Enum):
    BENIGN = "benign"   # security-seeking / defensive realist
    MALIGN = "malign"   # greedy / revisionist


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

    # Belief dynamics (transparent + testable; refine later with page-anchored claims)
    belief_lr: float = 0.8                    # update step size in log-odds space
    deterrence_threshold: float = 0.65        # if P(malign) >= this => deterrence
    spiral_threshold: float = 0.35            # if P(malign) <= this AND fear high => spiral risk

    # Fear (spiral-risk) dynamics
    fear_decay: float = 0.7                   # higher => fear persists longer
    fear_threshold: float = 0.6               # fear must exceed this for spiral mode


class ActirV01:
    """
    ACTIR v0.1: Indo-Pacific regional dynamics prototype

    - Agents: 1 XGP, 1 RGP, rest MPs
    - Partial observability: noisy signals
    - Dynamics: crisis escalation placeholder, now modulated by dyadic fear
    - New: dyadic latent intent + belief updates + spiral/deterrence/uncertain "mode"
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

        # Beliefs: b[i,j] = P(intent[j] = MALIGN | history_i)
        self.belief_intent: Dict[Tuple[str, str], float] = {}
        self._reset_beliefs()

        # Latent "true" intent type per actor (environment hidden variable)
        self.true_intent: Dict[str, IntentType] = {}

        # Fear: dyadic perceived threat accumulator (i feels threatened by j)
        self.fear: Dict[Tuple[str, str], float] = {}

        self._reset_latents()

    # ------------------------
    # Initialization helpers
    # ------------------------
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

    def _reset_latents(self) -> None:
        # Prior over malign intent by role (placeholder; later we’ll parameterize/anchor)
        self.true_intent = {}
        for aid in self.agent_ids:
            role = self.roles[aid]
            if role == Role.RGP:
                p_malign = 0.35
            elif role == Role.XGP:
                p_malign = 0.25
            else:
                p_malign = 0.20
            self.true_intent[aid] = IntentType.MALIGN if self.rng.random() < p_malign else IntentType.BENIGN

        self.fear = {}
        for i in self.agent_ids:
            for j in self.agent_ids:
                if i != j:
                    self.fear[(i, j)] = 0.0

    # ------------------------
    # Math helpers
    # ------------------------
    @staticmethod
    def _logit(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def _hawk_score(a: Action) -> float:
        """Simple evidence model: hawkish acts => +1, dovish => -1, hold => 0."""
        if a in (Action.ARM, Action.POSTURE_UP, Action.SIGNAL_HAWK):
            return 1.0
        if a in (Action.POSTURE_DOWN, Action.SIGNAL_DOVE, Action.DEESCALATE):
            return -1.0
        return 0.0

    # ------------------------
    # Public API
    # ------------------------
    def reset(self) -> Dict[str, Any]:
        self.t = 0
        self.crisis_level = 0.0
        self._reset_beliefs()
        self._reset_latents()
        return self._obs()

    def dyad_mode(self, i: str, j: str) -> str:
        """
        Operationalization (minimal, but meaningful):
        - High P(malign) -> deterrence problem
        - Low P(malign) + high fear -> spiral-risk problem
        - Else -> uncertain
        """
        b = self.belief_intent[(i, j)]
        if b >= self.cfg.deterrence_threshold:
            return "deterrence"
        if b <= self.cfg.spiral_threshold and self.fear[(i, j)] >= self.cfg.fear_threshold:
            return "spiral"
        return "uncertain"

    def step(self, actions: Dict[str, Action]) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """
        Placeholder transition:
        - belief update happens first (agents interpret others' actions through noise)
        - fear accumulates from perceived hawkishness
        - crisis increases with hawkish actions, decreases with dovish actions, amplified by fear
        """
        self.t += 1

        # Defensive: fill missing actions with HOLD
        full_actions: Dict[str, Action] = {aid: actions.get(aid, Action.HOLD) for aid in self.agent_ids}

        # Update beliefs and fear from observed actions
        self._belief_update(full_actions)

        hawk = sum(1 for a in full_actions.values() if a in (Action.ARM, Action.POSTURE_UP, Action.SIGNAL_HAWK))
        dove = sum(1 for a in full_actions.values() if a in (Action.POSTURE_DOWN, Action.SIGNAL_DOVE, Action.DEESCALATE))

        # Fear-amplified crisis update (still a placeholder, but now "dyadic-aware")
        avg_fear = float(np.mean(list(self.fear.values()))) if self.fear else 0.0
        fear_amp = 1.0 + 0.5 * avg_fear
        self.crisis_level = float(np.clip(self.crisis_level + fear_amp * (0.08 * hawk - 0.10 * dove), 0.0, 1.0))
        war = self.crisis_level > 0.95

        rewards: Dict[str, float] = {}
        for aid in self.agent_ids:
            # stability-first placeholder
            rewards[aid] = 1.0 - 2.0 * self.crisis_level
            if war:
                rewards[aid] -= 50.0

        done = war or (self.t >= self.cfg.horizon)
        info = {
            "t": self.t,
            "crisis_level": self.crisis_level,
            "war": war,
            "avg_fear": avg_fear,
        }
        return self._obs(), rewards, done, info

    # ------------------------
    # Internal: belief + observation model
    # ------------------------
    def _belief_update(self, actions: Dict[str, Action]) -> None:
        """
        Transparent belief update:
        - i observes j's action with noise (obs_noise * (1 - intel_quality_i))
        - perceived hawkishness -> increases P(j is MALIGN)
        - perceived dovishness -> decreases P(j is MALIGN)
        - fear accumulates from perceived hawkishness (bounded via decay)
        """
        for i in self.agent_ids:
            iq = self.params[i].intel_quality
            noise_std = self.cfg.obs_noise * (1.0 - iq)

            for j in self.agent_ids:
                if i == j:
                    continue

                perceived = self._hawk_score(actions[j]) + float(self.rng.normal(0.0, noise_std))

                # Fear update: only hawkish perception increases fear; dovish doesn't add threat
                threat = max(0.0, perceived)
                self.fear[(i, j)] = self.cfg.fear_decay * self.fear[(i, j)] + (1.0 - self.cfg.fear_decay) * threat

                # Belief update in log-odds space
                log_odds = self._logit(self.belief_intent[(i, j)])
                log_odds += self.cfg.belief_lr * perceived
                self.belief_intent[(i, j)] = float(np.clip(self._sigmoid(log_odds), 0.05, 0.95))

    def _obs(self) -> Dict[str, Any]:
        """
        Observation is still minimal; we keep it light so tests stay stable.
        Later we add dyad-level signals, alliances, trade, etc.
        """
        obs: Dict[str, Any] = {}
        for i in self.agent_ids:
            base_std = self.cfg.obs_noise * (1.0 - self.params[i].intel_quality)
            signal = float(np.clip(self.crisis_level + self.rng.normal(0.0, base_std), 0.0, 1.0))
            obs[i] = {
                "t": self.t,
                "role": self.roles[i].value,
                "crisis_signal": signal,
            }
        return obs


def demo() -> None:
    env = ActirV01(ActirConfig(seed=42))
    obs = env.reset()
    print("RESET:", obs)
    print("TRUE INTENT (hidden):", {k: v.value for k, v in env.true_intent.items()})

    for _ in range(6):
        actions = {aid: Action.HOLD for aid in env.agent_ids}
        actions[env.agent_ids[1]] = Action.SIGNAL_HAWK   # RGP hawkish
        actions[env.agent_ids[2]] = Action.DEESCALATE    # MP tries to calm

        obs, rew, done, info = env.step(actions)

        # Example: MP (a2) classifies RGP (a1)
        mode = env.dyad_mode("a2", "a1")
        b = env.belief_intent[("a2", "a1")]
        f = env.fear[("a2", "a1")]

        print(info, "MP(a2)->RGP(a1):", {"belief_malign": round(b, 2), "fear": round(f, 2), "mode": mode})
        if done:
            break


if __name__ == "__main__":
    demo()
