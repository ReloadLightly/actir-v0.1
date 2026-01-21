import unittest

from actir_v0_1 import ActirV01, ActirConfig, Action


class TestActirStylizedFacts(unittest.TestCase):
    def test_reset_returns_obs_per_agent(self):
        env = ActirV01(ActirConfig(seed=123))
        obs = env.reset()
        self.assertEqual(set(obs.keys()), set(env.agent_ids))
        for _, o in obs.items():
            self.assertIn("t", o)
            self.assertIn("role", o)
            self.assertIn("crisis_signal", o)

    def test_hawkish_actions_raise_crisis_more_than_dovish(self):
        env = ActirV01(ActirConfig(seed=42))
        env.reset()

        # Give the system headroom so DEESCALATE can reduce crisis below baseline
        env.crisis_level = 0.5

        actions_hold = {aid: Action.HOLD for aid in env.agent_ids}
        _, _, _, info0 = env.step(actions_hold)
        crisis0 = info0["crisis_level"]

        env.reset()
        env.crisis_level = 0.5
        actions_hawk = {aid: Action.SIGNAL_HAWK for aid in env.agent_ids}
        _, _, _, info1 = env.step(actions_hawk)
        crisis1 = info1["crisis_level"]

        env.reset()
        env.crisis_level = 0.5
        actions_dove = {aid: Action.DEESCALATE for aid in env.agent_ids}
        _, _, _, info2 = env.step(actions_dove)
        crisis2 = info2["crisis_level"]

        self.assertGreater(crisis1, crisis0)
        self.assertLess(crisis2, crisis0)

    def test_spiral_vs_deterrence_modes(self):
        env = ActirV01(ActirConfig(seed=0, obs_noise=0.0))
        env.reset()

        i, j = "a2", "a1"  # MP observing RGP

        env.belief_intent[(i, j)] = 0.8
        env.fear[(i, j)] = 0.0
        self.assertEqual(env.dyad_mode(i, j), "deterrence")

        env.belief_intent[(i, j)] = 0.2
        env.fear[(i, j)] = 0.9
        self.assertEqual(env.dyad_mode(i, j), "spiral")

        env.belief_intent[(i, j)] = 0.5
        env.fear[(i, j)] = 0.0
        self.assertEqual(env.dyad_mode(i, j), "uncertain")


if __name__ == "__main__":
    unittest.main(verbosity=2)
