import unittest

from actir_v0_1 import ActirV01, ActirConfig, Action


class TestActirStylizedFacts(unittest.TestCase):
    def test_reset_returns_obs_per_agent(self):
        env = ActirV01(ActirConfig(seed=123))
        obs = env.reset()
        self.assertEqual(set(obs.keys()), set(env.agent_ids))
        for aid, o in obs.items():
            self.assertIn("t", o)
            self.assertIn("role", o)
            self.assertIn("crisis_signal", o)

    def test_hawkish_actions_raise_crisis_more_than_dovish(self):
        env = ActirV01(ActirConfig(seed=42))
        env.reset()

        # Baseline: mostly HOLD
        actions_hold = {aid: Action.HOLD for aid in env.agent_ids}
        _, _, _, info0 = env.step(actions_hold)
        crisis0 = info0["crisis_level"]

        # More hawkish actions
        env.reset()
        actions_hawk = {aid: Action.SIGNAL_HAWK for aid in env.agent_ids}
        _, _, _, info1 = env.step(actions_hawk)
        crisis1 = info1["crisis_level"]

        # More dovish actions
        env.reset()
        actions_dove = {aid: Action.DEESCALATE for aid in env.agent_ids}
        _, _, _, info2 = env.step(actions_dove)
        crisis2 = info2["crisis_level"]

        self.assertGreater(crisis1, crisis0)
        self.assertLess(crisis2, crisis0)

    @unittest.skip("Enable after we implement dyadic intent + spiral/deterrence inference tied to Jervis pages.")
    def test_spiral_vs_deterrence_modes(self):
        pass


if __name__ == "__main__":
    unittest.main()
