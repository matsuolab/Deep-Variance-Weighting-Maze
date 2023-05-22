"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""


class BanditEvalMixIn:
    def initialize(self, env, config=None) -> None:
        super().initialize(env, config)

        # For regret analysis
        self._rews = self.env.rew_mean
        self._rew_avg_sum = 0
        self._rew_avg_max = self._rews.max()

    def update_regret(self, act):
        self._rew_avg_sum += self._rews[act]
        regret = self._rew_avg_max * self.n_step - self._rew_avg_sum
        return regret

    def evaluate(self):
        # No periodic evaluation in bandit algorithms
        return {}
