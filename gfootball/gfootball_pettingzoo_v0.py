""" 
Google Research Football (GRF) environment for PettingZoo (Parallel) APIs

"""

from gfootball.env import create_environment
from typing import Any, Dict, List, Tuple

import pettingzoo
import gymnasium as gym


class parallel_env(pettingzoo.ParallelEnv):
    def __init__(
        self,
        env_name: str = "",
        representation: str = "simplev1",
        rewards: str = "scoring",
        write_goal_dumps: bool = False,
        write_full_episode_dumps: bool = False,
        render: bool = False,
        write_video: bool = False,
        dump_frequency: int = 1,
        logdir: str = "",
        number_of_left_players_agent_controls: int = 1,
        number_of_right_players_agent_controls: int = 0,
        other_config_options: dict = {},
    ):
        """Creates a Google Research Football environment.
        Args:
        env_name: a name of a scenario to run, e.g. "11_vs_11_stochastic".
            The list of scenarios can be found in directory "scenarios".
        stacked: If True, stack 4 observations, otherwise, only the last
            observation is returned by the environment.
            Stacking is only possible when representation is one of the following:
            "pixels", "pixels_gray" or "extracted".
            In that case, the stacking is done along the last (i.e. channel)
            dimension.
        representation: String to define the representation used to build
            the observation. It can be one of the following:
            'pixels': the observation is the rendered view of the football field
            downsampled to 'channel_dimensions'. The observation size is:
            'channel_dimensions'x3 (or 'channel_dimensions'x12 when "stacked" is
            True).
            'pixels_gray': the observation is the rendered view of the football field
            in gray scale and downsampled to 'channel_dimensions'. The observation
            size is 'channel_dimensions'x1 (or 'channel_dimensions'x4 when stacked
            is True).
            'extracted': also referred to as super minimap. The observation is
            composed of 4 planes of size 'channel_dimensions'.
            Its size is then 'channel_dimensions'x4 (or 'channel_dimensions'x16 when
            stacked is True).
            The first plane P holds the position of players on the left
            team, P[y,x] is 255 if there is a player at position (x,y), otherwise,
            its value is 0.
            The second plane holds in the same way the position of players
            on the right team.
            The third plane holds the position of the ball.
            The last plane holds the active player.
            'simple115'/'simple115v2': the observation is a vector of size 115.
            It holds:
            - the ball_position and the ball_direction as (x,y,z)
            - one hot encoding of who controls the ball.
                [1, 0, 0]: nobody, [0, 1, 0]: left team, [0, 0, 1]: right team.
            - one hot encoding of size 11 to indicate who is the active player
                in the left team.
            - 11 (x,y) positions for each player of the left team.
            - 11 (x,y) motion vectors for each player of the left team.
            - 11 (x,y) positions for each player of the right team.
            - 11 (x,y) motion vectors for each player of the right team.
            - one hot encoding of the game mode. Vector of size 7 with the
                following meaning:
                {NormalMode, KickOffMode, GoalKickMode, FreeKickMode,
                CornerMode, ThrowInMode, PenaltyMode}.
            Can only be used when the scenario is a flavor of normal game
            (i.e. 11 versus 11 players).
            'simplev1': a compact simple representation, adapted from https://github.com/YuriCat/TamakEriFever, which is the implementation of 5th place solution in [gfootball Kaggle Competition](https://www.kaggle.com/c/google-football/discussion/203412).
            NOTE: this representation is only designed for cooperative MARL in academy scenarios.
                It holds:
                - (x,y) coordinate of current player
                - (x,y) direction of current player
                - (is_sprinting, is_dribbling) agent status
                - (Δx,Δy) relative coordinates of other left team players, size (n1-1) * 2
                - (Δx,Δy) relative coordinates of right team players, size n2 * 2
                - (Δx,Δy) relative coordinate of current player to the ball
                - (x,y) coordinates of other left team players, size (n1-1) * 2
                - (x,y) direction of other left team players, size (n1-1) * 2
                - (x,y) coordinates of right team players, size n2 * 2
                - (x,y) direction of right team players, size n2 * 2
                - (x,y,z) - ball position
                - (Δx,Δy,Δz) ball direction
                - one hot encoding of ball ownership (noone, left, right)
                - one hot encoding of `game_mode`
                - one hot encoding of which player is active (agent id), size n1
                Total dim:
                4 * 2 + (n1-1) * 2 * 3 + n2 * 2 * 3 + 3 + 3 + 3 + n1 + 7
                = 7 * n1 + 6 * n2 + 18
        rewards: Comma separated list of rewards to be added.
            Currently supported rewards are 'scoring' and 'checkpoints'.
        write_goal_dumps: whether to dump traces up to 200 frames before goals.
        write_full_episode_dumps: whether to dump traces for every episode.
        render: whether to render game frames.
            Must be enable when rendering videos or when using pixels
            representation.
        write_video: whether to dump videos when a trace is dumped.
        dump_frequency: how often to write dumps/videos (in terms of # of episodes)
            Sub-sample the episodes for which we dump videos to save some disk space.
        logdir: directory holding the logs.
        extra_players: A list of extra players to use in the environment.
            Each player is defined by a string like:
            "$player_name:left_players=?,right_players=?,$param1=?,$param2=?...."
        number_of_left_players_agent_controls: Number of left players an agent
            controls.
        number_of_right_players_agent_controls: Number of right players an agent
            controls.
        channel_dimensions: (width, height) tuple that represents the dimensions of
            SMM or pixels representation.
        other_config_options: dict that allows directly setting other options in
            the Config
        Returns:
        Google Research Football environment.
        """
        super().__init__()

        self._env = create_environment(
            env_name=env_name,
            representation=representation,
            rewards=rewards,
            write_goal_dumps=write_goal_dumps,
            write_full_episode_dumps=write_full_episode_dumps,
            render=render,
            write_video=write_video,
            dump_frequency=dump_frequency,
            logdir=logdir,
            number_of_left_players_agent_controls=number_of_left_players_agent_controls,
            number_of_right_players_agent_controls=number_of_right_players_agent_controls,
            other_config_options=other_config_options,
        )

        self._engine_config = self._env.unwrapped.engine_config
        self._control_config = self._env.unwrapped.control_config

        self.agents = [f"player_{i}" for i in range(self._control_config.number_of_players_agent_controls())]

        self.possible_agents = self.agents
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.observation_spaces, self.action_spaces = {}, {}

        assert isinstance(self._env.observation_space, gym.spaces.Box), "Observation space must be of type Box"
        assert isinstance(
            self._env.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)
        ), "action space must be of type Discrete or MultiDiscrete"

        if len(self.agents) == 1:
            self.observation_spaces = {self.agents[0]: self._env.observation_space}
            self.action_spaces = {self.agents[0]: self._env.action_space}
        else:
            for agent_id, agent in enumerate(self.agents):
                self.observation_spaces[agent] = gym.spaces.Box(
                    low=self._env.observation_space.low[agent_id],
                    high=self._env.observation_space.high[agent_id],
                    shape=self._env.observation_space.shape[1:],
                    dtype=self._env.observation_space.dtype,
                )
                self.action_spaces[agent] = gym.spaces.Discrete(self._env.action_space.nvec[agent_id])

    def observation_space(self, agent: str) -> gym.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]

    @property
    def num_agents(self) -> int:
        return len(self.agents)
    
    