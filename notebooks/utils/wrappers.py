'''
Custom wrappers for gym environments
'''

from typing import Any, Callable, List

import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

import gymnasium as gym
from gymnasium import logger
from gymnasium.core import ActType, ObsType, RenderFrame


class RecordGif(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    RecordGif is a wrapper that records the environment as a gif.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        gif_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        gif_length: int = 0,
        name_prefix: str = "rl-gif",
        fps: int | None = None,
        disable_logger: bool = True

    ):
        super().__init__(env)

        if env.render_mode in {None, "human", "ansi"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with RecordGif.",
                "Initialize your environment with a render_mode ",
                "that returns an image, such as rgb_array.",
            )

        if episode_trigger is None and step_trigger is None:
            from gymnasium.utils.save_video import capped_cubic_video_schedule # pylint: disable=import-outside-toplevel
            episode_trigger = capped_cubic_video_schedule

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.disable_logger = disable_logger

        self.gif_folder = os.path.abspath(gif_folder)
        if os.path.isdir(self.gif_folder):
            logger.warn(
                f"Overwriting existing gifs at {self.gif_folder} folder "
                f"(try specifying a different `gif_folder` for the `RecordGif` "
                f" wrapper if this is not desired)"
            )
        os.makedirs(self.gif_folder, exist_ok=True)

        self.frames_per_sec = fps if fps is not None else 15
        self.name_prefix = name_prefix
        self._gif_name: str|None = None
        self.gif_length = gif_length if gif_length != 0 else float("inf")
        self.recording = False
        self.recorded_frames: list[RenderFrame] = []
        self.render_history: list[RenderFrame] = []

        self.step_id = -1
        self.episode_id = -1

    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        frame = self.env.render()
        if isinstance(frame, List):
            if len(frame) == 0:
                return
            self.render_history += frame
            frame = frame[-1]

        if isinstance(frame, np.ndarray):
            self.recorded_frames.append(frame)

        else:
            self.stop_recording()
            logger.warn(
                f"Recording stopped: expected type of frame returned"
                f" by render to be a numpy array, got instead {type(frame)}."
            )

    def reset(
        self, *,seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and eventually starts a new recording."""
        obs, info = super().reset(seed=seed, options=options)
        self.episode_id += 1

        if self.recording and self.gif_length == float("inf"):
            self.stop_recording()

        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")

        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) >= self.gif_length:
                self.stop_recording()

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """
        Steps through the environment using action, recording observations 
        if :attr:`self.recording`.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        self.step_id += 1

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")

        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) >= self.gif_length:
                self.stop_recording()

        return obs, reward, terminated, truncated, info

    def render(self) -> RenderFrame | List[RenderFrame]:
        """
        Compute the render frames as specified by render_mode attribute 
        during initialization of the environment.
        """
        render_out = super().render()

        if self.recording and isinstance(render_out, List):
            self.recorded_frames += render_out

        if len(self.render_history) > 0:
            temp = self.render_history
            self.render_history = []
            return temp + render_out

        return render_out

    def _save_frames_as_gif(self):
        frames = self.recorded_frames
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        writer = animation.PillowWriter(fps=self.frames_per_sec, bitrate=1800)

        file_name = os.path.join(self.gif_folder, self._gif_name)
        anim.save(file_name, writer=writer)
        plt.close()

    def close(self):
        """Closes the wrapper then the gif recorder."""
        super().close()
        if self.recording:
            self.stop_recording()

    def start_recording(self, gif_name: str):
        """
        Start a new recording. If it is already recording, stops the 
        current recording before starting the new one.
        """
        if self.recording:
            self.stop_recording()
        self.recording = True
        self._gif_name = gif_name + '.gif'

    def stop_recording(self):
        """Stop current recording and saves the video."""
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a gif as there were zero frames to save.")
        else:
            self._save_frames_as_gif()

        self.recorded_frames = []
        self.recording = False
        self._gif_name = None

    def __del__(self):
        """Warn the user in case last gif wasn't saved."""
        if len(self.recorded_frames) > 0:
            logger.warn("Unable to save last gif! Did you call close()?")
