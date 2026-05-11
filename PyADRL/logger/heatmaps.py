import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from ray.rllib.callbacks.callbacks import RLlibCallback

from PyADRL.utils.paths import get_experiments_dir, get_model_maps_dir

# TODO: GRID SIZE SHOULD BE DYNAMIC BASED ON MAP CONFIG, NOT HARDCODED
_RESULTS_DIR = get_experiments_dir()


class HeatmapCallback(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.grid_w = 0
        self.grid_h = 0
        self.target_x = 0
        self.target_y = 0
        self.model_name = ""

    def on_algorithm_init(
        self,
        *,
        algorithm,
        metrics_logger=None,
        **kwargs,
    ) -> None:
        if algorithm and algorithm.config:
            self.grid_w = algorithm.config.env_config.get("map_width", 0)
            self.grid_h = algorithm.config.env_config.get("map_height", 0)
            self.target_x = algorithm.config.env_config.get("target_x", 0)
            self.target_y = algorithm.config.env_config.get("target_y", 0)
            self.model_name = algorithm.config.env_config.get("model_name", "")

        if algorithm.config is None or algorithm.config.env_config is None:
            raise ValueError(
                "HeatmapCallback requires env_config to be set in the algorithm config"
            )

        self.grid_w = algorithm.config.env_config.get("map_width", 0)
        self.grid_h = algorithm.config.env_config.get("map_height", 0)
        self.target_x = algorithm.config.env_config.get("target_x", 0)
        self.target_y = algorithm.config.env_config.get("target_y", 0)
        self.model_name = algorithm.config.env_config.get("model_name", "")

    def on_episode_created(self, *, episode, **kwargs):
        episode.custom_data["evader_states"] = {}
        episode.custom_data["pursuer_states"] = {}

    def _get_episode_info(self, env, env_index: int):
        if not env or not getattr(env, "_infos", None):
            return None

        if isinstance(env._infos, (list, tuple)):
            if 0 <= env_index < len(env._infos):
                return env._infos[env_index]
            return env._infos[0] if env._infos else None

        return env._infos

    def on_episode_step(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        **kwargs,
    ):
        episode_info = self._get_episode_info(env, env_index)
        if not episode_info:
            return

        for agent_id, agent_info in episode_info.items():
            drone = agent_info.get("drone_state", {})
            x, y = drone.get("x"), drone.get("y")

            if agent_id.startswith("evader"):
                states = episode.custom_data["evader_states"]
            elif agent_id.startswith("pursuer"):
                states = episode.custom_data["pursuer_states"]
            else:
                raise ValueError(f"Unknown agent_id {agent_id} in HeatmapCallback")

            if agent_id not in states:
                states[agent_id] = []
            states[agent_id].append([x, y])

    def on_episode_end(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        **kwargs,
    ):
        if metrics_logger is None:
            return

        drone_states = {
            "evader_states": episode.custom_data.get("evader_states", {}),
            "pursuer_states": episode.custom_data.get("pursuer_states", {}),
        }
        metrics_logger.log_value("drone_states", drone_states, reduce="item_series")

    def on_evaluate_end(
        self,
        *,
        algorithm,
        metrics_logger=None,
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:
        env_runners = evaluation_metrics.get("env_runners", {})
        drone_states = env_runners.get("drone_states")
        if not drone_states:
            print("[HeatmapCallback] No episode states found - nothing to plot.")
            return

        date_str = f"eval_{len(drone_states)}"

        # Plot the results
        self._plot_occupancy_heatmap(
            [episode.get("evader_states", {}) for episode in drone_states],
            title="Evader Position Heatmap",
            filename=get_model_maps_dir(self.model_name)
            / f"heatmap_evader_{date_str}.png",
            color="YlOrRd",
        )
        self._plot_occupancy_heatmap(
            [episode.get("pursuer_states", {}) for episode in drone_states],
            title="Pursuer Position Heatmap",
            filename=get_model_maps_dir(self.model_name)
            / f"heatmap_pursuer_{date_str}.png",
            color="Blues",
        )

        # For trace maps, show one representative episode instead of concatenating paths.
        evader_episode, pursuer_episode = self._select_representative_episode(
            [episode.get("evader_states", {}) for episode in drone_states],
            [episode.get("pursuer_states", {}) for episode in drone_states],
        )
        self._plot_trace_map(
            evader_episode,
            pursuer_episode,
            title="Agent Trace Map (Single Example Episode)",
            filename=get_model_maps_dir(self.model_name) / f"trace_map_{date_str}.png",
        )

    def _select_representative_episode(self, evader_episodes, pursuer_episodes):
        best_idx = -1
        best_score = -1
        n = min(len(evader_episodes), len(pursuer_episodes))

        for idx in range(n):
            ev = evader_episodes[idx]
            pu = pursuer_episodes[idx]
            if not isinstance(ev, dict) or not isinstance(pu, dict):
                continue

            ev_len = sum(len(path) for path in ev.values() if isinstance(path, list))
            pu_len = sum(len(path) for path in pu.values() if isinstance(path, list))
            score = pu_len + ev_len

            # choose the episode with the longest combined trajectory length
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            return {}, {}

        return evader_episodes[best_idx], pursuer_episodes[best_idx]

    # PLOTTING METHODS
    def _plot_occupancy_heatmap(self, episode_states, *, title, filename, color):
        all_positions = [
            pos
            for episode in episode_states
            for positions in episode.values()
            for pos in positions
        ]

        if not all_positions:
            print(f"[HeatmapCallback] No positions collected for {title}, skipping.")
            return

        grid = np.zeros((self.grid_h, self.grid_w), dtype=int)
        for x, y in all_positions:
            if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                grid[y, x] += 1

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            grid,
            cmap=color,
            linewidths=0.3,
            linecolor="grey",
            annot=(
                self.grid_w <= 20 and self.grid_h <= 20
            ),  # only show numbers if grid is small
            fmt="d",
            ax=ax,
            cbar_kws={"label": "Visit count"},
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.invert_yaxis()

        # Draw the target square
        self._draw_target(ax)

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        # plt.show()
        plt.close(fig)
        print(f"Heatmap Saved in {filename}")

    def _plot_capture_heatmap(self, episode_states, *, title, filename, color):
        # This method can be implemented to plot heatmaps of capture locations
        # This method would require additional data collection
        pass

    def _plot_trace_map(
        self,
        evader_episode_states,
        pursuer_episode_states,
        *,
        title,
        filename,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        def _plot_group(episode_states, *, color, role_name):
            plotted_any = False
            used_label = False

            if not isinstance(episode_states, dict):
                return False

            for positions in episode_states.values():
                if not positions:
                    continue

                cleaned = [
                    (x, y)
                    for x, y in positions
                    if isinstance(x, (int, float))
                    and isinstance(y, (int, float))
                    and 0 <= x < self.grid_w
                    and 0 <= y < self.grid_h
                ]
                if not cleaned:
                    continue

                # Draw agent paths at cell centers so markers sit inside cells.
                xs = [p[0] + 0.5 for p in cleaned]
                ys = [p[1] + 0.5 for p in cleaned]
                label = role_name if not used_label else None

                if len(cleaned) == 1:
                    ax.scatter(
                        xs,
                        ys,
                        color=color,
                        alpha=0.4,
                        s=20,
                        label=label,
                    )
                else:
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        alpha=0.7,
                        linewidth=1.5,
                        label=label,
                    )

                # Mark trajectory start/end so movement direction is visible.
                ax.scatter(
                    xs[0],
                    ys[0],
                    marker="o",  # start
                    color=color,
                    edgecolors="black",
                    linewidths=0.3,
                    s=28,
                    alpha=0.8,
                )
                ax.scatter(
                    xs[-1],
                    ys[-1],
                    marker="x",  # end
                    color=color,
                    s=30,
                    alpha=0.9,
                )

                used_label = True
                plotted_any = True

            return plotted_any

        has_evaders = _plot_group(
            evader_episode_states, color="#d95f02", role_name="Evader"
        )
        has_pursuers = _plot_group(
            pursuer_episode_states,
            color="#1f77b4",
            role_name="Pursuer",
        )

        if not (has_evaders or has_pursuers):
            plt.close(fig)
            print(
                "[HeatmapCallback] No valid trajectories collected, skipping trace map."
            )
            return

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, self.grid_w)
        ax.set_ylim(0, self.grid_h)

        # Label cell indices at centers (0.5, 1.5, ...) while keeping grid on boundaries.
        x_center_ticks = np.arange(0.5, self.grid_w, 1)
        y_center_ticks = np.arange(0.5, self.grid_h, 1)
        ax.set_xticks(x_center_ticks)
        ax.set_yticks(y_center_ticks)
        ax.set_xticklabels(np.arange(0, self.grid_w, 1))
        ax.set_yticklabels(np.arange(0, self.grid_h, 1))

        # Boundary grid lines at integer coordinates.
        ax.set_xticks(np.arange(0, self.grid_w + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, self.grid_h + 1, 1), minor=True)
        ax.grid(True, which="minor", linewidth=0.3, alpha=0.4)

        # Draw the target square
        self._draw_target(ax)

        ax.tick_params(axis="both", which="major", pad=8)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Actor Traces Saved in {filename}")

    def _draw_target(self, ax):
        rect = patches.Rectangle(
            (self.target_x, self.target_y),
            1,
            1,
            linewidth=2,
            edgecolor="green",
            facecolor="green",
            alpha=0.3,
            label="Target",
        )
        ax.add_patch(rect)
