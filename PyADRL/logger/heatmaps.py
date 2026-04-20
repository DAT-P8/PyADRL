import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from datetime import datetime
from ray.rllib.callbacks.callbacks import RLlibCallback
import time

# TODO: GRID SIZE SHOULD BE DYNAMIC BASED ON MAP CONFIG, NOT HARDCODED
_RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
)

EPISODE_FILE_PREFIX = "episode_states_tmp_"


class HeatmapCallback(RLlibCallback):
    grid_w = 0
    grid_h = 0
    target_x = 0
    target_y = 0

    def on_algorithm_init(
        self,
        *,
        algorithm,
        metrics_logger = None,
        **kwargs,
    ) -> None:
        # Clean up any old episode files from previous runs
        files = glob.glob(os.path.join(_RESULTS_DIR, f"{EPISODE_FILE_PREFIX}*.json"))
        for path in files:
            os.remove(path)
        
        if algorithm.config is None or algorithm.config.env_config is None:
            raise ValueError("HeatmapCallback requires env_config to be set in the algorithm config")
        
        self.grid_w = algorithm.config.env_config.get("map_width", 0)
        self.grid_h = algorithm.config.env_config.get("map_height", 0)
        self.target_x = algorithm.config.env_config.get("target_x", 0)
        self.target_y = algorithm.config.env_config.get("target_y", 0)

    def on_episode_created(self, *, episode, **kwargs):
        episode.custom_data["evader_states"] = {}
        episode.custom_data["pursuer_states"] = {}

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
        if not env or not env._infos:
            return
        episode_info = env._infos[0]

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
            states[agent_id].append((x, y))

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
        drone_states = {
            "evader_states": episode.custom_data.get("evader_states", {}),
            "pursuer_states": episode.custom_data.get("pursuer_states", {}),
        }
        # Each worker writes its own file — episode_id keeps filenames unique
        path = os.path.join(_RESULTS_DIR, f"{EPISODE_FILE_PREFIX}{id(episode)}.json")
        with open(path, "w") as f:
            json.dump(drone_states, f)

    def on_evaluate_end(
        self,
        *,
        algorithm,
        metrics_logger=None,
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:
        # Sleep to ensure all episode files have been written before we read them
        time.sleep(0.5)

        files = glob.glob(os.path.join(_RESULTS_DIR, f"{EPISODE_FILE_PREFIX}*.json"))
        if not files:
            print("[HeatmapCallback] No episode files found - nothing to plot.")
            return

        # Concatenate all temporary episode files into one big file
        concatenated = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "evader_states": [],
            "pursuer_states": [],
        }

        for path in files:
            with open(path) as f:
                data = json.load(f)
            concatenated["evader_states"].append(data.get("evader_states", {}))
            concatenated["pursuer_states"].append(data.get("pursuer_states", {}))

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_path = os.path.join(_RESULTS_DIR, f"drone_states_{date_str}.json")
        with open(merged_path, "w") as f:
            json.dump(concatenated, f)

        for path in files:  # clean up the temporary episode files after concatenation
            os.remove(path)

        # Plot the results
        self._plot_occupancy_heatmap(
            concatenated["evader_states"],
            title="Evader Position Heatmap",
            filename=f"results/heatmap_evader_{date_str}.png",
            color="YlOrRd",
        )
        self._plot_occupancy_heatmap(
            concatenated["pursuer_states"],
            title="Pursuer Position Heatmap",
            filename=f"results/heatmap_pursuer_{date_str}.png",
            color="Blues",
        )

        # For trace maps, show one representative episode instead of concatenating paths.
        evader_episode, pursuer_episode = self._select_representative_episode(
            concatenated["evader_states"],
            concatenated["pursuer_states"],
        )
        self._plot_trace_map(
            evader_episode,
            pursuer_episode,
            title="Agent Trace Map (Single Example Episode)",
            filename=f"results/trace_map_{date_str}.png",
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
        target_rect = patches.Rectangle(
            (self.target_x, self.target_y),
            1,
            1,
            linewidth=2,
            edgecolor="green",
            facecolor="green",
            alpha=0.3,
            label="Target",
        )
        ax.add_patch(target_rect)

        ax.tick_params(axis="both", which="major", pad=8)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Actor Traces Saved in {filename}")
