import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from ray.rllib.callbacks.callbacks import RLlibCallback
import time

# TODO: GRID SIZE SHOULD BE DYNAMIC BASED ON MAP CONFIG, NOT HARDCODED
GRID_W, GRID_H = 11, 11

_RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
)

EPISODE_FILE_PREFIX = "episode_states_tmp_"


class HeatmapCallback(RLlibCallback):
    def on_evaluation_start(self, *, algorithm, **kwargs):
        # Clean up any old episode files from previous runs
        files = glob.glob(os.path.join(_RESULTS_DIR, f"{EPISODE_FILE_PREFIX}*.json"))
        for path in files:
            os.remove(path)

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

        grid = np.zeros((GRID_H, GRID_W), dtype=int)
        for x, y in all_positions:
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                grid[y, x] += 1

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            grid,
            cmap=color,
            linewidths=0.3,
            linecolor="grey",
            annot=(GRID_W <= 20 and GRID_H <= 20),  # only show numbers if grid is small
            fmt="d",
            ax=ax,
            cbar_kws={"label": "Visit count"},
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        # plt.show()
        plt.close(fig)
        print(f"[HeatmapCallback] Saved {filename}")

    def _plot_capture_heatmap(self, episode_states, *, title, filename, color):
        # This method can be implemented to plot heatmaps of capture locations
        # This method would require additional data collection
        pass

    def _plot_trace_map(self, episode_states, *, title, filename):
        # This method can be implemented to plot trace maps of agent trajectories
        # This method might require additional data collection?
        pass
