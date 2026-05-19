import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from ray.rllib.callbacks.callbacks import RLlibCallback


class HeatmapCallback(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.grid_w = 0
        self.grid_h = 0
        self.target_x = 0
        self.target_y = 0
        self.objects = []
        self.figure_path = None

    def on_algorithm_init(
        self,
        *,
        algorithm,
        metrics_logger=None,
        **kwargs,
    ) -> None:
        if algorithm.config is None or algorithm.config.env_config is None:
            raise ValueError(
                "HeatmapCallback requires env_config to be set in the algorithm config"
            )

        self.grid_w = algorithm.config.env_config.get("width", 0)
        self.grid_h = algorithm.config.env_config.get("height", 0)
        self.target_x = algorithm.config.env_config.get("target_x", 0)
        self.target_y = algorithm.config.env_config.get("target_y", 0)
        self.objects = algorithm.config.env_config.get("objects", [])
        self.figure_path = algorithm.config.env_config.get("figure_path")

    # Shield type constants used as keys in shield_data and legend labels.
    SHIELD_DRONE_OBJ = "drone_object_collision"
    SHIELD_OUT_OF_BOUNDS = "out_of_bounds"
    SHIELD_COLLISION = "collision"

    SHIELD_COLORS = {
        SHIELD_DRONE_OBJ: "#e31a1c",
        SHIELD_OUT_OF_BOUNDS: "#ff7f00",
        SHIELD_COLLISION: "#6a3d9a",
    }

    def on_episode_created(self, *, episode, **kwargs):
        episode.custom_data["evader_states"] = {}
        episode.custom_data["pursuer_states"] = {}
        episode.custom_data["evader_shield_data"] = {}
        episode.custom_data["pursuer_shield_data"] = {}

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

        # Build a map of drone_id -> set of shield types activated this step.
        # shield_events are shared across agents, so we only need to scan once.
        step_shield_map: dict[int, str] = {}
        first_agent_info = next(iter(episode_info.values()), {})
        for event in first_agent_info.get("shield_events", []):
            if event.drone_object_collision_event is not None:
                for did in event.drone_object_collision_event.drone_ids:
                    step_shield_map.setdefault(did, self.SHIELD_DRONE_OBJ)
            elif event.out_of_bounds_event is not None:
                for did in event.out_of_bounds_event.drone_ids:
                    step_shield_map.setdefault(did, self.SHIELD_OUT_OF_BOUNDS)
            elif event.collision_event is not None:
                for did in event.collision_event.drone_ids:
                    step_shield_map.setdefault(did, self.SHIELD_COLLISION)

        for agent_id, agent_info in episode_info.items():
            drone = agent_info.get("drone_state", {})
            x, y = drone.get("x"), drone.get("y")

            if agent_id.startswith("evader"):
                states = episode.custom_data["evader_states"]
                shield_data = episode.custom_data["evader_shield_data"]
            elif agent_id.startswith("pursuer"):
                states = episode.custom_data["pursuer_states"]
                shield_data = episode.custom_data["pursuer_shield_data"]
            else:
                raise ValueError(f"Unknown agent_id {agent_id} in HeatmapCallback")

            if agent_id not in states:
                states[agent_id] = []
                shield_data[agent_id] = []

            states[agent_id].append([x, y])

            # Drone id is the integer suffix of the agent name (evader_2 -> 2).
            try:
                drone_id = int(agent_id.rsplit("_", 1)[-1])
            except ValueError:
                drone_id = -1
            shield_data[agent_id].append(step_shield_map.get(drone_id))

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
            "evader_shield_data": episode.custom_data.get("evader_shield_data", {}),
            "pursuer_shield_data": episode.custom_data.get("pursuer_shield_data", {}),
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

        # Plot the results
        self._plot_occupancy_heatmap(
            [episode.get("evader_states", {}) for episode in drone_states],
            title="Evader Position Heatmap",
            filename="heatmap_evader",
            color="YlOrRd",
        )
        self._plot_occupancy_heatmap(
            [episode.get("pursuer_states", {}) for episode in drone_states],
            title="Pursuer Position Heatmap",
            filename="heatmap_pursuer",
            color="Blues",
        )

        # For trace maps, show one representative episode instead of concatenating paths.
        evader_episodes = [episode.get("evader_states", {}) for episode in drone_states]
        pursuer_episodes = [episode.get("pursuer_states", {}) for episode in drone_states]
        evader_episode, pursuer_episode, best_idx = self._select_representative_episode(
            evader_episodes,
            pursuer_episodes,
        )
        evader_shield = (
            drone_states[best_idx].get("evader_shield_data", {}) if best_idx >= 0 else {}
        )
        pursuer_shield = (
            drone_states[best_idx].get("pursuer_shield_data", {}) if best_idx >= 0 else {}
        )
        self._plot_trace_map(
            evader_episode,
            pursuer_episode,
            evader_shield_data=evader_shield,
            pursuer_shield_data=pursuer_shield,
            title="Agent Trace Map (Single Example Episode)",
            filename="trace_map",
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
            return {}, {}, -1

        return evader_episodes[best_idx], pursuer_episodes[best_idx], best_idx

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
        # Draw objects as solid grey boxes
        self._draw_objects(ax)

        plt.tight_layout()

        if self.figure_path:
            path = self.figure_path / f"{filename}.svg"
            plt.savefig(path, dpi=150)
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
        evader_shield_data: dict | None = None,
        pursuer_shield_data: dict | None = None,
        title,
        filename,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        # shield_type -> list of (x, y) across all agents/groups
        shield_positions: dict[str, list[tuple[float, float]]] = {
            k: [] for k in self.SHIELD_COLORS
        }

        def _plot_group(episode_states, shield_data, *, color, role_name):
            plotted_any = False
            used_label = False

            if not isinstance(episode_states, dict):
                return False

            for agent_id, positions in episode_states.items():
                if not positions:
                    continue

                shields = (shield_data or {}).get(agent_id, [])

                valid_indices = [
                    i
                    for i, (x, y) in enumerate(positions)
                    if isinstance(x, (int, float))
                    and isinstance(y, (int, float))
                    and 0 <= x < self.grid_w
                    and 0 <= y < self.grid_h
                ]
                if not valid_indices:
                    continue

                cleaned = [positions[i] for i in valid_indices]
                cleaned_shields = [
                    shields[i] if i < len(shields) else None for i in valid_indices
                ]

                # Draw agent paths at cell centers so markers sit inside cells.
                xs = [p[0] + 0.5 for p in cleaned]
                ys = [p[1] + 0.5 for p in cleaned]
                label = role_name if not used_label else None

                if len(cleaned) == 1:
                    ax.scatter(xs, ys, color=color, alpha=0.4, s=20, label=label)
                else:
                    ax.plot(xs, ys, color=color, alpha=0.7, linewidth=1.5, label=label)

                # Mark trajectory start/end so movement direction is visible.
                ax.scatter(
                    xs[0], ys[0],
                    marker="o", color=color, edgecolors="black",
                    linewidths=0.3, s=28, alpha=0.8,
                )
                ax.scatter(xs[-1], ys[-1], marker="x", color=color, s=30, alpha=0.9)

                # Collect shielded positions for overlay after all paths are drawn.
                for (x, y), shield_type in zip(cleaned, cleaned_shields):
                    if shield_type in shield_positions:
                        shield_positions[shield_type].append((x + 0.5, y + 0.5))

                used_label = True
                plotted_any = True

            return plotted_any

        has_evaders = _plot_group(
            evader_episode_states, evader_shield_data, color="#d95f02", role_name="Evader"
        )
        has_pursuers = _plot_group(
            pursuer_episode_states, pursuer_shield_data, color="#1f77b4", role_name="Pursuer"
        )

        # Overlay shield activation markers on top of trajectories.
        shield_labels = {
            self.SHIELD_DRONE_OBJ: "Shield: drone-object collision",
            self.SHIELD_OUT_OF_BOUNDS: "Shield: out of bounds",
            self.SHIELD_COLLISION: "Shield: drone-drone collision",
        }
        for shield_type, color in self.SHIELD_COLORS.items():
            pts = shield_positions[shield_type]
            if not pts:
                continue
            sx, sy = zip(*pts)
            ax.scatter(
                sx, sy,
                marker="D",
                color=color,
                edgecolors="black",
                linewidths=0.4,
                s=40,
                alpha=0.9,
                zorder=5,
                label=shield_labels[shield_type],
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

        # Draw objects as solid grey boxes
        self._draw_objects(ax)

        ax.tick_params(axis="both", which="major", pad=8)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right")

        plt.tight_layout()

        if self.figure_path:
            path = self.figure_path / f"{filename}.svg"
            plt.savefig(path, dpi=150)
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

    def _draw_objects(self, ax):
        """Draw objects as solid grey 1x1 boxes."""
        print(self.objects)
        if not self.objects:
            return

        used_label = False
        for obj in self.objects:
            try:
                x, y = obj
            except Exception:
                # skip malformed entries
                continue

            rect = patches.Rectangle(
                (x, y),
                1,
                1,
                linewidth=1,
                edgecolor="black",
                facecolor="grey",
                alpha=1.0,
                label=("Object" if not used_label else None),
                zorder=5,
            )
            ax.add_patch(rect)
            used_label = True
