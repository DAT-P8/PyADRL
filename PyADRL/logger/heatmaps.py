import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from ray.rllib.callbacks.callbacks import RLlibCallback

# Shield type constants used as keys in shield_data and legend labels.
SHIELD_DRONE_OBJ = "drone_object_collision"
SHIELD_OUT_OF_BOUNDS = "out_of_bounds"
SHIELD_COLLISION = "collision"

SHIELD_COLORS = {
    SHIELD_DRONE_OBJ: "#e31a1c",
    SHIELD_OUT_OF_BOUNDS: "#ff7f00",
    SHIELD_COLLISION: "#6a3d9a",
}


class HeatmapCallback(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.grid_w = 0
        self.grid_h = 0
        self.target_x = 0
        self.target_y = 0
        self.objects = []
        self.figure_path = None
        self.n_evaders = 0

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
        self.n_evaders = algorithm.config.env_config.get("n_evaders", 0)

    def on_episode_created(self, *, episode, **kwargs):
        episode.custom_data["evader_states"] = {}
        episode.custom_data["pursuer_states"] = {}
        episode.custom_data["evader_shield_data"] = {}
        episode.custom_data["pursuer_shield_data"] = {}
        episode.custom_data["evader_unsafe_positions"] = {}
        episode.custom_data["pursuer_unsafe_positions"] = {}
        episode.custom_data["capture_positions"] = []
        episode.custom_data["breached"] = False

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

        # Detect breach: an evader reached the target this step.
        try:
            for event in first_agent_info.get("events", []):
                if getattr(event, "target_reached_event", None) is not None:
                    episode.custom_data["breached"] = True
                    break
        except Exception:
            pass

        for event in first_agent_info.get("shield_events", []):
            if event.drone_object_collision_event is not None:
                for did in event.drone_object_collision_event.drone_ids:
                    step_shield_map.setdefault(did, SHIELD_DRONE_OBJ)
            elif event.out_of_bounds_event is not None:
                for did in event.out_of_bounds_event.drone_ids:
                    step_shield_map.setdefault(did, SHIELD_OUT_OF_BOUNDS)
            elif event.collision_event is not None:
                for did in event.collision_event.drone_ids:
                    step_shield_map.setdefault(did, SHIELD_COLLISION)

        for agent_id, agent_info in episode_info.items():
            drone = agent_info.get("drone_state", {})
            x, y = drone.get("x"), drone.get("y")

            if agent_id.startswith("evader"):
                states = episode.custom_data["evader_states"]
                shield_data = episode.custom_data["evader_shield_data"]
                unsafe_positions = episode.custom_data["evader_unsafe_positions"]
            elif agent_id.startswith("pursuer"):
                states = episode.custom_data["pursuer_states"]
                shield_data = episode.custom_data["pursuer_shield_data"]
                unsafe_positions = episode.custom_data["pursuer_unsafe_positions"]
            else:
                raise ValueError(f"Unknown agent_id {agent_id} in HeatmapCallback")

            if agent_id not in states:
                states[agent_id] = []
                shield_data[agent_id] = []
                unsafe_positions[agent_id] = []

            states[agent_id].append([x, y])

            # Drone id is the integer suffix of the agent name (evader_2 -> 2).
            try:
                drone_id = int(agent_id.rsplit("_", 1)[-1])
            except ValueError:
                drone_id = -1
            shield_data[agent_id].append(step_shield_map.get(drone_id))

            unsafe = agent_info.get("unsafe_drone_state")
            unsafe_positions[agent_id].append(unsafe)

            capture = agent_info.get("capture_position")
            if capture is not None:
                episode.custom_data["capture_positions"].append(capture)

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

        capture_positions = episode.custom_data.get("capture_positions", [])
        breached = episode.custom_data.get("breached", False)
        n_ev = self.n_evaders or len(episode.custom_data.get("evader_states", {}))
        pursuer_win = (not breached) and n_ev > 0 and len(capture_positions) >= n_ev

        drone_states = {
            "evader_states": episode.custom_data.get("evader_states", {}),
            "pursuer_states": episode.custom_data.get("pursuer_states", {}),
            "evader_shield_data": episode.custom_data.get("evader_shield_data", {}),
            "pursuer_shield_data": episode.custom_data.get("pursuer_shield_data", {}),
            "evader_unsafe_positions": episode.custom_data.get(
                "evader_unsafe_positions", {}
            ),
            "pursuer_unsafe_positions": episode.custom_data.get(
                "pursuer_unsafe_positions", {}
            ),
            "capture_positions": capture_positions,
            "breached": breached,
            "pursuer_win": pursuer_win,
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
            filename="heatmap_evader",
            color="YlOrRd",
        )
        self._plot_occupancy_heatmap(
            [episode.get("pursuer_states", {}) for episode in drone_states],
            filename="heatmap_pursuer",
            color="Blues",
        )
        self._plot_capture_heatmap(
            [
                cap
                for episode in drone_states
                for cap in episode.get("capture_positions", [])
            ],
            filename="heatmap_captures",
        )
        self._plot_shielding_heatmap(
            [episode.get("evader_states", {}) for episode in drone_states],
            [episode.get("evader_shield_data", {}) for episode in drone_states],
            filename="heatmap_shielding_evader",
        )
        self._plot_shielding_heatmap(
            [episode.get("pursuer_states", {}) for episode in drone_states],
            [episode.get("pursuer_shield_data", {}) for episode in drone_states],
            filename="heatmap_shielding_pursuer",
        )

        # Three trace maps: longest, pursuer win, evader win (breach).
        trace_specs = [
            ("trace_map", None),
            ("trace_map_pursuer_win", "pursuer_win"),
            ("trace_map_evader_win", "breached"),
        ]
        evader_episodes = [episode.get("evader_states", {}) for episode in drone_states]
        pursuer_episodes = [
            episode.get("pursuer_states", {}) for episode in drone_states
        ]

        for filename, outcome_key in trace_specs:
            idx = self._select_trace_episode(
                evader_episodes, pursuer_episodes, drone_states, outcome_key
            )
            if idx < 0:
                if outcome_key is not None:
                    print(
                        f"[HeatmapCallback] No episode with {outcome_key}=True found,"
                        f" skipping {filename}."
                    )
                continue
            self._plot_trace_map(
                evader_episodes[idx],
                pursuer_episodes[idx],
                evader_shield_data=drone_states[idx].get("evader_shield_data", {}),
                pursuer_shield_data=drone_states[idx].get("pursuer_shield_data", {}),
                evader_unsafe_data=drone_states[idx].get("evader_unsafe_positions", {}),
                pursuer_unsafe_data=drone_states[idx].get(
                    "pursuer_unsafe_positions", {}
                ),
                capture_positions=drone_states[idx].get("capture_positions", []),
                filename=filename,
            )

    def _select_trace_episode(
        self,
        evader_episodes: list,
        pursuer_episodes: list,
        drone_states: list,
        outcome_key: str | None,
    ) -> int:
        """Return the index of the best episode for a trace map.

        If outcome_key is None, pick the longest episode overall.
        If outcome_key is 'pursuer_win' or 'breached', restrict to episodes
        where that flag is True, then pick the longest among them.
        Returns -1 if no suitable episode exists.
        """
        best_idx = -1
        best_score = -1
        n = min(len(evader_episodes), len(pursuer_episodes), len(drone_states))

        for idx in range(n):
            try:
                if outcome_key is not None and not drone_states[idx].get(
                    outcome_key, False
                ):
                    continue

                ev = evader_episodes[idx]
                pu = pursuer_episodes[idx]
                if not isinstance(ev, dict) or not isinstance(pu, dict):
                    continue

                ev_len = sum(
                    len(path) for path in ev.values() if isinstance(path, list)
                )
                pu_len = sum(
                    len(path) for path in pu.values() if isinstance(path, list)
                )
                score = ev_len + pu_len

                if score > best_score:
                    best_score = score
                    best_idx = idx
            except Exception:
                continue

        return best_idx

    # PLOTTING METHODS
    def _plot_occupancy_heatmap(self, episode_states, *, filename, color):
        all_positions = [
            pos
            for episode in episode_states
            for positions in episode.values()
            for pos in positions
        ]

        if not all_positions:
            print(f"[HeatmapCallback] No positions collected for {filename}, skipping.")
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

    def _plot_shielding_heatmap(self, episode_states, shield_data, *, filename):
        grids = {
            stype: np.zeros((self.grid_h, self.grid_w), dtype=int)
            for stype in SHIELD_COLORS
        }

        for ep_idx, ep_states in enumerate(episode_states):
            if not isinstance(ep_states, dict):
                continue
            ep_shields = shield_data[ep_idx] if ep_idx < len(shield_data) else {}
            if not isinstance(ep_shields, dict):
                continue

            for agent_id, positions in ep_states.items():
                shields = ep_shields.get(agent_id, [])
                for i, (x, y) in enumerate(positions):
                    stype = shields[i] if i < len(shields) else None
                    if stype in grids:
                        px, py = positions[i - 1] if i > 0 else (x, y)
                        if 0 <= px < self.grid_w and 0 <= py < self.grid_h:
                            grids[stype][int(py), int(px)] += 1

        if sum(g.sum() for g in grids.values()) == 0:
            print(f"[HeatmapCallback] No shield activations for {filename}, skipping.")
            return

        n = len(SHIELD_COLORS)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (stype, _) in zip(axes, SHIELD_COLORS.items()):
            grid = grids[stype]
            sns.heatmap(
                grid,
                mask=grid == 0,
                cmap="Reds",
                vmin=0,
                vmax=max(1, int(grid.max())),
                linewidths=0.3,
                linecolor="grey",
                annot=(self.grid_w <= 20 and self.grid_h <= 20),
                fmt="d",
                ax=ax,
                cbar_kws={"label": "Shield activations"},
            )
            ax.set_title(stype.replace("_", " ").title())
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.invert_yaxis()
            self._draw_target(ax)
            self._draw_objects(ax)

        plt.tight_layout()

        if self.figure_path:
            path = self.figure_path / f"{filename}.svg"
            plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Shielding Heatmap Saved in {filename}")

    def _plot_capture_heatmap(self, all_capture_positions, *, filename):
        grid = np.zeros((self.grid_h, self.grid_w), dtype=int)
        for cap in all_capture_positions:
            try:
                x, y = int(cap["x"]), int(cap["y"])
                if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                    grid[y, x] += 1
            except (TypeError, KeyError, ValueError):
                continue

        if grid.sum() == 0:
            print(f"[HeatmapCallback] No captures collected for {filename}, skipping.")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            grid,
            cmap="Purples",
            linewidths=0.3,
            linecolor="grey",
            annot=(self.grid_w <= 20 and self.grid_h <= 20),
            fmt="d",
            ax=ax,
            cbar_kws={"label": "Capture count"},
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.invert_yaxis()

        self._draw_target(ax)
        self._draw_objects(ax)

        plt.tight_layout()

        if self.figure_path:
            path = self.figure_path / f"{filename}.svg"
            plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Capture Heatmap Saved in {filename}")

    def _plot_trace_map(
        self,
        evader_episode_states,
        pursuer_episode_states,
        *,
        evader_shield_data: dict | None = None,
        pursuer_shield_data: dict | None = None,
        evader_unsafe_data: dict | None = None,
        pursuer_unsafe_data: dict | None = None,
        capture_positions: list | None = None,
        filename,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw the target square
        self._draw_target(ax)
        # Draw objects as solid grey boxes
        self._draw_objects(ax)

        capture_set = set()
        for cap in capture_positions or []:
            try:
                capture_set.add((int(cap["x"]), int(cap["y"])))
            except (TypeError, KeyError, ValueError):
                pass

        # shield_type -> list of (x, y) across all agents/groups
        shield_positions: dict[str, list[tuple[float, float]]] = {
            k: [] for k in SHIELD_COLORS
        }

        def _plot_group(episode_states, shield_data, unsafe_data, *, color, use_capture_markers=False):
            plotted_any = False

            if not isinstance(episode_states, dict):
                return False

            for agent_id, positions in episode_states.items():
                if not positions:
                    continue

                shields = (shield_data or {}).get(agent_id, [])
                unsafe_list = (unsafe_data or {}).get(agent_id, [])

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
                cleaned_unsafe = [
                    unsafe_list[i] if i < len(unsafe_list) else None
                    for i in valid_indices
                ]

                # Draw agent paths at cell centers so markers sit inside cells.
                xs = [p[0] + 0.5 for p in cleaned]
                ys = [p[1] + 0.5 for p in cleaned]

                if len(cleaned) == 1:
                    ax.scatter(xs, ys, color=color, alpha=0.4, s=20)
                else:
                    n_seg = len(xs) - 1
                    ax.plot(xs, ys, color=color, alpha=0.4, linewidth=1.5, zorder=2)

                    for i in range(n_seg):
                        dx = xs[i + 1] - xs[i]
                        dy = ys[i + 1] - ys[i]
                        # cleaned_shields[i+1] is set when the move TO position i+1 was shielded
                        is_shielded = cleaned_shields[i + 1] is not None

                        if is_shielded:
                            # Green arrow: safe action the shield substituted.
                            # If dx=dy=0 the shield chose "stay in place" — draw a dot instead.
                            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                                ax.annotate(
                                    "",
                                    xy=(xs[i + 1], ys[i + 1]),
                                    xytext=(xs[i], ys[i]),
                                    annotation_clip=False,
                                    arrowprops=dict(
                                        arrowstyle="-|>",
                                        color="#2ca02c",
                                        alpha=0.8,
                                        lw=1.5,
                                        mutation_scale=7,
                                    ),
                                    zorder=7,
                                )
                            else:
                                ax.scatter(
                                    xs[i],
                                    ys[i],
                                    marker="o",
                                    color="#2ca02c",
                                    s=5,
                                    alpha=0.8,
                                    zorder=7,
                                )
                            # Dashed red arrow: action blocked by shield.
                            # Draw shaft and head separately — FancyArrowPatch with linestyle="dashed"
                            # mangles the arrowhead tip.
                            # Use float() to handle numpy scalars (np.float64 not subclass of float in NumPy 2.x).
                            unsafe = cleaned_unsafe[i + 1]
                            if unsafe is not None:
                                try:
                                    ux = float(unsafe.get("x")) + 0.5
                                    uy = float(unsafe.get("y")) + 0.5
                                    udx, udy = ux - xs[i], uy - ys[i]
                                    if abs(udx) > 1e-6 or abs(udy) > 1e-6:
                                        # Dashed shaft — stop before the arrowhead tip
                                        dist = (udx**2 + udy**2) ** 0.5
                                        nx, ny = udx / dist, udy / dist
                                        head_len = min(0.25, dist * 0.3)
                                        ax.plot(
                                            [xs[i], ux - nx * head_len],
                                            [ys[i], uy - ny * head_len],
                                            color="#d62728",
                                            alpha=0.7,
                                            lw=1.5,
                                            linestyle="dashed",
                                            zorder=7,
                                        )
                                        # Solid arrowhead only
                                        ax.annotate(
                                            "",
                                            xy=(ux, uy),
                                            xytext=(
                                                ux - nx * head_len,
                                                uy - ny * head_len,
                                            ),
                                            annotation_clip=False,
                                            arrowprops=dict(
                                                arrowstyle="-|>",
                                                color="#d62728",
                                                alpha=0.7,
                                                lw=1.5,
                                                mutation_scale=7,
                                            ),
                                            zorder=7,
                                        )
                                    else:
                                        # Drone was stationary — another drone moved into it.
                                        # Mark its position with a red × (no direction to arrow).
                                        ax.scatter(
                                            xs[i],
                                            ys[i],
                                            marker="x",
                                            color="#d62728",
                                            s=60,
                                            linewidths=1.5,
                                            alpha=0.8,
                                            zorder=7,
                                        )
                                except (TypeError, ValueError):
                                    pass
                        else:
                            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                                ax.annotate(
                                    "",
                                    xy=(xs[i + 1], ys[i + 1]),
                                    xytext=(xs[i], ys[i]),
                                    arrowprops=dict(
                                        arrowstyle="-|>",
                                        color=color,
                                        alpha=0.4,
                                        lw=0.5,
                                        mutation_scale=5,
                                    ),
                                    zorder=3,
                                )

                # Mark trajectory start (circle) and end (×).
                ax.scatter(
                    xs[0],
                    ys[0],
                    marker="o",
                    color=color,
                    edgecolors="black",
                    linewidths=0.3,
                    s=28,
                    alpha=0.8,
                    zorder=4,
                )
                last_grid = (int(xs[-1] - 0.5), int(ys[-1] - 0.5))
                if use_capture_markers and last_grid in capture_set:
                    ax.scatter(
                        xs[-1], ys[-1], marker="*", color="yellow", edgecolors="black",
                        linewidths=0.4, s=120, alpha=1.0, zorder=5,
                    )
                else:
                    ax.scatter(
                        xs[-1], ys[-1], marker="x", color=color, s=30, alpha=0.9, zorder=4
                    )

                # Collect shielded positions for overlay after all paths are drawn.
                # Shield fires when trying to move FROM the previous position,
                # so mark position i-1 (fall back to i for the first step).
                for i, ((x, y), shield_type) in enumerate(
                    zip(cleaned, cleaned_shields)
                ):
                    if shield_type in shield_positions:
                        px, py = cleaned[i - 1] if i > 0 else (x, y)
                        shield_positions[shield_type].append((px + 0.5, py + 0.5))

                plotted_any = True

            return plotted_any

        has_evaders = _plot_group(
            evader_episode_states,
            evader_shield_data,
            evader_unsafe_data,
            color="#d95f02",
            use_capture_markers=True,
        )
        has_pursuers = _plot_group(
            pursuer_episode_states,
            pursuer_shield_data,
            pursuer_unsafe_data,
            color="#1f77b4",
        )

        # Overlay shield activation markers on top of trajectories.
        for shield_type, shield_color in SHIELD_COLORS.items():
            pts = shield_positions[shield_type]
            if not pts:
                continue
            sx, sy = zip(*pts)
            ax.scatter(
                sx,
                sy,
                marker="D",
                color=shield_color,
                edgecolors="black",
                linewidths=0.4,
                s=40,
                alpha=0.9,
                zorder=6,
            )

        if not (has_evaders or has_pursuers):
            plt.close(fig)
            print(
                "[HeatmapCallback] No valid trajectories collected, skipping trace map."
            )
            return

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

        ax.tick_params(axis="both", which="major", pad=8)
        ax.set_aspect("equal", adjustable="box")

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
