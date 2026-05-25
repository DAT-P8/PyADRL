from datetime import datetime
from pydantic import BaseModel


class EvaluationResult(BaseModel):
    timestamp: datetime
    capture_rate_at_k: dict[str, float]
    mean_capture_step_at_k: list[float]
    mean_rewards: dict[str, float]
    mean_capture_step: float
    breach_rate: float
    mean_episode_length: float
    mean_evader_drone_collision_rate: float
    mean_pursuer_drone_collision_rate: float
    mean_evader_obstacle_collision_rate: float
    mean_pursuer_obstacle_collision_rate: float
    mean_evader_out_of_bounds_rate: float
    mean_pursuer_out_of_bounds_rate: float
    evader_shield_intervention_rate: float
    pursuer_shield_intervention_rate: float
    mean_pursuer_entered_target_rate: float
    capture_score: float
    weighted_acs: float
    score_p: float
    comb_score: float


class EvaluationPoolMetrics(BaseModel):
    experiment_name: str

    pursuer_config: str
    pursuer_training: str

    evader_config: str
    evader_training: str

    metrics: list[EvaluationResult]


def combine(results: list[EvaluationResult]) -> EvaluationResult:
    if len(results) == 1:
        return results[0]

    m_mean_capture_step = 0
    m_breach_rate = 0
    m_mean_episode_length = 0
    m_mean_evader_drone_collision_rate = 0
    m_mean_pursuer_drone_collision_rate = 0
    m_mean_evader_obstacle_collision_rate = 0
    m_mean_pursuer_obstacle_collision_rate = 0
    m_mean_evader_out_of_bounds_rate = 0
    m_mean_pursuer_out_of_bounds_rate = 0
    m_evader_shield_intervention_rate = 0
    m_pursuer_shield_intervention_rate = 0
    m_mean_pursuer_entered_target_rate = 0
    m_capture_score = 0
    m_weighted_acs = 0
    m_score_p = 0
    m_comb_score = 0

    for result in results:
        m_mean_capture_step += result.mean_capture_step
        m_breach_rate += result.breach_rate
        m_mean_episode_length += result.mean_episode_length
        m_mean_evader_drone_collision_rate += result.mean_evader_drone_collision_rate
        m_mean_pursuer_drone_collision_rate += result.mean_pursuer_drone_collision_rate
        m_mean_evader_obstacle_collision_rate += (
            result.mean_evader_obstacle_collision_rate
        )
        m_mean_pursuer_obstacle_collision_rate += (
            result.mean_pursuer_obstacle_collision_rate
        )
        m_mean_evader_out_of_bounds_rate += result.mean_evader_out_of_bounds_rate
        m_mean_pursuer_out_of_bounds_rate += result.mean_pursuer_out_of_bounds_rate
        m_evader_shield_intervention_rate += result.evader_shield_intervention_rate
        m_pursuer_shield_intervention_rate += result.pursuer_shield_intervention_rate
        m_mean_pursuer_entered_target_rate += result.mean_pursuer_entered_target_rate
        m_capture_score += result.capture_score
        m_weighted_acs += result.weighted_acs
        m_score_p += result.score_p
        m_comb_score += result.comb_score

    n = len(results)
    m_mean_capture_step = m_mean_capture_step / n
    m_breach_rate = m_breach_rate / n
    m_mean_episode_length = m_mean_episode_length / n
    m_mean_evader_drone_collision_rate = m_mean_evader_drone_collision_rate / n
    m_mean_pursuer_drone_collision_rate = m_mean_pursuer_drone_collision_rate / n
    m_mean_evader_obstacle_collision_rate = m_mean_evader_obstacle_collision_rate / n
    m_mean_pursuer_obstacle_collision_rate = m_mean_pursuer_obstacle_collision_rate / n
    m_mean_evader_out_of_bounds_rate = m_mean_evader_out_of_bounds_rate / n
    m_mean_pursuer_out_of_bounds_rate = m_mean_pursuer_out_of_bounds_rate / n
    m_evader_shield_intervention_rate = m_evader_shield_intervention_rate / n
    m_pursuer_shield_intervention_rate = m_pursuer_shield_intervention_rate / n
    m_mean_pursuer_entered_target_rate = m_mean_pursuer_entered_target_rate / n
    m_capture_score = m_capture_score / n
    m_weighted_acs = m_weighted_acs / n
    m_score_p = m_score_p / n
    m_comb_score = m_comb_score / n

    n_capture_rate_at_k: dict[str, int] = {}
    m_capture_rate_at_k: dict[str, float] = {}

    n_mean_capture_step_at_k: list[tuple[float, int]] = []
    m_mean_capture_step_at_k: list[float] = []

    n_mean_rewards: dict[str, int] = {}
    m_mean_rewards: dict[str, float] = {}

    for result in results:
        for key, value in result.capture_rate_at_k.items():
            if key not in n_capture_rate_at_k:
                n_capture_rate_at_k[key] = 0
            if key not in m_capture_rate_at_k:
                m_capture_rate_at_k[key] = 0

            m_capture_rate_at_k[key] += value
            n_capture_rate_at_k[key] += 1

        for key, value in result.mean_rewards.items():
            if key not in n_mean_rewards:
                n_mean_rewards[key] = 0
            if key not in m_mean_rewards:
                m_mean_rewards[key] = 0

            m_mean_rewards[key] += value
            n_mean_rewards[key] += 1

        for key, value in enumerate(result.mean_capture_step_at_k):
            while len(n_mean_capture_step_at_k) - 1 < key:
                n_mean_capture_step_at_k.append((0, 0))

            summed, count = n_mean_capture_step_at_k[key]
            n_mean_capture_step_at_k[key] = (summed + value, count + 1)

    for key, value in m_capture_rate_at_k.items():
        m_capture_rate_at_k[key] = value / n_capture_rate_at_k[key]

    for key, value in m_mean_rewards.items():
        m_mean_rewards[key] = value / n_mean_rewards[key]

    for summed, count in n_mean_capture_step_at_k:
        if count == 0:
            m_mean_capture_step_at_k.append(-1)
        else:
            m_mean_capture_step_at_k.append(summed / count)

    return EvaluationResult(
        timestamp=datetime.now(),
        capture_rate_at_k=m_capture_rate_at_k,
        mean_capture_step_at_k=m_mean_capture_step_at_k,
        mean_rewards=m_mean_rewards,
        mean_capture_step=m_mean_capture_step,
        breach_rate=m_breach_rate,
        mean_episode_length=m_mean_episode_length,
        mean_evader_drone_collision_rate=m_mean_evader_drone_collision_rate,
        mean_pursuer_drone_collision_rate=m_mean_pursuer_drone_collision_rate,
        mean_evader_obstacle_collision_rate=m_mean_evader_obstacle_collision_rate,
        mean_pursuer_obstacle_collision_rate=m_mean_pursuer_obstacle_collision_rate,
        mean_evader_out_of_bounds_rate=m_mean_evader_out_of_bounds_rate,
        mean_pursuer_out_of_bounds_rate=m_mean_pursuer_out_of_bounds_rate,
        evader_shield_intervention_rate=m_evader_shield_intervention_rate,
        pursuer_shield_intervention_rate=m_pursuer_shield_intervention_rate,
        mean_pursuer_entered_target_rate=m_mean_pursuer_entered_target_rate,
        capture_score=m_capture_score,
        weighted_acs=m_weighted_acs,
        score_p=m_score_p,
        comb_score=m_comb_score,
    )
