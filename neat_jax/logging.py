from typing import Any, Callable, Dict


def render_and_log_episode(
    wandb_run,
    prev_stats: Dict[str, Any],
    metrics: Dict[str, Any],
    episode_data: Dict[str, Any],
    render_fn: Callable,
    generation_num: int,
) -> None:
    print(f"Rendering and logging generation {generation_num}")
    rendered = render_fn(episode_data)
    metrics["rendered"] = rendered
    wandb_run.log(prev_stats, step=generation_num)
    wandb_run.log(metrics, step=generation_num)


def log_to_wandb(
    wandb_run,
    prev_stats: Dict[str, Any],
    metrics: Dict[str, Any],
    generation_num: int,
) -> None:
    wandb_run.log(prev_stats, step=generation_num)
    wandb_run.log(metrics, step=generation_num)
