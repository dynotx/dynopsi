import torch

type ValidSteps = dict[str, list[int]]


def all_steps_for_modalities(modalities: list[str], num_steps: int) -> ValidSteps:
    """
    Build ValidSteps where each modality is active at every step in [0, num_steps).

    Parameters
    ----------
    modalities : list[str]
        Modality keys (e.g. ["coords"]).
    num_steps : int
        Number of integration steps.

    Returns
    -------
    ValidSteps
        dict mapping each modality to list(range(num_steps)).
    """
    return {modality: list(range(num_steps)) for modality in modalities}


def valid_modalities_at(valid_steps: ValidSteps, step_num: int) -> list[str]:
    """
    Return modalities that are active at the given step.

    Parameters
    ----------
    valid_steps : ValidSteps
        Modality -> list of step indices.
    step_num : int
        Current step index.

    Returns
    -------
    list[str]
        Modalities for which step_num is in valid_steps[modality].
    """
    return [modality for modality in valid_steps if step_num in valid_steps[modality]]


def is_active_at(valid_steps: ValidSteps, step_num: int) -> bool:
    """
    Return True if any modality is active at the given step.

    Parameters
    ----------
    valid_steps : ValidSteps
        Modality -> list of step indices.
    step_num : int
        Current step index.

    Returns
    -------
    bool
        True if there exists a modality with step_num in valid_steps[modality].
    """
    return any(step_num in valid_steps[modality] for modality in valid_steps)


def infer_batch_size(device: torch.device) -> int:
    """Infer a batch size that fits on one device."""
    if device.type == "cpu":
        return 1

    device_memory = torch.cuda.get_device_properties(device).total_memory
    device_memory_gb = device_memory / 1e9

    if device_memory_gb >= 80:  # H100/A100 80GB
        return 5
    if device_memory_gb >= 40:  # A100 40GB
        return 2
    return 1
