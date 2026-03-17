import torch
from pydantic import BaseModel, PrivateAttr

from dynopsi.model.dyno_psi_model import DynoPsiModel
from dynopsi.model.loaders import HuggingFaceLoader


class DynoPsiModelConfig(BaseModel):
    """Configuration for the DynoPsi model, allowing serialization/deserialization
    and lazy loading of the model.

    Attributes
    ----------
    repo_id : str
        The Hugging Face repository ID of the model.
    filename : str
        The filename of the model.
    device_name : str
        The name of the device to load the model on. (default: "cpu")
    use_self_conditioning : bool
        Whether to use self-conditioning for the model. (default: True)
    cvf_eps : float
        The epsilon value for prediction using the model. (default: 1e-6)
    """

    repo_id: str
    filename: str
    device_name: str = "cpu"
    use_self_conditioning: bool = True
    cvf_eps: float = 1e-6

    _dynopsi_model: DynoPsiModel | None = PrivateAttr(default=None)

    @property
    def dynopsi_model(self) -> DynoPsiModel:
        if self._dynopsi_model is None:
            self._dynopsi_model = HuggingFaceLoader(self.repo_id, self.filename).load(self.device_name)
        assert self._dynopsi_model is not None
        return self._dynopsi_model

    def to_device(self, device: torch.device) -> None:
        self.dynopsi_model.to_device(device)
        self.device_name = str(device)


DEFAULT_DYNOPSI_MODEL_CONFIG = DynoPsiModelConfig(
    repo_id="dynotx/dynopsi",
    filename="dynopsi-1.ckpt",
)
