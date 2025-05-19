import pytorch_lightning.trainer.connectors.callback_connector
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from lightning_utilities.core.imports import RequirementCache
from pytorch_lightning.callbacks import ModelCheckpoint

def _configure_checkpoint_callbacks(self, enable_checkpointing: bool) -> None:
    if self.trainer.checkpoint_callbacks:
        if not enable_checkpointing:
            raise MisconfigurationException(
                "Trainer was configured with `enable_checkpointing=False`"
                " but found `ModelCheckpoint` in callbacks list."
            )
    elif enable_checkpointing:
        if RequirementCache("litmodels >=0.1.7") and self.trainer._model_registry:
            raise NotImplementedError("SBL: I have disabled this for now. If needed, switch back to pytorch_lightning's original implementation")
        else:
            # rank_zero_info( SBL: Silencing this annoying thing
            #     "Using default `ModelCheckpoint`. Consider installing `litmodels` package to enable"
            #     " `LitModelCheckpoint` for automatic upload to the Lightning model registry."
            # )
            model_checkpoint = ModelCheckpoint()
        self.trainer.callbacks.append(model_checkpoint)

pytorch_lightning.trainer.connectors.callback_connector._CallbackConnector._configure_checkpoint_callbacks = _configure_checkpoint_callbacks