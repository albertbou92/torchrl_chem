import logging
import tarfile
from functools import partial
from importlib import import_module, resources
from pathlib import Path

from acegen.models.gpt2 import (
    create_gpt2_actor,
    create_gpt2_actor_critic,
    create_gpt2_critic,
)

from acegen.models.gru import (
    create_gru_actor,
    create_gru_actor_critic,
    create_gru_critic,
)
from acegen.models.llama2 import (
    create_llama2_actor,
    create_llama2_actor_critic,
    create_llama2_critic,
)
from acegen.models.lstm import (
    create_lstm_actor,
    create_lstm_actor_critic,
    create_lstm_critic,
)
from acegen.models.mamba import (
    create_mamba_actor,
    create_mamba_actor_critic,
    create_mamba_critic,
)
from acegen.models.utils import adapt_state_dict
from acegen.vocabulary.tokenizers import (
    AsciiSMILESTokenizer,
    SMILESTokenizerChEMBL,
    SMILESTokenizerEnamine,
    SMILESTokenizerGuacaMol,
)


def extract(path):
    """Extract tarfile if it exists."""
    if not path.exists():
        tar_path = path.with_suffix(".tar.gz")
        if tar_path.exists():
            logging.info("Extracting model checkpoint...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=tar_path.parent)
                return path
        else:
            raise FileNotFoundError(f"File {path} not found.")
    else:
        return path


# extracting big model files only if they are not already extracted
if not (resources.files("acegen.priors") / "gpt2_enamine_real.ckpt").exists():
    extract(resources.files("acegen.priors") / "gpt2_enamine_real.ckpt")


models = {
    "gru": (
        create_gru_actor,
        create_gru_critic,
        create_gru_actor_critic,
        resources.files("acegen.priors") / "chembl_filtered_vocabulary.txt",
        resources.files("acegen.priors") / "gru_chembl_filtered.ckpt",
        SMILESTokenizerChEMBL(),
    ),
    "gru_chembl34": (
        create_gru_actor,
        create_gru_critic,
        create_gru_actor_critic,
        resources.files("acegen.priors") / "gru_chembl34_vocabulary.ckpt",
        resources.files("acegen.priors") / "gru_chembl34.ckpt",
        SMILESTokenizerChEMBL(),
    ),
    "lstm": (
        create_lstm_actor,
        create_lstm_critic,
        create_lstm_actor_critic,
        resources.files("acegen.priors") / "chembl_vocabulary.txt",
        resources.files("acegen.priors") / "lstm_chembl.ckpt",
        SMILESTokenizerChEMBL(),
    ),
    "lstm_guacamol": (
        partial(create_lstm_actor, embedding_size=1024, hidden_size=1024, dropout=0.2),
        partial(create_lstm_critic, embedding_size=1024, hidden_size=1024, dropout=0.2),
        partial(
            create_lstm_actor_critic, embedding_size=1024, hidden_size=1024, dropout=0.2
        ),
        resources.files("acegen.priors") / "lstm_guacamol_vocabulary.txt",
        resources.files("acegen.priors")
        / "lstm_guacamol_model_final_0.473_acegen.ckpt",
        SMILESTokenizerGuacaMol(),
    ),
    "gpt2": (
        create_gpt2_actor,
        create_gpt2_critic,
        create_gpt2_actor_critic,
        resources.files("acegen.priors") / "enamine_real_vocabulary.txt",
        resources.files("acegen.priors") / "gpt2_enamine_real.ckpt",
        SMILESTokenizerEnamine(),
    ),
    "llama2": (
        create_llama2_actor,
        create_llama2_critic,
        create_llama2_actor_critic,
        resources.files("acegen.priors") / "ascii.pt",
        resources.files("acegen.priors") / "llama2_enamine_real_6B.ckpt",
        AsciiSMILESTokenizer(),
    ),
    "mamba": (
        create_mamba_actor,
        create_mamba_critic,
        create_mamba_actor_critic,
        resources.files("acegen.priors") / "mamba_chembl_filtered_vocabulary.ckpt",
        resources.files("acegen.priors") / "mamba_chembl_filtered.ckpt",
        SMILESTokenizerChEMBL(),
    ),
}


def register_model(name, factory):
    """Register a model factory.

    The factory can be a function or a string in the form "module.factory".
    Running the factory should return a tuple with the following elements:
    - create_actor: a function that creates the actor model
    - create_critic: a function that creates the critic model (optional, otherwise use None)
    - create_actor_critic: a function that creates the actor-critic model (optional, otherwise use None)
    - vocabulary: a path to the vocabulary file
    - checkpoint: a path to the model checkpoint
    - tokenizer: a tokenizer instance (optional, otherwise use None)

    For more details, see the tutorial in acegen-open/tutorials/adding_custom_model.md.
    """
    if isinstance(factory, str):
        m, f = factory.rsplit(".", 1)
        factory = getattr(import_module(m), f)
    models[name] = factory
