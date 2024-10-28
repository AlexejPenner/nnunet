import os
import shutil

from zenml import register_artifact, load_artifact
from zenml import step, pipeline
from zenml.client import Client

from nnunet_v1_to_v2 import nnunet_v1_to_v2

V1_DATSET_NAME = "v1_nnunet_dataset"
V2_DATSET_NAME = "v2_nnunet_dataset"


@step(enable_cache=True)
def load_v1():
    """Loads dataset from outside artifact store bounds into artifact store."""
    client = Client()
    dataset_rel_path = "v1data"
    as_path = client.active_stack.artifact_store.path
    # Define path within artifact store bounds
    dataset_path_in_as = os.path.join(as_path, V1_DATSET_NAME)

    # copy dataset into artifact store
    shutil.copytree(dataset_rel_path, dataset_path_in_as, dirs_exist_ok=True)

    try:
        # create artifact from the preexisting folder
        registered_artifact = register_artifact(
            folder_or_file_uri=dataset_path_in_as,
            name=V1_DATSET_NAME
        )
    except RuntimeError:
        # In case the artifact already existed
        registered_artifact = client.get_artifact_version(V1_DATSET_NAME)

    return str(registered_artifact.name)


@step(enable_cache=True)
def convert_v1_to_v2(input_artifact_name: str):
    # load dataset
    source_dataset = load_artifact(name_or_id=input_artifact_name)

    # define destination path within artifact store bounds
    client = Client()
    as_path = client.active_stack.artifact_store.path
    dataset_as_path = os.path.join(as_path, V2_DATSET_NAME)

    # convert v1 dataset into v2 dataset
    nnunet_v1_to_v2(source_dataset, dataset_as_path)

    try:
        # create artifact from the created v2 folder
        registered_artifact = register_artifact(
            folder_or_file_uri=dataset_as_path,
            name=V2_DATSET_NAME
        )
    except RuntimeError:
        # In case the artifact already existed
        registered_artifact = client.get_artifact_version(V2_DATSET_NAME)

    return str(registered_artifact.name)


@step(enable_cache=False)
def train_model(training_dataset: str):
    print(training_dataset)
    return


@pipeline
def convert_and_train():
    v1_dataset = load_v1()
    v2_dataset = convert_v1_to_v2(v1_dataset)
    train_model(v2_dataset)


if __name__ == "__main__":
    convert_and_train()
