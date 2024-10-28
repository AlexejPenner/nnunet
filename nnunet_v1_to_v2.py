import json
import os
import shutil
from copy import deepcopy

def nnunet_v1_to_v2(source_folder: str, target_dataset_name: str) -> None:
    """
    Convert old nnUNet dataset format to new format.
    Old tasks were called TaskXXX_YYY and new ones are called DatasetXXX_YYY.

    :param source_folder: Path to the source folder containing the old format dataset
    :param target_dataset_name: Name of the target dataset in the new format
    """
    target_path = os.path.join("", target_dataset_name)

    if os.path.isdir(target_path):
        raise RuntimeError(f'Target dataset name {target_dataset_name} already exists. Aborting... '
                           f'(we might break something). If you are sure you want to proceed, please manually '
                           f'delete {target_path}')

    try:
        # Create target directory
        os.makedirs(target_path)

        for dataset in os.listdir(source_folder):
            # Copy folders
            for folder in ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs', 'imagesVal', 'labelsVal']:
                src = os.path.join(source_folder, dataset, folder)
                dst = os.path.join(target_path, dataset, folder)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)

            # Copy and modify dataset.json
            json_src = os.path.join(source_folder, dataset, 'dataset.json')
            json_dst = os.path.join(target_path, dataset, 'dataset.json')

            with open(json_src, 'r') as f:
                dataset_json = json.load(f)

            # Modify dataset.json
            keys_to_remove = ['tensorImageSize', 'numTest', 'training', 'test']
            for key in keys_to_remove:
                dataset_json.pop(key, None)

            dataset_json['channel_names'] = deepcopy(dataset_json.pop('modality', {}))
            dataset_json['labels'] = {j: int(i) for i, j in dataset_json['labels'].items()}
            dataset_json['file_ending'] = ".nii.gz"

            with open(json_dst, 'w') as f:
                json.dump(dataset_json, f, indent=4)

            print(f"Successfully converted dataset to {target_path}")

    except Exception as e:
        print(f"An error occurred during conversion: {str(e)}")
        # If an error occurs, attempt to remove the partially created target directory
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        raise e

if __name__ == "__main__":
    src = "v1data/Task01_BrainTumour"
    trg = "v2data/Dataset001_BrainTumour"
    nnunet_v1_to_v2(src, trg)
