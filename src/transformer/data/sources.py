from datasets import load_dataset, load_from_disk, DatasetDict
from ..paths import project_root


def load_data_from_hf(dataset_name: str) -> DatasetDict:
    local_path = project_root() / 'raw' / dataset_name.replace('/', '_')
    if local_path.exists():
        return load_from_disk(local_path)
    dataset = load_dataset(dataset_name)
    dataset.save_to_disk(local_path)
    return dataset
