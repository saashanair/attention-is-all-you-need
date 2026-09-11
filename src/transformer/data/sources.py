from datasets import DatasetDict, load_dataset, load_from_disk

from ..paths import project_root


def load_data_from_hf(dataset_name: str) -> DatasetDict:
    local_path = project_root() / 'raw' / dataset_name.replace('/', '_')
    if local_path.exists():
        dataset = load_from_disk(local_path)
    else:
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(local_path)

    # load_from_disk's return type depends on what's on disk, which isn't statically knowable;
    # this codebase only ever caches what load_dataset (called with no split) produces, so it
    # should always be a DatasetDict -- this both documents and enforces that assumption.
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f'expected a DatasetDict for {dataset_name!r}, got {type(dataset).__name__}')

    return dataset
