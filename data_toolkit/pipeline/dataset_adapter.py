import os

import pandas as pd


def process_single_metadata_row(
    dataset_utils,
    metadata,
    output_dir,
    func,
    *,
    requires_local_path=True,
):
    """Run one adapter row directly without creating a nested executor."""
    records = metadata.to_dict("records")
    if len(records) != 1:
        raise ValueError("single-row adapter processing requires one metadata row")

    metadatum = records[0]
    processor = getattr(dataset_utils, "_process_instance", None)
    if not requires_local_path:
        record = func(None, metadatum["sha256"])
    elif processor is not None:
        record = processor((metadatum, output_dir, func))
    else:
        try:
            file_path = os.path.join(output_dir, metadatum["local_path"])
            record = func(file_path, metadatum["sha256"])
        except Exception as error:
            print(
                f"Error processing object {metadatum.get('sha256', '?')}: "
                f"{error}"
            )
            record = None

    if record is None:
        return pd.DataFrame()
    if isinstance(record, pd.DataFrame):
        return record
    return pd.DataFrame.from_records([record])
