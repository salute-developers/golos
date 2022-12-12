from pathlib import Path
from typing import List

import pandas as pd
from crowdkit.aggregation import DawidSkene as CrowdKitDawidSkene
from utils.datacls import DawidSkeneEntryDataclass, DawidSkeneResultDataclass


def get_dawidskene_pred(
    data: List[DawidSkeneEntryDataclass],
    threshold: float,
    meta_path: Path,
    n_iter: int = 100,
) -> List[DawidSkeneResultDataclass]:
    labels = {row.label for row in data}
    assert "task" not in labels, 'Labels cant contains the name "task"!'
    aggregated_labels = CrowdKitDawidSkene(n_iter=n_iter).fit_predict_proba(
        pd.DataFrame(data)
    )
    aggregated_labels.to_csv(meta_path, sep="\t")

    aggregated_labels_list = aggregated_labels.reset_index().to_dict("records")
    aggregated_data = []
    for row in aggregated_labels_list:
        tmp_dict = {val: key for key, val in row.items() if key in labels}
        max_item_proba = max(tmp_dict)
        if max_item_proba >= threshold:
            key_with_max_value = tmp_dict[max_item_proba]
            aggregated_row = DawidSkeneResultDataclass(
                task=row["task"],
                pred=key_with_max_value,
            )
            aggregated_data.append(aggregated_row)
    return aggregated_data
