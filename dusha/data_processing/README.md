## Raw data processing

To process data download a raw Dusha dataset (crowd.tar, podcast.tar), untar it to DATASET_PATH, and run the processing script:

    python processing.py -dataset_path  DATASET_PATH 

It processes sound files and creates a folder in DATASET_PATH with precalculated features, aggregates labels, and creates manifest file in jsonl format.


If you want to change the threshold for aggregation run the processing with -threshold flag:

    python processing.py  -dataset_path  DATASET_PATH -threshold THRESHOLD

You can also use tsv format for manifest file:

    python processing.py -dataset_path  DATASET_PATH -tsv  

Force recalculate features:

    python processing.py  -dataset_path  DATASET_PATH  -rf
