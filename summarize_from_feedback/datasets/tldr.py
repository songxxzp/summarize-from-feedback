import json
with open("./config.json") as f:
    DATA_PATH = json.load(f)["DATA_PATH"]

from summarize_from_feedback.utils import blobs


def tldr_filtered_generator(split):
    assert split in ["test", "train", "valid"]

    gcs_path = f"{DATA_PATH}/datasets/tldr_3_filtered/{split}.jsonl"
    with blobs.open_file_cached(gcs_path, "rb") as f:
        datas = [json.loads(l) for l in f.readlines()]

    for data in datas:
        yield dict(reference=data["summary"], **{k: v for (k, v) in data.items() if k != "summary"})


def tldr_filtered_queries_generator(split):
    assert split in ["test", "train", "valid"]

    gcs_path = f"{DATA_PATH}/datasets/tldr_3_filtered_queries/{split}.jsonl"
    with blobs.open_file_cached(gcs_path, "rb") as f:
        datas = [json.loads(l) for l in f.readlines()]

    for data in datas:
        # NOTE: don't use ref summary, not filtered
        yield dict(reference=data["summary"], **{k: v for (k, v) in data.items() if k != "summary"})
