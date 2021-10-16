from typing import List


def predict(input_file_path: str) -> List[str]:
    """
    Given a path to a json file with unlabeled sentences it returns a list with the
    predicted classes of each of the sentences in the same order as they were in the file.

    Example:
    predict("some/page/file/with3sentences.json") -> ["section", "paragraph", "title"]
    """
    pass
