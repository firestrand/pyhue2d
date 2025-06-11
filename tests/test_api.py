import pytest
import pyhue2d
import os
import json

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'example_images')
MANIFEST_PATH = os.path.join(EXAMPLES_DIR, 'examples_manifest.json')

with open(MANIFEST_PATH) as f:
    EXAMPLES = json.load(f)

@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.xfail(reason="Encode not yet implemented.")
def test_encode_manifest_examples(example):
    with pytest.raises(NotImplementedError):
        pyhue2d.encode(example["text"].encode("utf-8"))

@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.xfail(reason="Decode not yet implemented.")
def test_decode_manifest_examples(example):
    image_path = os.path.join(EXAMPLES_DIR, example["output"])
    with pytest.raises(NotImplementedError):
        pyhue2d.decode(image_path)

def test_encode_not_implemented():
    with pytest.raises(NotImplementedError):
        pyhue2d.encode(b"data")


def test_decode_not_implemented():
    with pytest.raises(NotImplementedError):
        pyhue2d.decode("image.png")
