import pytest
import pyhue2d
import os
import json
from PIL import Image

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'example_images')
MANIFEST_PATH = os.path.join(EXAMPLES_DIR, 'examples_manifest.json')

with open(MANIFEST_PATH) as f:
    EXAMPLES = json.load(f)

@pytest.mark.parametrize("example", EXAMPLES)
def test_encode_manifest_examples(example):
    """Test that core encode function can encode manifest examples to images."""
    result = pyhue2d.encode(example["text"].encode("utf-8"))
    assert isinstance(result, Image.Image)
    assert result.size[0] > 0 and result.size[1] > 0

@pytest.mark.parametrize("example", EXAMPLES)  
def test_decode_manifest_examples(example):
    """Test that core decode function raises NotImplementedError for manifest examples."""
    image_path = os.path.join(EXAMPLES_DIR, example["output"])
    with pytest.raises(NotImplementedError):
        pyhue2d.decode(image_path)

def test_encode_basic_functionality():
    """Test basic encode functionality."""
    result = pyhue2d.encode(b"test data")
    assert isinstance(result, Image.Image)
    assert result.size[0] > 0 and result.size[1] > 0


def test_decode_not_implemented():
    with pytest.raises(NotImplementedError):
        pyhue2d.decode("image.png")
