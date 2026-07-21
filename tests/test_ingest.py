import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import to_float_or_none


def test_to_float_or_none_valid_string():
    assert to_float_or_none("3.14") == 3.14


def test_to_float_or_none_valid_number():
    assert to_float_or_none(2) == 2.0


def test_to_float_or_none_none_input():
    assert to_float_or_none(None) is None


def test_to_float_or_none_garbage_string():
    assert to_float_or_none("None") is None
    assert to_float_or_none("N/A") is None
    assert to_float_or_none("") is None
