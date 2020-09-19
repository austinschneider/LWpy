from context import LWpy
import unittest

class LicStreamLoad(unittest.TestCase):
    """Basic test cases."""

    def test_load(self):
        s = LWpy.read_stream('./config_DUNE.lic')
        blocks = s.read()
        assert(len(blocks) == 2)

    def test_merge_null(self):
        s = LWpy.read_stream('./config_DUNE.lic')
        blocks = s.read()
        blocks = LWpy.merge_blocks(blocks)
        assert(len(blocks) == 2)

if __name__ == '__main__':
    unittest.main()
