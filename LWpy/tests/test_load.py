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

    def test_read_write(self):
        s = LWpy.read_stream('./config_DUNE.lic')
        blocks = s.read()
        sw = LWpy.write_stream('./test_config.lic')
        sw.write(blocks)
        ss = LWpy.read_stream('./test_config.lic')
        new_blocks = ss.read()
        assert(len(new_blocks) == len(blocks))
        for i in range(len(blocks)):
            assert(LWpy.block_equal(blocks[i], new_blocks[i]))

if __name__ == '__main__':
    unittest.main()
