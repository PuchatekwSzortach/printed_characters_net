import numpy as np
import net.utilities

class TestEncoder:

    def test_encoding(self):

        labels = ["A", "B", "C", "D"]
        encoder = net.utilities.Encoder(labels)

        assert np.all(np.array([0, 1, 0, 0]).reshape(4, 1) == encoder.encode("B"))
        assert np.all(np.array([0, 0, 0, 1]).reshape(4, 1) == encoder.encode("D"))

    def test_decoding(self):

        labels = ["A", "B", "C", "D"]
        encoder = net.utilities.Encoder(labels)

        assert "A" == encoder.decode([1, 0, 0, 0])
        assert "D" == encoder.decode([0, 0, 0, 1])
