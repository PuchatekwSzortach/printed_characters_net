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


def test_get_data_batches_exact_batches_cut():

    data = list(range(12))
    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

    assert expected == net.utilities.get_data_batches(data, 3)


def test_get_data_batches_leftover_elements():

    data = list(range(12))
    expected = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    assert expected == net.utilities.get_data_batches(data, 5)