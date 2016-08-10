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


def test_data_tuples_to_matrices_simple():

    a = np.ones([5, 1])
    b = 2 * np.ones([10, 1])

    data = [(a, b), (a, b), (a, b)]

    expected_a = np.ones([5, 3])
    expected_b = 2 * np.ones([10, 3])

    matrices = net.utilities.data_tuples_to_matrices(data)

    assert 2 == len(matrices)
    assert np.all(expected_a == matrices[0])
    assert np.all(expected_b == matrices[1])


def test_data_tuples_to_matrices_complex():

    a = np.ones([5, 1])
    b = np.ones([10, 1]).astype(np.int32)
    c = np.ones([8, 1])

    data = [(a, b, c), (2.5 * a, 2 * b, 2 * c), (-0.5 * a, -b, -c), (a, b, c)]

    expected_a = np.zeros([5, 4])
    expected_a[:, 0] = 1.0
    expected_a[:, 1] = 2.5
    expected_a[:, 2] = -0.5
    expected_a[:, 3] = 1.0

    expected_b = np.zeros([10, 4]).astype(np.int32)
    expected_b[:, 0] = 1
    expected_b[:, 1] = 2
    expected_b[:, 2] = -1
    expected_b[:, 3] = 1

    expected_c = np.zeros([8, 4])
    expected_c[:, 0] = 1.0
    expected_c[:, 1] = 2.0
    expected_c[:, 2] = -1.0
    expected_c[:, 3] = 1.0

    matrices = net.utilities.data_tuples_to_matrices(data)

    assert 3 == len(matrices)
    assert np.all(expected_a == matrices[0])
    assert np.all(expected_b == matrices[1])
    assert np.all(expected_c == matrices[2])


def test_remove_visually_identical_characters_trivial():

    characters = ['a', 'b', 'c']

    assert characters == net.utilities.remove_visually_identical_characters(characters)


def test_remove_visually_identical_characters_trivial_repetitions():

    characters = ['a', 'b', 'c', 'a']
    expected = ['a', 'b', 'c']

    assert expected == net.utilities.remove_visually_identical_characters(characters)


def test_remove_visually_identical_characters_hiragana_katakana_repetitions_1():

    # Please note first and last characters are in fact different characters
    # with different unicodes. They just look the same
    characters = ['ぺ', 'a', 'b', 'c', 'ペ']
    expected = ['ぺ', 'a', 'b', 'c']

    assert expected == net.utilities.remove_visually_identical_characters(characters)


def test_remove_visually_identical_characters_hiragana_katakana_repetitions_2():

    # Please note first and last characters are in fact different characters
    # with different unicodes. They just look the same
    characters = ['ぺ', 'a', 'べ', 'b', 'c', 'ペ']
    expected = ['ぺ', 'a', 'べ', 'b', 'c']

    assert expected == net.utilities.remove_visually_identical_characters(characters)
