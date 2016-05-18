import numpy as np
import net.network

def test_sigmoid():

    z = np.array([-1000, 0, 0, -1000])
    expected = np.array([0, 0.5, 0.5, 0])

    assert np.all(expected == net.network.sigmoid(z))