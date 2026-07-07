
class FunctionsForPolarizationOptics:

    @staticmethod
    def jones_to_stokes(jones_vector):
        """
        Convert a Jones vector to Stokes parameters.

        Parameters:
        jones_vector (list or array): A 2-element list or array representing the Jones vector [E_x, E_y].

        Returns:
        list: A list of Stokes parameters [S0, S1, S2, S3].
        """
        import numpy as np

        E_x, E_y = jones_vector
        I = np.abs(E_x)**2 + np.abs(E_y)**2
        Q = np.abs(E_x)**2 - np.abs(E_y)**2
        U = 2 * np.real(E_x * np.conj(E_y))
        V = 2 * np.imag(E_x * np.conj(E_y))

        return [I, Q, U, V]

    @staticmethod
    def JDelay(theta, delta):
        """
        Calculate the Jones matrix for a waveplate with a given retardance and orientation.

        Parameters:
        theta (float): The angle of the fast axis of the waveplate in radians.  
        delta (float): The retardance of the waveplate in radians.
        Returns:
        2D array: The Jones matrix representing the waveplate.
        """
        import numpy as np

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_delta = np.cos(delta)
        sin_delta = np.sin(delta)

        J = np.array([[cos_theta**2 + sin_theta**2 * cos_delta,
                       cos_theta * sin_theta * (1 - cos_delta) - 1j * sin_theta * sin_delta],
                      [cos_theta * sin_theta * (1 - cos_delta) + 1j * sin_theta * sin_delta,
                       sin_theta**2 + cos_theta**2 * cos_delta]])
        return J

    @staticmethod
    def MDelay(theta, delta):
        """
        Calculate the Mueller matrix for a waveplate with a given retardance and orientation.
        Parameters:
        theta (float): The angle of the fast axis of the waveplate in radians.
        delta (float): The retardance of the waveplate in radians.
        Returns:
        2D array: The Mueller matrix representing the waveplate.
        """
        import numpy as np

        cos_2theta = np.cos(2 * theta)
        sin_2theta = np.sin(2 * theta)
        cos_delta = np.cos(delta)
        sin_delta = np.sin(delta)

        M = np.array([[1, 0, 0, 0],
                      [0, cos_2theta**2 + sin_2theta**2 * cos_delta,
                       cos_2theta * sin_2theta * (1 - cos_delta),
                       -sin_2theta * sin_delta],
                      [0, cos_2theta * sin_2theta * (1 - cos_delta),
                       sin_2theta**2 + cos_2theta**2 * cos_delta,
                       cos_2theta * sin_delta],
                      [0, sin_2theta * sin_delta,
                       -cos_2theta * sin_delta,
                       cos_delta]])
        return M