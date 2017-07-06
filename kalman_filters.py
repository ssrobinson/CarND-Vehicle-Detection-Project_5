import numpy as np


def kalman_predict(x, P, F, Q):
    """
    Prediction step of Kalman Filter
    Propagates the dynamics of the system
    Parameters
    
    z : array_like
        Measurement vector
    x : array_like
        Last state vector
    P : numpy.array
        Covariance matrix
    F : numpy.array
        The linear system
    Q : numpy.array
        Process noise
    """
    # prediction
    x = x + F @ x
    P = F @ P @ F.T + Q

    return x, P

def kalman_correct(z, x, P, H, R):
    """
    Performs the correction step of the kalman filter
    H : numpy.array
        Measurement-State relationship matrix
    R : numpy.array
        Measurement covariance
    """
    I = np.eye(x.shape[0])

    # measurement update
    Z = np.array(z)
    y = Z.T - (H @ x)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + (K @ y)
    P = (I - (K @ H)) @ P

    return x, P


def extended_kalman_filter(z, x, P, u, F_fn, x_fn, H, R, Q):
    """
    Applies extended kalman filter on system
    Parameters
    ----------
    z : array_like
        Measurement vector
    x : array_like
        Last state vector
    u : array_like
        control vector
    P : numpy.array
        Covariance matrix
    F_fn : function
        Function that returns F matrix for given 'x'
    x_fn : function
        : Updates 'x' using the non-linear derivatives
    H : numpy.array
        Measurement-State relationship matrix
    R : numpy.array
        Measurement covariance
    Q : numpy.array
        Process noise
    """
    I = np.eye(x.shape[0])
    # prediction
    F = F_fn(x)
    x = x_fn(x) + u
    P = F @ P @ F.T + Q

    if z is not None:
        # measurement update
        Z = matrix([z])
        y = Z.T - (H * x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ y)
        P = (I - (K @ H)) @ P

    return x, P
