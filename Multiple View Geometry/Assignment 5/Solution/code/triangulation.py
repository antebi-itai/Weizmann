import numpy as np
import cvxpy as cp
import utils
import tqdm


def DLT_triangulation(P, x, visible_points):
    """
    Use direct linear transformation in order to triangulate the points.
    :param P: ndarray of shape [n_cam, 3, 4], the cameras
    :param x: ndarray of shape [n_cam, 3, n_points], the projected image points
    :param visible_points: boolean matrix of shape [n_cam, n_points], what cameras see what points
    :return: X ndarray of shape [4, n_points], the predicted 3D points
    """
    n_points = x.shape[-1]
    X = np.zeros((4, n_points))
    for point_idx in range(n_points):
        # take only relevant cameras and 2D points
        visible_cameras = visible_points[:,point_idx]
        curr_P = P[visible_cameras]
        curr_x = x[visible_cameras, :, point_idx]
        # compute the matrix for DLT
        n_cameras = curr_P.shape[0]
        M = np.zeros((3*n_cameras, 4+n_cameras))
        M[:, 0:4] = curr_P.reshape(-1, 4)
        for camera_idx in range(n_cameras):
            M[3*camera_idx:3*camera_idx+3, 4+camera_idx] = -curr_x[camera_idx]
        # solve the equation
        U, S, V_h = np.linalg.svd(M)
        # extract the 3D points from the equation
        curr_X = utils.pflat(V_h.T[0:4, -1])
        X[:, point_idx] = curr_X
    return X


def max_single_point_errors(P, x, Xi):
    """
    Get the maximum reprojection error of the 3D predicted point U with the cameras P and real image points u.
    :param P: the cameras, with shape [n_cam, 3, 4]
    :param x: image points of size [n_cam, 3]
    :param Xi: 3D point of size [4]
    :return: The maximum reprojection error over all the cameras. (a scalar)
    The reprojection error on camera i is:
    ei = sqrt(sum((u_i - P_i*U)**2))
    """
    proj = P @ Xi
    proj = proj / proj[:,[-1]]
    errors = np.linalg.norm(proj - x, axis=1)
    return np.max(errors)


def get_triangulation_parameters(P, x):
    """
    Given an array of cameras and projected points, find the parameters needed for SOCP.
    :param P: cameras array of size [n_cam, 3, 4].
    :param x: projected points of size [n_cam, 3]
    :return: A,c - the variables for SOCP.
    Each variable should be a list in the length of the number of constraints (number of cameras that see the point).
    The i'th constraint will correspond to: ||A[i] * x|| <= gamma* c[i]^T * x
    """
    P_3s = P[:, [2], :]
    A = (np.expand_dims(x, axis=2) * P_3s) - P
    A = A[:, 0:2, :] # ignore last row (homogeneous)
    A = [np.squeeze(e, 0) for e in np.split(A, A.shape[0], axis=0)] # split into list of arrays
    P_3s = np.squeeze(P_3s, 1)
    c = [np.squeeze(e, 0) for e in np.split(P_3s, P_3s.shape[0], axis=0)] # split into list of arrays
    return A, c

def solve_SOCP(A,c,gamma):
    """
    Use cvxpy to solve a second order cone program of the form:
    minimize f^TX s.t ||A[i]x + b[i]|| <= c[i]^Tx+d[i] for i=1,...
    :param A: a list of numpy arrays
    :param c: a list of numpy arrays
    :return: If there is a solution return it. else, return None.
     A solution is an array of size [4,1]. Make sure it's least coordinate is 1.

    Hint: use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    """
    assert len(A) == len(c)
    num_cams = len(A)
    s = 0 # s = cp.Parameter() didn't work :'(
    X = cp.Variable((4,))
    constraints = [cp.SOC(gamma * (c[i].T @ X) + s, A[i] @ X) for i in range(num_cams)]
    problem = cp.Problem(cp.Minimize(s), constraints)
    try:
        problem.solve()
        if X.value is not None:
            return utils.pflat(X.value)
        else:
            return None
    except:
        return None


def SOCP_triangulate(x, P, max_iter, low=0, high=1, tol=1e-2):
    """
    Get a single 3D point from a list of cameras that see it and corresponding image points.
    Use an iterative algorithm similar to bisection.
    :param x: ndarray of shape [n_i, 3] the matching image points for the 3D points we are looking for
    :param P: ndarray of shape [n_i, 3, 4] cameras that see the 3D point
    :param max_iter: maximal number of iterations before stopping.
    :param low: minimal current value for gamma.
    :param high: maximal current value for gamma.
    :param tol: the boundaries around gamma which are close enough for a good solution.
    :return: Xi ndarray of shape [4,1] the point we got in the last feasible iteration
    """
    gamma = high
    curr_iter = 0
    best_X = None
    while (high - low > tol) and (curr_iter <= max_iter):
        curr_iter += 1
        A, c = get_triangulation_parameters(P, x)
        X = solve_SOCP(A, c, gamma)
        err = max_single_point_errors(P, x, X) if X is not None else None
        if (X is not None) and (err <= gamma):
            high = err
            best_X = X
        else:
            low = gamma
            if best_X is None:
                high = 2*high
        gamma = (high + low) / 2
    return best_X


def SOCP_triangulate_all_points(x, P, visible_points, max_iter=50, low=0, high=1024, tol=1e-2):
    n = x.shape[-1]
    X = np.zeros((4, n))
    for i in tqdm.tqdm(range(n)):
        X[:, i] = SOCP_triangulate(x[visible_points[:,i],:,i], P[visible_points[:,i]], max_iter=max_iter, low=low, high=high, tol=tol)
    return X
