import numpy as np

def wheeler(moments, n , adaptive=False, rmin = 1e-10, eabs = 1e-10, cutoff = 1e-30):
    """
    Inverts moments into 1D quadrature weights and abscissas using the adaptive Wheeler algorithm.

    The function calculates quadrature nodes and weights by inverting the provided statistical moments of a 
    probability density function (PDF) using an adaptive Wheeler approach. This method is used to find the 
    nodes and weights that are consistent with the moments of the distribution.

    :param moments: Statistical moments of the transported PDF.
    :type moments: array-like
        An array of moments, where each element corresponds to a moment (e.g., mean, variance, etc.) of the distribution.
        
    :param n: Maximum number of nodes (abscissas) for the quadrature.
    :type n: int
        The desired number of nodes for the quadrature approximation.

    :param adaptive: Flag to indicate whether to use the adaptive Wheeler algorithm.
    :type adaptive: bool, optional
        If `True`, the algorithm will adaptively refine the quadrature. Defaults to `False`.

    :param rmin: Minimum ratio of the weights for the nodes.
    :type rmin: float, optional
        The minimum ratio of the smallest to the largest weight, useful for controlling precision. Defaults to 1e-10.
        
    :param eabs: Minimum absolute distance between distinct abscissas (nodes).
    :type eabs: float, optional
        The minimum distance between nodes to ensure they are distinct. Defaults to 1e-10.

    :param cutoff: Minimum value for the weights.
    :type cutoff: float, optional
        The minimum weight value to consider a node valid. Defaults to 1e-10.
        
    :return: A tuple containing the abscissas (nodes), weights, and a flag indicating if an error occurred.
    :rtype: tuple of (array-like, array-like, bool)
        The first array contains the quadrature nodes (abscissas), the second array contains the corresponding weights,
        and the third element is a boolean flag `werror`. If `werror` is `True`, it indicates that an error occurred
        during the calculation.

    The algorithm works by using the given moments to generate quadrature points and their associated weights that
    satisfy the given statistical moments. The `adaptive` flag, if set to `True`, will refine the quadrature 
    nodes to improve accuracy by adjusting the points iteratively based on the error from previous iterations.
    
    The parameters `rmin` and `eabs` provide additional control over the accuracy and precision of the quadrature.
    If an error occurs during the process (e.g., the moments don't lead to a realizable quadrature), the function 
    will set `werror` to `True`.
    """

    werror = 0
    
    # Check if moments are unrealizable.
    if moments[0] <= 0:
        print("Wheeler: Moments are NOT realizable (moment[0] <= 0.0). Run failed.")
        werror = 1
        return np.array([moments[1]/moments[0]]), np.array([moments[0]]), werror

    if n == 1 or (adaptive and moments[0] <= rmin):
        w = moments[0]
        x = moments[1] / moments[0]
        return x, w, werror

    # Set modified moments equal to input moments.
    nu = moments.copy()
    ind = n
    # Construct recurrence matrix
    a = np.zeros(ind)
    b = np.zeros(ind)
    sigma = np.zeros((2 * ind + 1, 2 * ind + 1))

    for i in range(1, 2 * ind + 1):
        sigma[1, i] = nu[i - 1]

    a[0] = nu[1] / nu[0]
    b[0] = 0

    for k in range(2, n + 1):
        for l in range(k, 2 * ind - k + 2):
            sigma[k, l] = (
                sigma[k - 1, l + 1]
                - a[k - 2] * sigma[k - 1, l]
                - b[k - 2] * sigma[k - 2, l]
            )
        a[k - 1] = sigma[k, k + 1] / sigma[k, k] - sigma[k - 1, k] / sigma[k - 1, k - 1]
        b[k - 1] = sigma[k, k] / sigma[k - 1, k - 1]

    # Find maximum n using diagonal element of sigma
    if adaptive:
        for k in range(ind, 1, -1):
            if sigma[k, k] < cutoff:
                n = k - 1
                # n = k - 2
                if n == 1:
                    w = moments[0]
                    x = moments[1] / moments[0]
                    return np.array([x]), np.array([w]), werror
                
        # Use maximum n to re-calculate recurrence matrix
        a = np.zeros(n)
        b = np.zeros(n)
        w = np.zeros(n)
        x = np.zeros(n)
        sigma = np.zeros((2 * n + 1, 2 * n + 1))
        
        for i in range(1, 2 * n + 1):
            sigma[1, i] = nu[i - 1]

        a[0] = nu[1] / nu[0]
        b[0] = 0
        for k in range(2, n + 1):
            for l in range(k, 2 * n - k + 2):
                sigma[k, l] = (
                    sigma[k - 1, l + 1]
                    - a[k - 2] * sigma[k - 1, l]
                    - b[k - 2] * sigma[k - 2, l]
                )
            a[k - 1] = (
                sigma[k, k + 1] / sigma[k, k] - sigma[k - 1, k] / sigma[k - 1, k - 1]
            )
            b[k - 1] = sigma[k, k] / sigma[k - 1, k - 1]

    # Check if moments are unrealizable (should not happen)
    if b.min() < 0:
        print("Moments in Wheeler_moments are not realizable! Program exits.")
        werror = 1
        return np.array([moments[1]/moments[0]]), np.array([moments[0]]), werror

    # Setup Jacobi matrix for n-point quadrature, adapt n using rmin and eabs
    for n1 in range(n, 0, -1):
        if n1 == 1:
            w = moments[0]
            x = moments[1] / moments[0]
            return np.array([x]), np.array([w]), werror


        # Jacobi matrix
        sqrt_b = np.sqrt(b[1:n1])
        jacobi = np.diag(a[:n1]) + np.diag(sqrt_b, -1) + np.diag(sqrt_b, 1)

        # Compute weights and abscissas
        eigenvalues, eigenvectors = np.linalg.eig(jacobi)
        idx = eigenvalues.argsort()
        x = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
        w = moments[0] * eigenvectors[0, :] ** 2
        # Adaptive conditions. When both satisfied, return the results.
        if adaptive:
            dab = np.zeros(n1)
            mab = np.zeros(n1)

            for i in range(n1 - 1, 0, -1):
                dab[i] = min(abs(x[i] - x[0:i]))
                mab[i] = max(abs(x[i] - x[0:i]))

            mindab = min(dab[1:n1])
            maxmab = max(mab[1:n1])
            if n1 == 2:
                maxmab = 1
            if min(w) / max(w) > rmin and mindab / maxmab > eabs:
                return np.array(x), np.array(w), werror
        else:
            return np.array(x), np.array(w), werror

def CQMOM_Bivariate(N1, N2, m, adaptive = False, rmin = [1e-3]*2, eabs = [1e-8]*2, cutoff = 1e-10):
    """
    Compute a bivariate quadrature approximation using CQMOM.
    """
    flag = False

    xi1, w1, werror = wheeler(m[:, 0], N1, adaptive, rmin[0], eabs[0])

    if werror > 0:
        print("1D quadrature failed on first step!")
        flag = True
        
    N1 = len(w1)
    for i in range(N1):
        if abs(w1[i]) / m[0, 0] < cutoff:
            print("One of the weights is null! Reduce the number of nodes in direction 1")
            flag = True
            break

    if not flag:
        # Define the Vandermonde and the R1 Matrices for conditional moments
        V1 = np.vander(xi1, N1, increasing=True).T 
        R1 = np.diag(w1)
        
        # Calculate the conditional moments
        c_m = np.zeros((N1, 2 * N2))
        inv_V1_R1 = np.linalg.inv(V1 @ R1)
        for j in range(2 * N2):
            c_m[:, j] = inv_V1_R1 @ m[:N1, j]

        xi2 = np.zeros((N2, N1))
        w2 = np.zeros((N2, N1))
        for i in range(N1):
            
            temp_xi2, temp_w2, werror = wheeler(c_m[i, :], N2, adaptive, rmin[1], eabs[1])
            xi2[:len(temp_xi2), i], w2[:len(temp_w2), i] = temp_xi2, temp_w2
        
            if werror > 0:
                print("1D quadrature failed on second step!")
                flag = True

        w = np.zeros(N1 * N2)
        xi = np.zeros((2, N1 * N2))
        index = 0
        for i in range(N1):
            for j in range(N2):
                w[index] = w1[i] * w2[j, i]
                xi[0, index] = xi1[i]
                xi[1, index] = xi2[j, i]
                index += 1
    else:
        w = np.zeros(N1 * N2)
        xi = np.zeros((2, N1 * N2))

    return w, xi


def CQMOM_Trivariate(N1, N2, N3, m, adaptive=False, rmin = [1e-2]*3, eabs = [1e-8]*3, cutoff = 1e-2):
    """
    Compute a trivariate quadrature approximation using CQMOM.
    
    Parameters:
    N1, N2, N3: int
        Number of nodes in each dimension.
    m: ndarray
        Moment array with shape (N1, 2*N2, 2*N3).
    adaptive: bool
        Whether to use adaptive node calculation.

    Returns:
    w: ndarray
        Weights of the quadrature nodes.
    xi: ndarray
        Coordinates of the quadrature nodes.
    """
    flag = False

    # Step 1: Compute nodes and weights for the first variable
    xi1, w1, werror = wheeler(m[:, 0, 0], N1, adaptive, rmin[0], eabs[0])

    if werror > 0:
        print("1D quadrature failed on first step!")
        flag = True
        
    N1 = len(w1)
    for i in range(N1):
        if abs(w1[i]) / m[0, 0, 0] < cutoff:
            print("One of the weights is null! Reduce the number of nodes in direction 1")
            flag = True
            break

    if not flag:
        # Define the Vandermonde and R1 matrices for conditional moments in direction 1
        V1 = np.vander(xi1, N1, increasing=True).T
        R1 = np.diag(w1)

        # Step 2: Calculate the conditional moments for the second variable
        xi2 = np.zeros((N2,N1))
        w2 = np.zeros((N2, N1))
        c_m2 = np.zeros((N1, 2 * N2, 2 * N3))
        inv_V1_R1 = np.linalg.inv(V1 @ R1)
        for j in range(2 * N2):
            for k in range(2 * N3):
                c_m2[:, j, k] = inv_V1_R1 @ m[:N1, j, k]

        for i in range(N1):
            temp_xi2, temp_w2, werror = wheeler(c_m2[i, :, 0], N2, adaptive, rmin[1], eabs[1])
            xi2[:len(temp_xi2), i], w2[:len(temp_w2), i] = temp_xi2, temp_w2
            
            if werror > 0:
                print("1D quadrature failed on second step!")
                flag = True
                
        
        # Step 3: Compute nodes and weights for the third variable
        xi3 = np.zeros((N3, N2, N1))
        w3 = np.zeros((N3, N2, N1))
        for i in range(N1):
            c_m3 = np.zeros((N2, 2 * N3))
            for j in range(N2):
                for k in range(2 * N3):
                    V2 = np.vander(xi2[:, i], N2, increasing=True).T
                    R2 = np.diag(w2[:, i])
                    inv_V2_R2 = np.linalg.inv(V2 @ R2)
                    
                    c_m3[:,k] = inv_V2_R2 @ c_m2[i, :N2, k]  

                temp_xi3, temp_w3, werror = wheeler(c_m3[j,:], N3, adaptive, rmin[2], eabs[2])
                xi3[:len(temp_xi3), j, i], w3[:len(temp_w3), j, i] = temp_xi3, temp_w3
            
                if werror > 0:
                    print("1D quadrature failed on third step!")
                    flag = True
                    
        # Combine weights and nodes into final arrays
        w = np.zeros(N1 * N2 * N3)
        xi = np.zeros((3, N1 * N2 * N3))
        index = 0
        for i in range(N1):
            for j in range(N2):
                for k in range(N3):
                    w[index] = w1[i] * w2[j, i] * w3[k, j, i]
                    xi[0, index] = xi1[i]
                    xi[1, index] = xi2[j, i]
                    xi[2, index] = xi3[k, j, i]
                    index += 1
    else:
        w = np.zeros(N1 * N2 * N3)
        xi = np.zeros((3, N1 * N2 * N3))

    return w, xi


def CQMOM_Quadrivariate(N1, N2, N3, N4, m, adaptive=False, rmin = [1e-2]*4, eabs = [1e-8]*4, cutoff = 1e-2):
    """
    Compute a quadrivariate quadrature approximation using CQMOM.
    
    Parameters:
    N1, N2, N3, N4: int
        Number of nodes in each dimension.
    m: ndarray
        Moment array with shape (N1, 2*N2, 2*N3, 2*N4).
    adaptive: bool
        Whether to use adaptive node calculation.

    Returns:
    w: ndarray
        Weights of the quadrature nodes.
    xi: ndarray
        Coordinates of the quadrature nodes.
    """
    flag = False

    # Step 1: Compute nodes and weights for the first variable
    xi1, w1, werror = wheeler(m[:, 0, 0, 0], N1, adaptive, rmin[0], eabs[0])
    if werror > 0:
        print("1D quadrature failed on first step!")
        flag = True
            
    N1 = len(w1)
    for i in range(N1):
        if abs(w1[i]) / m[0, 0, 0, 0] < cutoff:
            print("One of the weights is null! Reduce the number of nodes in direction 1")
            flag = True
            break

    if not flag:
        # Define the Vandermonde and R1 matrices for conditional moments in direction 1
        V1 = np.vander(xi1, N1, increasing=True).T
        R1 = np.diag(w1)

        # Calculate the conditional moments for the second variable
        c_m2 = np.zeros((N1, 2 * N2, 2 * N3, 2 * N4))
        inv_V1_R1 = np.linalg.inv(V1 @ R1)
        for j in range(2 * N2):
            for k in range(2 * N3):
                for l in range(2 * N4):
                    c_m2[:, j, k, l] = inv_V1_R1 @ m[:N1, j, k, l]

        xi2 = np.zeros((N2, N1))
        w2 = np.zeros((N2, N1))
        for i in range(N1):
            temp_xi2, temp_w2, werror = wheeler(c_m2[i, :, 0, 0], N2, adaptive, rmin[1], eabs[1])
            xi2[:len(temp_xi2), i], w2[:len(temp_w2), i] = temp_xi2, temp_w2
            
            if werror > 0:
                print("1D quadrature failed on second step!")
                flag = True
            

        # Step 2: Compute nodes and weights for the third variable
        xi3 = np.zeros((N3, N2, N1))
        w3 = np.zeros((N3, N2, N1))
        for i in range(N1):
            c_m3 = np.zeros((N2, 2 * N3, 2 * N4))
            for j in range(N2):
                for k in range(2 * N3):
                    for l in range(2 * N4):
                        V2 = np.vander(xi2[:, i], N2, increasing=True).T
                        R2 = np.diag(w2[:, i])
                        inv_V2_R2 = np.linalg.inv(V2 @ R2)
                        
                        c_m3[:, k, l] = inv_V2_R2 @ c_m2[i, :N2, k, l]
                
                temp_xi3, temp_w3, werror = wheeler(c_m3[j, :, 0], N3, adaptive, rmin[2], eabs[2])
                xi3[:len(temp_xi3), j, i], w3[:len(temp_w3), j, i] = temp_xi3, temp_w3
                
                if werror > 0:
                    print("1D quadrature failed on third step!")
                    flag = True
                
                
        # Step 3: Compute nodes and weights for the fourth variable
        xi4 = np.zeros((N4, N3, N2, N1))
        w4 = np.zeros((N4, N3, N2, N1))
        for i in range(N1):
            for j in range(N2):
                c_m4 = np.zeros((N3, 2 * N4))
                for k in range(N3):
                    for l in range(2 * N4):
                        V3 = np.vander(xi3[:, j, i], N3, increasing=True).T
                        R3 = np.diag(w3[:, j, i])
                        inv_V3_R3 = np.linalg.inv(V3 @ R3)
                        
                        c_m4[:, l] = inv_V3_R3 @ c_m3[j, :N3, l]
                    
                    temp_xi4, temp_w4, werror = wheeler(c_m4[k, :], N4, adaptive, rmin[3], eabs[3])
                    xi4[:len(temp_xi4), k, j, i], w4[:len(temp_w4), k, j, i] = temp_xi4, temp_w4
                    if werror > 0:
                        print("1D quadrature failed on fourth step!")
                        flag = True
                    
                    
        # Combine weights and nodes into final arrays
        w = np.zeros(N1 * N2 * N3 * N4)
        xi = np.zeros((4, N1 * N2 * N3 * N4))
        index = 0
        for i in range(N1):
            for j in range(N2):
                for k in range(N3):
                    for l in range(N4):
                        w[index] = w1[i] * w2[j, i] * w3[k, j, i] * w4[l, k, j, i]
                        xi[0, index] = xi1[i]
                        xi[1, index] = xi2[j, i]
                        xi[2, index] = xi3[k, j, i]
                        xi[3, index] = xi4[l, k, j, i]
                        index += 1
    else:
        w = np.zeros(N1 * N2 * N3 * N4)
        xi = np.zeros((4, N1 * N2 * N3 * N4))
        
    return w, xi


def CQMOM_Quadrivariate(N1, N2, N3, N4, m, adaptive=False, rmin=None, eabs=None, cutoff=1e-2):
    """
    Compute a quadrivariate quadrature approximation using CQMOM.

    Parameters:
    N1, N2, N3, N4: int
        Number of nodes in each dimension.
    m: ndarray
        Moment array with shape (N1, 2*N2, 2*N3, 2*N4).
    adaptive: bool
        Whether to use adaptive node calculation.
    rmin: list of float
        Minimum relative error for each dimension.
    eabs: list of float
        Absolute error tolerance for each dimension.
    cutoff: float
        Threshold to consider a weight as negligible.

    Returns:
    w: ndarray
        Weights of the quadrature nodes.
    xi: ndarray
        Coordinates of the quadrature nodes.
    """
    if rmin is None:
        rmin = [1e-2] * 4
    if eabs is None:
        eabs = [1e-8] * 4

    def compute_conditional_moments(V, R, moments):
        """
        Compute conditional moments by solving linear systems.

        Parameters:
        V: ndarray
            Vandermonde matrix.
        R: ndarray
            Diagonal weight matrix.
        moments: ndarray
            Moments to condition.

        Returns:
        c_moments: ndarray
            Conditional moments.
        """
        VR = V @ R
        c_moments = np.linalg.solve(VR, moments)
        return c_moments

    flag = False

    # Step 1: Compute nodes and weights for the first variable
    xi1, w1, werror = wheeler(m[:, 0, 0, 0], N1, adaptive, rmin[0], eabs[0])
    if werror > 0:
        print("1D quadrature failed on first step!")
        flag = True

    N1 = len(w1)
    for i in range(N1):
        if abs(w1[i]) / m[0, 0, 0, 0] < cutoff:
            print("One of the weights is negligible! Reduce the number of nodes in direction 1.")
            flag = True
            break

    # if flag:
    #     w = np.zeros(N1 * N2 * N3 * N4)
    #     xi = np.zeros((4, N1 * N2 * N3 * N4))
    #     return w, xi

    # Precompute Vandermonde and weight matrices
    V1 = np.vander(xi1, N1, increasing=True).T
    R1 = np.diag(w1)

    # Step 2: Compute conditional moments for the second variable
    c_m2 = np.zeros((N1, 2 * N2, 2 * N3, 2 * N4))
    for j in range(2 * N2):
        for k in range(2 * N3):
            for l in range(2 * N4):
                c_m2[:, j, k, l] = compute_conditional_moments(V1, R1, m[:N1, j, k, l])

    xi2 = np.zeros((N2, N1))
    w2 = np.zeros((N2, N1))
    for i in range(N1):
        temp_xi2, temp_w2, werror = wheeler(c_m2[i, :, 0, 0], N2, adaptive, rmin[1], eabs[1])
        xi2[:len(temp_xi2), i], w2[:len(temp_w2), i] = temp_xi2, temp_w2
        if werror > 0:
            print(f"1D quadrature failed on second step at index {i}!")
            flag = True

    # if flag:
    #     w = np.zeros(N1 * N2 * N3 * N4)
    #     xi = np.zeros((4, N1 * N2 * N3 * N4))
    #     return w, xi

    # Step 3: Compute nodes and weights for the third variable
    xi3 = np.zeros((N3, N2, N1))
    w3 = np.zeros((N3, N2, N1))
    for i in range(N1):
        V2 = np.vander(xi2[:, i], N2, increasing=True).T
        R2 = np.diag(w2[:, i])
        c_m3 = np.zeros((N2, 2 * N3, 2 * N4))
        for j in range(2 * N3):
            for k in range(2 * N4):
                c_m3[:, j, k] = compute_conditional_moments(V2, R2, c_m2[i, :N2, j, k])
        for j in range(N2):
            temp_xi3, temp_w3, werror = wheeler(c_m3[j, :, 0], N3, adaptive, rmin[2], eabs[2])
            xi3[:len(temp_xi3), j, i], w3[:len(temp_w3), j, i] = temp_xi3, temp_w3
            if werror > 0:
                print(f"1D quadrature failed on third step at indices ({i}, {j})!")
                flag = True

    # if flag:
    #     w = np.zeros(N1 * N2 * N3 * N4)
    #     xi = np.zeros((4, N1 * N2 * N3 * N4))
    #     return w, xi

    # Step 4: Compute nodes and weights for the fourth variable
    xi4 = np.zeros((N4, N3, N2, N1))
    w4 = np.zeros((N4, N3, N2, N1))
    for i in range(N1):
        for j in range(N2):
            V3 = np.vander(xi3[:, j, i], N3, increasing=True).T
            R3 = np.diag(w3[:, j, i])
            c_m4 = np.zeros((N3, 2 * N4))
            for k in range(2 * N4):
                c_m4[:, k] = compute_conditional_moments(V3, R3, c_m3[j, :N3, k])
            for k in range(N3):
                temp_xi4, temp_w4, werror = wheeler(c_m4[k, :], N4, adaptive, rmin[3], eabs[3])
                xi4[:len(temp_xi4), k, j, i], w4[:len(temp_w4), k, j, i] = temp_xi4, temp_w4
                if werror > 0:
                    print(f"1D quadrature failed on fourth step at indices ({i}, {j}, {k})!")
                    flag = True

    # if flag:
    #     w = np.zeros(N1 * N2 * N3 * N4)
    #     xi = np.zeros((4, N1 * N2 * N3 * N4))
    #     return w, xi

    # Combine weights and nodes into final arrays
    w = np.zeros(N1 * N2 * N3 * N4)
    xi = np.zeros((4, N1 * N2 * N3 * N4))
    index = 0
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                for l in range(N4):
                    w[index] = w1[i] * w2[j, i] * w3[k, j, i] * w4[l, k, j, i]
                    xi[0, index] = xi1[i]
                    xi[1, index] = xi2[j, i]
                    xi[2, index] = xi3[k, j, i]
                    xi[3, index] = xi4[l, k, j, i]
                    index += 1

    return w, xi


def CQMOM_Quintivariate(N1, N2, N3, N4, N5, m, adaptive=True, rmin=[1e-2]*5, eabs=[1e-8]*5, cutoff=1e-2):
    """
    Compute a quintivariate quadrature approximation using CQMOM.
    
    Parameters:
    N1, N2, N3, N4, N5: int
        Number of nodes in each dimension.
    m: ndarray
        Moment array with shape (N1, 2*N2, 2*N3, 2*N4, 2*N5).
    adaptive: bool
        Whether to use adaptive node calculation.
    rmin: list
        Minimum relative error for node calculation in each dimension.
    eabs: list
        Absolute error tolerance for node calculation in each dimension.
    cutoff: float
        Minimum weight threshold relative to the zeroth moment.

    Returns:
    w: ndarray
        Weights of the quadrature nodes.
    xi: ndarray
        Abscissas of the quadrature nodes.
    """
    flag = False

    # Step 1: Compute nodes and weights for the first variable
    xi1, w1, werror = wheeler(m[:, 0, 0, 0, 0], N1, adaptive, rmin[0], eabs[0])
    
    if werror > 0:
        print("1D quadrature failed on first step!")
        flag = True
        
    N1 = len(w1)
    if N1 == 1:
        if abs(w1) / m[0, 0, 0, 0, 0] < cutoff:
            print("One of the weights is null! Reduce the number of nodes in direction 1")
            flag = True
    else:
        for i in range(N1):
            if abs(w1[i]) / m[0, 0, 0, 0, 0] < cutoff:
                print("One of the weights is null! Reduce the number of nodes in direction 1")
                flag = True
                break

    if not flag:
        # Define the Vandermonde and R1 matrices for conditional moments in direction 1
        V1 = np.vander(xi1, N1, increasing=True).T
        R1 = np.diag(w1)

        # Calculate the conditional moments for the second variable
        c_m2 = np.zeros((N1, 2 * N2, 2 * N3, 2 * N4, 2 * N5))
        inv_V1_R1 = np.linalg.inv(V1 @ R1)
        for j in range(2 * N2):
            for k in range(2 * N3):
                for l in range(2 * N4):
                    for m_idx in range(2 * N5):
                        c_m2[:, j, k, l, m_idx] = inv_V1_R1 @ m[:N1, j, k, l, m_idx]

        xi2 = np.zeros((N2, N1))
        w2 = np.zeros((N2, N1))
        for i in range(N1):
            
            temp_xi2, temp_w2, werror = wheeler(c_m2[i, :, 0, 0, 0], N2, adaptive, rmin[1], eabs[1])
            xi2[:len(temp_xi2), i], w2[:len(temp_w2), i] = temp_xi2, temp_w2
            
            if werror > 0:
                print("1D quadrature failed on second step!")
                flag = True

        # Step 2: Compute nodes and weights for the third variable
        xi3 = np.zeros((N3, N2, N1))
        w3 = np.zeros((N3, N2, N1))
        for i in range(N1):
            c_m3 = np.zeros((N2, 2 * N3, 2 * N4, 2 * N5))
            for j in range(N2):
                for k in range(2 * N3):
                    for l in range(2 * N4):
                        for m_idx in range(2 * N5):
                            V2 = np.vander(xi2[:, i], N2, increasing=True).T
                            R2 = np.diag(w2[:, i])
                            inv_V2_R2 = np.linalg.inv(V2 @ R2)
                            
                            c_m3[:, k, l, m_idx] = inv_V2_R2 @ c_m2[i, :N2, k, l, m_idx]
                
                temp_xi3, temp_w3, werror = wheeler(c_m3[j, :, 0, 0], N3, adaptive, rmin[2], eabs[2])
                xi3[:len(temp_xi3), j, i], w3[:len(temp_w3), j, i] = temp_xi3, temp_w3
                
                if werror > 0:
                    print("1D quadrature failed on third step!")
                    flag = True

        # Step 3: Compute nodes and weights for the fourth variable
        xi4 = np.zeros((N4, N3, N2, N1))
        w4 = np.zeros((N4, N3, N2, N1))
        for i in range(N1):
            for j in range(N2):
                c_m4 = np.zeros((N3, 2 * N4, 2 * N5))
                for k in range(N3):
                    for l in range(2 * N4):
                        for m_idx in range(2 * N5):
                            V3 = np.vander(xi3[:, j, i], N3, increasing=True).T
                            R3 = np.diag(w3[:, j, i])
                            inv_V3_R3 = np.linalg.inv(V3 @ R3)
                            
                            c_m4[:, l, m_idx] = inv_V3_R3 @ c_m3[j, :N3, l, m_idx]
                        
                    xi4[:, k, j, i], w4[:, k, j, i], werror = wheeler(c_m4[k, :, 0], N4, adaptive, rmin[3], eabs[3])
                    
                    temp_xi4, temp_w4, werror = wheeler(c_m4[k, :, 0], N4, adaptive, rmin[3], eabs[3])
                    xi4[:len(temp_xi4), k, j, i], w4[:len(temp_w4), k, j, i] = temp_xi4, temp_w4
                    
                    if werror > 0:
                        print("1D quadrature failed on fourth step!")
                        flag = True

        # Step 4: Compute nodes and weights for the fifth variable
        xi5 = np.zeros((N5, N4, N3, N2, N1))
        w5 = np.zeros((N5, N4, N3, N2, N1))
        for i in range(N1):
            for j in range(N2):
                for k in range(N3):
                    c_m5 = np.zeros((N4, 2 * N5))
                    for l in range(N4):
                        for m_idx in range(2 * N5):
                            V4 = np.vander(xi4[:, k, j, i], N4, increasing=True).T
                            R4 = np.diag(w4[:, k, j, i])
                            inv_V4_R4 = np.linalg.inv(V4 @ R4)
                            
                            c_m5[:, m_idx] = inv_V4_R4 @ c_m4[k, :N4, m_idx]
                        
                        xi5[:, l, k, j, i], w5[:, l, k, j, i], werror = wheeler(c_m5[l, :], N5, adaptive, rmin[4], eabs[4])
                        
                        temp_xi5, temp_w5, werror = wheeler(c_m5[l, :], N5, adaptive, rmin[4], eabs[4])
                        xi5[:len(temp_xi5), l, k, j, i], w5[:len(temp_w5), l, k, j, i] = temp_xi5, temp_w5
                        
                        if werror > 0:
                            print("1D quadrature failed on fifth step!")
                            flag = True

        # Combine weights and nodes into final arrays
        w = np.zeros(N1 * N2 * N3 * N4 * N5)
        xi = np.zeros((5, N1 * N2 * N3 * N4 * N5))
        index = 0
        for i in range(N1):
            for j in range(N2):
                for k in range(N3):
                    for l in range(N4):
                        for m_idx in range(N5):
                            w[index] = w1[i] * w2[j, i] * w3[k, j, i] * w4[l, k, j, i] * w5[m_idx, l, k, j, i]
                            xi[0, index] = xi1[i]
                            xi[1, index] = xi2[j, i]
                            xi[2, index] = xi3[k, j, i]
                            xi[3, index] = xi4[l, k, j, i]
                            xi[4, index] = xi5[m_idx, l, k, j, i]
                            index += 1
    else:
        w = np.zeros(N1 * N2 * N3 * N4 * N5)
        xi = np.zeros((5, N1 * N2 * N3 * N4 * N5))
                
    return w, xi



import numpy as np
from functools import reduce

def CQMOM_Multivariate(N_list, m, adaptive=True, rmin=None, eabs=None, cutoff=1e-2):
    """
    Generalized CQMOM routine for arbitrary dimensions. (Not tested yet)

    Parameters:
    - N_list: list of ints, number of nodes in each dimension.
    - m: ndarray, multidimensional moment tensor with shape (2*N1, 2*N2, ..., 2*Nd)
    - adaptive: bool, whether to use adaptive wheeler routine
    - rmin, eabs: lists of floats, relative and absolute error tolerances per dimension
    - cutoff: float, weight cutoff threshold

    Returns:
    - w: array of final quadrature weights
    - xi: array of final quadrature nodes, shape (d, total_nodes)
    """
    dim = len(N_list)
    if rmin is None:
        rmin = [1e-2] * dim
    if eabs is None:
        eabs = [1e-8] * dim

    flag = False
    weights_list = []
    nodes_list = []

    def recursive_quadrature(level, m_current, idx_stack):
        nonlocal flag

        N = N_list[level]

        # Initial 1D quadrature
        try:
            xi, w, werror = wheeler(m_current, N, adaptive, rmin[level], eabs[level])
        except Exception as e:
            print(f"Quadrature failed at dimension {level+1}: {e}")
            xi = np.zeros(N)
            w = np.zeros(N)
            werror = 1

        if werror > 0:
            print(f"1D quadrature failed on step {level+1}")
            flag = True

        if N == 1:
            if abs(w) / m[tuple([0]*dim)] < cutoff:
                print(f"Null weight in dimension {level+1}")
                flag = True
        else:
            for wi in w:
                if abs(wi) / m[tuple([0]*dim)] < cutoff:
                    print(f"Null weight in dimension {level+1}")
                    flag = True
                    break

        weights_list.append(w)
        nodes_list.append(xi)

        if level == dim - 1:
            return

        # Prepare conditional moments for next level
        V = np.vander(xi, N, increasing=True).T
        R = np.diag(w)
        inv_VR = np.linalg.inv(V @ R)

        next_shape = [N] + list(m_current.shape[1:])
        c_m = np.zeros([N] + list(m_current.shape[1:]))

        it = np.ndindex(*m_current.shape[1:])
        for i in it:
            flat_i = [slice(None)] + list(i)
            mom_vec = m_current[tuple(flat_i)]
            c_m[(slice(None),) + i] = inv_VR @ mom_vec

        for i in range(N):
            recursive_quadrature(level + 1, c_m[i], idx_stack + [i])

    # Step 1: Start recursion
    recursive_quadrature(0, m, [])

    # Step 2: Combine final weights and nodes
    if flag:
        total_nodes = reduce(lambda x, y: x * y, N_list)
        return np.zeros(total_nodes), np.zeros((dim, total_nodes))

    mesh_grids = np.meshgrid(*nodes_list, indexing='ij')
    xi = np.array([g.flatten() for g in mesh_grids])
    w = reduce(lambda x, y: (x[..., None] * y), weights_list).flatten()

    return w, xi


def CQMOM(N, M, adaptive, rmin, eabs, cutoff = 1e-2):
    """
    This function inverts moments into N-D quadrature weights and abscissas using adaptive Wheeler algorithm.

    :param moments: Statistical moments of the transported PDF
    :type moments: array like
    :return: Abscissas, weights
    :rtype: array like
    :param n: Order of quadrature
    :type n: array like
    :param adaptive: Adaptive Wheeler algorithm
    :type adaptive: bool
    
    rmin = minimum ratio wmin/wmax
    rmin(0) = "vacuum" state
    rmin(i) = minimum ratio for i nodes: i=2, ..., n
    eabs = minimum distance between distinct abscissas
    cutoff = minimum value for the weights
    """
    if type(N) == int:
        nodes, weights, _ = wheeler(M, N, adaptive, rmin, eabs, cutoff)
        return weights, nodes
    if len(N)==2:
        return CQMOM_Bivariate(*N, M, adaptive, rmin, eabs, cutoff)
    if len(N)==3:
        return CQMOM_Trivariate(*N, M, adaptive, rmin, eabs, cutoff)
    if len(N)==4:
        return CQMOM_Quadrivariate(*N, M, adaptive, rmin, eabs, cutoff)
    if len(N)==5:
        return CQMOM_Quintivariate(*N, M, adaptive, rmin, eabs, cutoff)
    if len(N)>=6:
        return CQMOM_Multivariate(N, M, adaptive, rmin, eabs, cutoff)