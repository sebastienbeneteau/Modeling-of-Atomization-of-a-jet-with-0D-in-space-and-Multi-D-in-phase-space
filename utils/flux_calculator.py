import numpy as np
import itertools

def first_order_local_flux(nodes, weights, k, function = lambda xi1, xi2, alpha: 0):
    """
    Calculate the multivariate general first order point processes local flux for a given set of nodes and weights.
    Useful for: Collision, aggregation, and coalescence processes.
    
    Parameters:
    nodes (numpy.ndarray): Array of nodes in phase space. (xi1, xi2, ...)
    weights (numpy.ndarray): Array of weights corresponding to the nodes. (w1, w2, ...)
    k (numpy.ndarray): order of the moment. (k11, k12, ...)
    
    function (callable): Function to compute the local flux.
    
            Defaults to a function that returns 0.
            
            An example of a function could be:
            def function(xi1, xi2, k1, k2):
                return k1 * xi1**(k1-1) * xi2**k2 + k2 * xi1**k1 * xi2**(k2-1)
                # This is just an example, the actual function will depend on the specific problem.
                
    Returns:
    numpy.ndarray: The calculated local flux for the order k1 and k2.
    """
    # Ensure the input arrays are numpy arrays
    nodes = np.asarray(nodes)
    weights = np.asarray(weights)

    local_flux = 0
    
    for i in range(len(weights)):
        for j in range(len(weights)):
            local_flux += weights[i] * weights[j] * function(nodes[:,i], nodes[:,j], k)

    return local_flux



def zeroth_order_local_flux(t, nodes, weights, k, function = lambda t,xi,alpha: 0):
    """
    Calculate the multivariate general zeroth order point process local flux for a given set of nodes and weights.
    Useful for: nucleation, growth, and diffusion processes.
    
    
    Parameters:
    nodes (numpy.ndarray): Array of nodes in phase space. (xi1, xi2, ...)
    weights (numpy.ndarray): Array of weights corresponding to the nodes. (w1, w2, ...)
    k (numpy.ndarray): order of the moment. (k1, k2, ...)
    function (callable): Function to compute the local flux.
    
            Defaults to a function that returns 0.
            
            An example of a function could be:
            def function(xi, k):
                return k[0] * xi[0]**(k[0]-1) * xi[1]**k[1] * ...
            
    Returns:
    numpy.ndarray: The calculated local flux for the order k.
    """
    # Ensure the input arrays are numpy arrays
    nodes = np.asarray(nodes)
    weights = np.asarray(weights)

    local_flux = 0
    if weights.ndim == 0:
        weights = np.array([weights])
        nodes = np.array([nodes])
        
    for i in range(len(weights)):
        if len(nodes.shape) == 1:
            local_flux += weights[i] * function(t,nodes[i],k)
        else:
            local_flux += weights[i] * function(t,nodes[:,i],k)
        

    return local_flux




###################################### Old Code #######################################


# def phase_space_drift_local_flux(nodes, weights, growth_rate, k):
#     """
#     Calculate the phase space drift flux for a given set of nodes and weights.

#     Parameters:
#     nodes (numpy.ndarray): Array of nodes in phase space.
#     weights (numpy.ndarray): Array of weights corresponding to the nodes.
#     growth_rate (function): The growth rate of the system.
#     k (int): order of the moment.
    
#     Returns:
#     numpy.ndarray: The calculated drift flux for each node.
#     """
#     # Ensure the input arrays are numpy arrays
#     nodes = np.asarray(nodes)
#     weights = np.asarray(weights)

#     drift_flux = 0

#     if k == 0:
#         # Nucleation rate
#         try:
#             drift_flux = np.sum(weights * nodes)*growth_rate(0)
#         except ZeroDivisionError:
#             return 0
        
#     else:
#         for i in range(len(weights)):
#             drift_flux += weights[i] * nodes[i]**(k-1) * growth_rate(nodes[i]) 
#         drift_flux*= k

#     return drift_flux


# def breakage_local_flux(nodes, weights, a0, x, k):
#     """
#     Calculate the fragmentation flux for a given set of nodes and weights.

#     Parameters:
#     nodes (numpy.ndarray): Array of nodes in phase space.
#     weights (numpy.ndarray): Array of weights corresponding to the nodes.
#     a0 (float): constant breakage kernel.
#     x (float): fraction of fragmentation.
#     k (int): order of the moment.

#     Returns:
#     numpy.ndarray: The calculated fragmentation flux for each node.
#     """
#     # Ensure the input arrays are numpy arrays
#     nodes = np.asarray(nodes)
#     weights = np.asarray(weights)

#     m_k = np.dot(weights, nodes**k) 
#     breakage_flux = a0 * (x**k + (1-x)**k) * m_k - a0 * m_k

#     return breakage_flux

# def drag_force_local_flux(nodes, weights, params, k):
#     """
#     Calculate the drag force flux for a given set of nodes and weights.

#     Parameters:
#     nodes (numpy.ndarray): Array of nodes in phase space.
#     weights (numpy.ndarray): Array of weights corresponding to the nodes.
#     params (dict): Dictionary containing parameters for the drag force calculation.

#     Returns:
#     numpy.ndarray: The calculated drag force flux for each node.
#     """
#     # Ensure the input arrays are numpy arrays
#     nodes = np.asarray(nodes)
#     weights = np.asarray(weights)

#     # Extract parameters from the dictionary
#     Cd = params['Drag coefficient']
#     rhol = params['Fluid density']
#     dp = params['Particle diameter']
    
#     # Calculate the drag force flux
#     drag_force_flux = 0
#     for i in range(len(weights)):
#         drag_force_flux += weights[i] * nodes[i]**(k+1) * np.abs(nodes[i])
#     drag_force_flux *= -.125*Cd*rhol*np.pi*dp**2
        
#     return drag_force_flux

# # For bivariate fluxes
# def drift_1_local_flux(nodes, weights, growth_rate, k1, k2):
#     """
#     Calculate the phase space drift for first parameter flux for a given set of nodes and weights.

#     Parameters:
#     nodes (numpy.ndarray): Array of nodes in phase space.
#     weights (numpy.ndarray): Array of weights corresponding to the nodes.
#     growth_rate (function): The growth rate of the system.
#     k1 (int): order of the first moment.
#     k2 (int): order of the second moment.

#     Returns:
#     numpy.ndarray: The calculated drift flux for each node.
#     """
#     # Ensure the input arrays are numpy arrays
#     nodes = np.asarray(nodes)
#     weights = np.asarray(weights)

#     drift_flux = 0

#     if k1 == 0 and k2 == 0:
#         # Nucleation rate
#         try:
#             drift_flux = np.sum(weights * nodes[0,:] * nodes[1,:], axis = 0)*growth_rate(0,0)
#         except ZeroDivisionError:
#             return 0
        
#     else:
#         for i in range(len(weights)):
#             drift_flux += weights[i] * nodes[0,i]**(k1-1) * nodes[1,i]**k2 * growth_rate(nodes[0,i], nodes[1,i]) 
#         drift_flux*= k1

#     return drift_flux

# def drift_2_local_flux(nodes, weights, growth_rate, k1, k2):
#     """
#     Calculate the phase space drift for first parameter flux for a given set of nodes and weights.

#     Parameters:
#     nodes (numpy.ndarray): Array of nodes in phase space.
#     weights (numpy.ndarray): Array of weights corresponding to the nodes.
#     growth_rate (function): The growth rate of the system.
#     k1 (int): order of the first moment.
#     k2 (int): order of the second moment.

#     Returns:
#     numpy.ndarray: The calculated drift flux for each node.
#     """
#     # Ensure the input arrays are numpy arrays
#     nodes = np.asarray(nodes)
#     weights = np.asarray(weights)

#     drift_flux = 0

#     if k1 == 0 and k2 == 0:
#         # Nucleation rate
#         try:
#             drift_flux = np.sum(weights * nodes[0,:] * nodes[1,:]) * growth_rate(0,0)
#         except ZeroDivisionError:
#             return 0
        
#     else:
#         for i in range(len(weights)):
#             drift_flux += weights[i] * nodes[0,i]**k1 * nodes[1,i]**(k2-1) * growth_rate(nodes[0,i], nodes[1,i]) 
#         drift_flux*= k2

#     return drift_flux

# def bivariate_general_local_flux(nodes, weights, k1, k2, function = lambda x1,x2,a1,a2: 1):
#     """
  
#     """
#     # Ensure the input arrays are numpy arrays
#     nodes = np.asarray(nodes)
#     weights = np.asarray(weights)

#     local_flux = 0
    
#     for i in range(len(weights)):
#         local_flux += weights[i] * function(nodes[0,i], nodes[1,i], k1, k2)

#     return local_flux
