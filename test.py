# Graphs for testing
# Example graphs for testing
# Graph 1: A square with vertices 1, 2, 3, and 4
V1 = ['1', '2', '3', '4']
E1 = [('1', '2'), ('1', '3'), ('3', '4'), ('4', '1')]
G1 = (V1, E1)

# Graph 2: A square with vertices 1, 2, 3, and 4
V2 = ['1', '2', '3', '4']
E2 = [('1', '2'), ('1', '3'), ('3', '4'), ('4', '1')]
G2 = (V2, E2)

def matrix_isomorphisms(G1, G2):
    """
    Method 3: Checks isomorphism using permutation matrices.
    Tries all permutations of the identity matrix (P),
    and tests if A1 = P * A2 * P.T

    Parameters:
        G1: (nodes1, edges1)
        G2: (nodes2, edges2)

    Returns:
        List of valid permutations (as tuples) that represent isomorphisms
    """
    import numpy as np
    from itertools import permutations
    
    nodes1, edges1 = G1
    nodes2, edges2 = G2

    if len(nodes1) != len(nodes2):
        return []

    n = len(nodes1)

    # Create adjacency matrices for both graphs
    A1 = np.zeros((n, n), dtype=int)
    A2 = np.zeros((n, n), dtype=int)

    index_map1 = {node: i for i, node in enumerate(nodes1)}
    index_map2 = {node: i for i, node in enumerate(nodes2)}

    for u, v in edges1:
        i, j = index_map1[u], index_map1[v]
        A1[i][j] = 1
        A1[j][i] = 1  # Undirected

    for u, v in edges2:
        i, j = index_map2[u], index_map2[v]
        A2[i][j] = 1
        A2[j][i] = 1

    valid_mappings = []

    for perm in permutations(range(n)):
        # Create permutation matrix P
        P = np.zeros((n, n), dtype=int)
        for i in range(n):
            P[i][perm[i]] = 1

        # Apply the formula: A1 = P * A2 * P^T
        transformed = P @ A2 @ P.T
        
        # Only add the permutation if it's a valid isomorphism
        if np.array_equal(A1, transformed):
            valid_mappings.append(perm)

    return valid_mappings

# Example usage
print(matrix_isomorphisms(G1, G2))