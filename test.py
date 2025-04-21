# Kyrylo Kozlovskyi
# G00425385
# https://github.com/KyryloKozlovskyi/Graph-Definitely-Isomorphic-Analysis

import numpy as np
from itertools import permutations


# Function to get the degree of each vertex in a graph
def get_degree(V, E):
    """
    Get the degree of each vertex in a graph. The degree of a vertex is the number of edges connected to it.
    :param V: List of vertices
    :param E: List of edges
    :return: Dictionary with vertices as keys and their degrees as values
    """
    # Create a dictionary to store the degree of each vertex and initialize it with 0
    counts = {v: 0 for v in V}
    # Iterate over the vertices
    for v in V:
        # Iterate over the edges
        for e in E:
            # If the vertex is in the edge, increment the count
            if v in e:
                counts[v] += 1
    # Return the dictionary with degrees
    return counts


# Method 1
def decision_tree_candidates(G1, G2):
    """
    Generates candidate mappings between two graphs (likely isomorphic).

    Approach:
    1. First, check the degree of each vertex in both graphs
    2. For each vertex in G1, find all vertices in G2 with the same degree
    3. Build possible mappings using a tree approach:
       - Start with an empty mapping
       - For each vertex in G1, extend existing mappings with compatible vertices from G2
       - A vertex from G2 is compatible if it has the same degree and hasn't been used yet
    4. Results are all possible vertex mappings that preserve degree

    This method significantly reduces the number of total mappings compared to testing all permutations.
    It acts as a decision tree, avoiding combinations early if the degrees don't match.

    :param G1: First graph (V1, E1)
    :param G2: Second graph (V2, E2)
    :return: A sorted list of all possible mappings from G1 to G2.
    """

    # Unpack the graphs
    V1, E1 = G1  # Graph 1 (V1, E1)
    V2, E2 = G2  # Graph 2 (V2, E2)

    # Get the degrees of each vertex in both graphs
    deg1 = get_degree(V1, E1)  # Degree of vertices in graph 1
    deg2 = get_degree(V2, E2)  # Degree of vertices in graph 2

    # Check if the number of vertices is the same in both graphs
    mapping_dict = {}  # Dictionary to store possible mappings
    for v in V1:  # Iterate over vertices in graph 1
        d1 = deg1[v]  # Get the degree of the current vertex in graph 1
        possible = {w for w in V2 if deg2[w] == d1}  # Find vertices in graph 2 with the same degree
        mapping_dict[v] = possible  # Store the possible mappings for the current vertex

    # Check if the number of possible mappings is the same for each vertex
    index_map = {label: i for i, label in enumerate(V2)}

    # Check if the number of possible mappings is the same for each vertex
    mappings = [[]]  # Initialize mappings with an empty list
    for v in V1:  # Iterate over vertices in graph 1
        current = []  # Initialize current mappings
        for m in mappings:  # Iterate over existing mappings
            for option in mapping_dict[v]:  # Iterate over possible mappings for the current vertex
                idx = index_map[option]  # Get the index of the option in graph 2
                if idx not in m:  # Check if the option is already used in the current mapping
                    current.append(m + [idx])  # Add the new mapping to the current list
        mappings = current  # Update mappings with the current list
    # Return sorted mappings
    return sorted([tuple(m) for m in mappings])


# Method 2
def isomorphism_checker(G1, G2):
    """
    Checks which candidate mappings from Method 1 are actual isomorphisms.

    Approach:
    1. Get all candidate mappings from Method 1
    2. For each mapping:
       - Transform each edge in G1 according to the mapping
       - Check if the transformed edges match exactly with G2's edges
    3. The mapping is an isomorphism if the transformed edge set equals G2's edge set

    This works because a graph isomorphism must preserve the edge structure,
    if vertices u and v are connected in G1, then their mapped vertices must
    also be connected in G2.

    The approach is efficient because:
    - It only checks mappings that already preserve vertex degrees

    :param G1: First graph (V1, E1)
    :param G2: Second graph (V2, E2)
    :return: List of valid isomorphisms
    """
    # Unpack the graph tuples into vertices and edges
    V1, E1 = G1  # Graph 1 (V1, E1)
    V2, E2 = G2  # Graph 2 (V2, E2)

    # Get vertex mappings from the first method
    candidates = decision_tree_candidates(G1, G2)

    # Create dictionaries to convert between vertex labels and their positions in the vertex lists
    label_to_index_1 = {label: i for i, label in enumerate(V1)}  # Maps vertex labels to their indices in V1
    label_to_index_2 = {label: i for i, label in enumerate(V2)}  # Maps vertex labels to their indices in V2

    # Transform G2 edges into a set of index pairs and sort them for consistency
    edge_set_2 = {tuple(sorted((label_to_index_2[u], label_to_index_2[v]))) for u, v in E2}

    # Initialize an empty list to store isomorphisms mapped from G1 to G2
    isomorphisms = []

    # Iterate over each candidate mapping
    for mapping in candidates:
        # For each mapping, create a new list of edges for G1 based on the current mapping
        translated_edges = []

        # Iterate over each edge in the first graph
        for u, v in E1:
            # Find the position of the endpoints in the vertices list V1
            u_idx = label_to_index_1[u]  # Get index position of vertex u in V1
            v_idx = label_to_index_1[v]  # Get index position of vertex v in V1

            # Map these vertices to their corresponding vertices in G2 to the current mapping
            mapped_u = mapping[u_idx]  # Get the corresponding vertex index in G2 for u
            mapped_v = mapping[v_idx]  # Get the corresponding vertex index in G2 for v

            translated_edges.append(
                tuple(sorted((mapped_u, mapped_v))))  # Store the edge in normalized form, sorted for consistency

        # Check if the translated edges from G1 match the edges of G2
        if set(translated_edges) == edge_set_2:  # Compare the sets of edges to determine isomorphism
            isomorphisms.append(mapping)  # If edges are the same, this mapping is a valid isomorphism

    # Return the list of all valid graph isomorphisms found
    return isomorphisms


# Method 3
def matrix_isomorphisms(G1, G2):
    """
    Check if two graphs are isomorphic using adjacency matrices and permutation matrices.

    Approach:
    1. Convert both graphs to adjacency matrices A1 and A2
    2. Generate all possible permutations of vertices
    3. For each permutation:
       - Create the corresponding permutation matrix P
       - Apply the formula A1 = P * A2 * P^T
       - The permutation represents an isomorphism

    The matrix equation works because:
    - P represents a relabeling of vertices
    - P*A2*P^T applies this relabeling to the adjacency matrix
    - If the result equals A1, then the structures match exactly

    This is a brute force approach that tries all permutations, so it becomes
    very slow for larger graphs.

    :param G1: First graph (V1, E1)
    :param G2: Second graph (V2, E2)
    :return: Valid permutations representing isomorphisms.
    """
    # Unpack the graph tuples into vertices and edges
    V1, E1 = G1  # Graph 1 (V1, E1)
    V2, E2 = G2  # Graph 2 (V2, E2)

    # Check if the number of vertices is the same in both graphs
    if len(V1) != len(V2):
        return []  # If not, return an empty list

    n = len(V1)  # Number of vertices in the graphs

    # Create adjacency matrices
    A1 = np.zeros((n, n), dtype=int)  # Initialize adjacency matrix for graph 1
    A2 = np.zeros((n, n), dtype=int)  # Initialize adjacency matrix for graph 2

    map1 = {v: i for i, v in enumerate(V1)}  # Map vertex labels to indices for graph 1
    map2 = {v: i for i, v in enumerate(V2)}  # Map vertex labels to indices for graph 2

    # Fill adjacency matrix A1 for graph 1
    for u, v in E1:
        i, j = map1[u], map1[v]  # Get indices of vertices u and v in graph 1
        # Set the edge in the adjacency matrix
        A1[i][j] = 1
        A1[j][i] = 1

    # Fill adjacency matrix A2 for graph 2
    for u, v in E2:  # Get the indices of vertices u and v in graph 2
        i, j = map2[u], map2[v]  # Get indices of vertices u and v in graph 2
        # Set the edge in the adjacency matrix
        A2[i][j] = 1
        A2[j][i] = 1

    valid_mappings = []  # List to store valid mappings

    # Iterate over all permutations of the vertex indices
    for perm in permutations(range(n)):
        # Build permutation matrix P
        P = np.zeros((n, n), dtype=int)
        for i in range(n):  # For each vertex index
            P[i][perm[i]] = 1  # Set the corresponding entry in the permutation matrix

        # Apply the permutation to A2
        transformed = P @ A2 @ P.T  # Matrix multiplication to get the transformed adjacency matrix

        # Check if the transformed matrix equals A1
        if np.array_equal(A1, transformed):
            valid_mappings.append(perm)  # If they are equal, add the permutation to valid mappings
            
    # Return the list of valid mappings
    return valid_mappings


# Method 4
def filtered_matrix_isomorphisms(G1, G2):
    """
    Optimized isomorphism check combining the candidate filtering from Method 1
    with the matrix approach from Method 3.

    Approach:
    1. First, use Method 1 to get candidate mappings that preserve vertex degrees
    2. Convert both graphs to adjacency matrices A1 and A2
    3. For each candidate mapping:
       - Create the corresponding permutation matrix P
       - Apply the formula A1 = P * A2 * P^T
       - If the equation holds, the mapping is an isomorphism

    This hybrid approach combines the best of both worlds:
    - We dramatically reduce the search space by only checking mappings that preserve degrees
    - We use the mathematically elegant matrix formula to verify each candidate

    This should be much faster than Method 3 for most graphs, especially those where
    vertices have different degrees, while still being as accurate.

    :param G1: First graph (V1, E1)
    :param G2: Second graph (V2, E2)
    :return: Valid permutations representing isomorphisms.
    """

    # Unpack the graph tuples into vertices and edges
    V1, E1 = G1  # Graph 1 (V1, E1)
    V2, E2 = G2  # Graph 2 (V2, E2)

    candidate_mappings = decision_tree_candidates(G1, G2)  # Get candidate mappings from Method 1

    # Check if the number of vertices is the same in both graphs
    if len(V1) != len(V2):
        return []  # If not, return an empty list

    n = len(V1)  # Number of vertices in the graphs

    # Create adjacency matrices
    A1 = np.zeros((n, n), dtype=int)  # Initialize adjacency matrix for graph 1
    A2 = np.zeros((n, n), dtype=int)  # Initialize adjacency matrix for graph 2

    # Create dictionaries to convert between vertex labels and their positions in the vertex lists
    idx1 = {node: i for i, node in enumerate(V1)}  # Map vertex labels to indices for graph 1
    idx2 = {node: i for i, node in enumerate(V2)}  # Map vertex labels to indices for graph 2

    # Fill adjacency matrix A1 for graph 1
    for u, v in E1:
        i, j = idx1[u], idx1[v]  # Get indices of vertices u and v in graph 1
        # Set the edge in the adjacency matrix
        A1[i][j] = 1
        A1[j][i] = 1
    # Fill adjacency matrix A2 for graph 2
    for u, v in E2:
        i, j = idx2[u], idx2[v]  # Get indices of vertices u and v in graph 2
        # Set the edge in the adjacency matrix
        A2[i][j] = 1
        A2[j][i] = 1

    valid = []  # List to store valid mappings

    # Iterate over all candidate mappings
    for mapping in candidate_mappings:
        # Build permutation matrix from mapping
        P = np.zeros((n, n), dtype=int)  # Initialize permutation matrix
        for i in range(n):  # For each vertex index
            P[i][mapping[i]] = 1  # Set the corresponding entry in the permutation matrix

        # Apply the permutation to A2
        if np.array_equal(A1, P @ A2 @ P.T):  # Check if the transformed matrix equals A1
            valid.append(mapping)  # If they are equal, add the permutation to valid mappings
    # Return the list of valid mappings
    return valid


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

# Graph 1: A square - worst case scenario for decision tree
nodes3 = ['a', 'b', 'c', 'd']
edges3 = [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')]
G3 = (nodes3, edges3)

# Graph 2: A square
nodes4 = ['k', 'm', 'p', 'r']
edges4 = [('k', 'm'), ('m', 'p'), ('p', 'r'), ('r', 'k')]
G4 = (nodes4, edges4)
# Test the function with the example graphs
print(decision_tree_candidates(G1, G2))
print(isomorphism_checker(G1, G2))
print(matrix_isomorphisms(G1, G2))
print(filtered_matrix_isomorphisms(G1, G2))
