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


def compare_method_results(G1, G2):
    """
    Verifies that Methods 2, 3, and 4 produce the same results for given graph pairs.

    This function:
    1. Runs all three isomorphism checking methods
    2. Normalizes the results to ensure they're comparable
    3. Compares the sets of mappings from each method
    4. Reports whether all methods agree

    :param G1: First graph (V1, E1)
    :param G2: Second graph (V2, E2)
    :return: True if all methods produce the same results, False otherwise
    """
    # Run all three isomorphism checking methods
    method2_results = isomorphism_checker(G1, G2)
    method3_results = matrix_isomorphisms(G1, G2)
    method4_results = filtered_matrix_isomorphisms(G1, G2)

    # Convert each result to a set of tuples for comparison
    method2_set = set(map(tuple, method2_results))
    method3_set = set(map(tuple, method3_results))
    method4_set = set(map(tuple, method4_results))

    # Check if all methods produce the same results
    methods_match = (method2_set == method3_set) and (method3_set == method4_set)

    # Print details about consistency
    print("\nMETHOD CONSISTENCY CHECK:")
    print(f"Methods 2, 3, and 4 produce the same results: {'YES' if methods_match else 'NO'}")

    # If methods don't match, show the differences
    if not methods_match:
        print("Differences found:")
        if method2_set != method3_set:
            print(f"Method 2 vs Method 3: Different results")
        if method2_set != method4_set:
            print(f"Method 2 vs Method 4: Different results")
        if method3_set != method4_set:
            print(f"Method 3 vs Method 4: Different results")

    return methods_match


# Graph pairs for testing
# Pair 1
# Graph 1: Simple 4-vertex graph
V1 = ['a', 'b', 'c', 'd']
E1 = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'd'), ('c', 'd')]
G1 = (V1, E1)  # Graph 1 tuple

# Graph 2: Simple 4-vertex graph
V2 = ['e', 'f', 'g', 'h']
E2 = [('e', 'g'), ('e', 'h'), ('f', 'g'), ('f', 'h'), ('g', 'h')]
G2 = (V2, E2)  # Graph 2 tuple

# Pair 2
# Graph 1: 6-vertex graph
V3 = ['a', 'b', 'c', 'd', 'e', 'f']
E3 = [('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'e'), ('b', 'd'),
      ('b', 'e'), ('b', 'f'), ('c', 'd'), ('c', 'f'), ('d', 'f'), ('e', 'f')]
G3 = (V3, E3)  # Graph 3 tuple

# Graph 2: 6-vertex graph
V4 = ['p', 'q', 'r', 's', 't', 'u']
E4 = [('p', 'q'), ('p', 's'), ('p', 't'), ('p', 'u'), ('q', 's'),
      ('q', 't'), ('r', 's'), ('r', 't'), ('r', 'u'), ('s', 'u'), ('t', 'u')]
G4 = (V4, E4)  # Graph 4 tuple

# Pair 3
# Graph 1: 5-vertex wheel graph
V5 = ['a', 'b', 'c', 'd', 'e']
E5 = [('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'e'),
      ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'b')]
G5 = (V5, E5)  # Graph 5 tuple

# Graph 2: 5-vertex wheel graph
V6 = ['v', 'w', 'x', 'y', 'z']
E6 = [('v', 'w'), ('v', 'x'), ('v', 'y'), ('v', 'z'),
      ('w', 'x'), ('x', 'y'), ('y', 'z'), ('z', 'w')]
G6 = (V6, E6)  # Graph 6 tuple

# Pair 4
# Graph 1: 4-vertex complete graph
V7 = ['a', 'b', 'c', 'd']
E7 = [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('c', 'd')]
G7 = (V7, E7)  # Graph 7 tuple

# Graph 2: 4-vertex complete graph
V8 = ['w', 'x', 'y', 'z']
E8 = [('w', 'x'), ('w', 'y'), ('w', 'z'), ('x', 'y'), ('y', 'z')]
G8 = (V8, E8)  # Graph 8 tuple

# Pair 5
# Graph 1: Square without diagonal non-isomorphic graph
V9 = ['A', 'B', 'C', 'D']
E9 = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('A', 'C')]
G9 = (V9, E9)  # Graph 9 tuple

# Graph 2: Square without diagonal non-isomorphic graph
V10 = ['W', 'X', 'Y', 'Z']
E10 = [('W', 'X'), ('X', 'Y'), ('Y', 'Z'), ('Z', 'W')]
G10 = (V10, E10)  # Graph 10 tuple


def run_methods():
    """
    Test and display isomorphism results for graph pairs.
    Runs all four methods and prints the results including all mappings.
    """

    # Test Case 1: Simple 4-vertex graphs
    print("\n" + "=" * 60)
    print("TEST 1: Simple 4-vertex graphs")
    print("=" * 60)
    print(f"Graph 1: Vertices {V1}, Edges {E1}")
    print(f"Graph 2: Vertices {V2}, Edges {E2}")

    # Get results for each algorithm
    dt_candidates = decision_tree_candidates(G1, G2)
    iso_results = isomorphism_checker(G1, G2)
    matrix_results = matrix_isomorphisms(G1, G2)
    filtered_results = filtered_matrix_isomorphisms(G1, G2)

    # Display results
    print("\nRESULTS:")
    print(f"Method 1 - Decision Tree Candidates ({len(dt_candidates)}):")
    for i, mapping in enumerate(dt_candidates, 1):
        print(f"  Mapping {i}: {V1} → {[V2[idx] for idx in mapping]}")

    print(f"\nMethod 2 - Isomorphism Checker ({len(iso_results)}):")
    for i, mapping in enumerate(iso_results, 1):
        print(f"  Isomorphism {i}: {V1} → {[V2[idx] for idx in mapping]}")

    print(f"\nMethod 3 - Matrix Approach ({len(matrix_results)}):")
    for i, mapping in enumerate(matrix_results, 1):
        print(f"  Isomorphism {i}: {V1} → {[V2[idx] for idx in mapping]}")

    print(f"\nMethod 4 - Filtered Matrix ({len(filtered_results)}):")
    for i, mapping in enumerate(filtered_results, 1):
        print(f"  Isomorphism {i}: {V1} → {[V2[idx] for idx in mapping]}")

    print(f"\nGraphs are {'ISOMORPHIC' if len(iso_results) > 0 else 'NOT ISOMORPHIC'}")

    # Verify that methods 2, 3, and 4 produce identical results
    compare_method_results(G1, G2)

    # Test Case 2: 6-vertex graph
    print("\n" + "=" * 60)
    print("TEST 2: 6-vertex graphs")
    print("=" * 60)
    print(f"Graph 1: Vertices {V3}, Edges {E3}")
    print(f"Graph 2: Vertices {V4}, Edges {E4}")

    # Get results for each algorithm
    dt_candidates = decision_tree_candidates(G3, G4)
    iso_results = isomorphism_checker(G3, G4)
    matrix_results = matrix_isomorphisms(G3, G4)
    filtered_results = filtered_matrix_isomorphisms(G3, G4)

    # Display results
    print("\nRESULTS:")
    print(f"Method 1 - Decision Tree Candidates ({len(dt_candidates)}):")
    for i, mapping in enumerate(dt_candidates, 1):
        print(f"  Mapping {i}: {V3} → {[V4[idx] for idx in mapping]}")

    print(f"\nMethod 2 - Isomorphism Checker ({len(iso_results)}):")
    for i, mapping in enumerate(iso_results, 1):
        print(f"  Isomorphism {i}: {V3} → {[V4[idx] for idx in mapping]}")

    print(f"\nMethod 3 - Matrix Approach ({len(matrix_results)}):")
    for i, mapping in enumerate(matrix_results, 1):
        print(f"  Isomorphism {i}: {V3} → {[V4[idx] for idx in mapping]}")

    print(f"\nMethod 4 - Filtered Matrix ({len(filtered_results)}):")
    for i, mapping in enumerate(filtered_results, 1):
        print(f"  Isomorphism {i}: {V3} → {[V4[idx] for idx in mapping]}")

    print(f"\nGraphs are {'ISOMORPHIC' if len(iso_results) > 0 else 'NOT ISOMORPHIC'}")

    # Verify that methods 2, 3, and 4 produce identical results
    compare_method_results(G3, G4)

    # Apply the same pattern for the remaining test cases...
    # Test Case 3:  5-vertex wheel graphs
    print("\n" + "=" * 60)
    print("TEST 3: 5-vertex wheel graph")
    print("=" * 60)
    print(f"Graph 1: Vertices {V5}, Edges {E5}")
    print(f"Graph 2: Vertices {V6}, Edges {E6}")

    # Get results for each algorithm
    dt_candidates = decision_tree_candidates(G5, G6)
    iso_results = isomorphism_checker(G5, G6)
    matrix_results = matrix_isomorphisms(G5, G6)
    filtered_results = filtered_matrix_isomorphisms(G5, G6)

    # Display results
    print("\nRESULTS:")
    print(f"Method 1 - Decision Tree Candidates ({len(dt_candidates)}):")
    for i, mapping in enumerate(dt_candidates, 1):
        print(f"  Mapping {i}: {V5} → {[V6[idx] for idx in mapping]}")

    print(f"\nMethod 2 - Isomorphism Checker ({len(iso_results)}):")
    for i, mapping in enumerate(iso_results, 1):
        print(f"  Isomorphism {i}: {V5} → {[V6[idx] for idx in mapping]}")

    print(f"\nMethod 3 - Matrix Approach ({len(matrix_results)}):")
    for i, mapping in enumerate(matrix_results, 1):
        print(f"  Isomorphism {i}: {V5} → {[V6[idx] for idx in mapping]}")

    print(f"\nMethod 4 - Filtered Matrix ({len(filtered_results)}):")
    for i, mapping in enumerate(filtered_results, 1):
        print(f"  Isomorphism {i}: {V5} → {[V6[idx] for idx in mapping]}")

    print(f"\nGraphs are {'ISOMORPHIC' if len(iso_results) > 0 else 'NOT ISOMORPHIC'}")

    # Verify that methods 2, 3, and 4 produce identical results
    compare_method_results(G5, G6)

    # Test Case 4: 4-vertex complete graphs
    print("\n" + "=" * 60)
    print("TEST 4: 4-vertex complete graphs")
    print("=" * 60)
    print(f"Graph 1: Vertices {V7}, Edges {E7}")
    print(f"Graph 2: Vertices {V8}, Edges {E8}")

    # Get results for each algorithm
    dt_candidates = decision_tree_candidates(G7, G8)
    iso_results = isomorphism_checker(G7, G8)
    matrix_results = matrix_isomorphisms(G7, G8)
    filtered_results = filtered_matrix_isomorphisms(G7, G8)

    # Display results
    print("\nRESULTS:")
    print(f"Method 1 - Decision Tree Candidates ({len(dt_candidates)}):")
    for i, mapping in enumerate(dt_candidates, 1):
        print(f"  Mapping {i}: {V7} → {[V8[idx] for idx in mapping]}")

    print(f"\nMethod 2 - Isomorphism Checker ({len(iso_results)}):")
    for i, mapping in enumerate(iso_results, 1):
        print(f"  Isomorphism {i}: {V7} → {[V8[idx] for idx in mapping]}")

    print(f"\nMethod 3 - Matrix Approach ({len(matrix_results)}):")
    for i, mapping in enumerate(matrix_results, 1):
        print(f"  Isomorphism {i}: {V7} → {[V8[idx] for idx in mapping]}")

    print(f"\nMethod 4 - Filtered Matrix ({len(filtered_results)}):")
    for i, mapping in enumerate(filtered_results, 1):
        print(f"  Isomorphism {i}: {V7} → {[V8[idx] for idx in mapping]}")

    print(f"\nGraphs are {'ISOMORPHIC' if len(iso_results) > 0 else 'NOT ISOMORPHIC'}")

    # Verify that methods 2, 3, and 4 produce identical results
    compare_method_results(G7, G8)

    # Test Case 5: Square without diagonal non-isomorphic graphs
    print("\n" + "=" * 60)
    print("TEST 5: Square without diagonal non-isomorphic graphs")
    print("=" * 60)
    print(f"Graph 1: Vertices {V9}, Edges {E9}")
    print(f"Graph 2: Vertices {V10}, Edges {E10}")

    # Get results for each algorithm
    dt_candidates = decision_tree_candidates(G9, G10)
    iso_results = isomorphism_checker(G9, G10)
    matrix_results = matrix_isomorphisms(G9, G10)
    filtered_results = filtered_matrix_isomorphisms(G9, G10)

    # Display results
    print("\nRESULTS:")
    print(f"Method 1 - Decision Tree Candidates ({len(dt_candidates)}):")
    for i, mapping in enumerate(dt_candidates, 1):
        print(f"  Mapping {i}: {V9} → {[V10[idx] for idx in mapping]}")

    print(f"\nMethod 2 - Isomorphism Checker ({len(iso_results)}):")
    for i, mapping in enumerate(iso_results, 1):
        print(f"  Isomorphism {i}: {V9} → {[V10[idx] for idx in mapping]}")

    print(f"\nMethod 3 - Matrix Approach ({len(matrix_results)}):")
    for i, mapping in enumerate(matrix_results, 1):
        print(f"  Isomorphism {i}: {V9} → {[V10[idx] for idx in mapping]}")

    print(f"\nMethod 4 - Filtered Matrix ({len(filtered_results)}):")
    for i, mapping in enumerate(filtered_results, 1):
        print(f"  Isomorphism {i}: {V9} → {[V10[idx] for idx in mapping]}")

    print(f"\nGraphs are {'ISOMORPHIC' if len(iso_results) > 0 else 'NOT ISOMORPHIC'}")

    # Verify that methods 2, 3, and 4 produce identical results
    compare_method_results(G9, G10)


run_methods()

