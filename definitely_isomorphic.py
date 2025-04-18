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
    # Return the dictionary with the degree (a number of times a vertex appears in edges)
    return counts


def decision_tree_candidates(G1, G2):
    """
    Generate candidate mappings between two graphs assumed to be likely isomorphic.

    This function uses a decision tree approach:
        1. Match nodes from G1 to nodes in G2 based on degree. (Check if vertices have the same degree distribution)
        2. Creating a mapping dictionary based on matching degrees
        3. Recursively build all valid combinations without reusing nodes.

    Accepts two graphs in the form (V, E), where V is a list of vertices,
    and E is a list of edges.
    :param G1: First graph (V1, E1)
    :param G2: Second graph (V2, E2)
    :return: A sorted list of valid candidate mappings based on degrees.
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
    # Sort the mappings to ensure they are in order
    return sorted([tuple(m) for m in mappings])


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

# Test the function with the example graphs
print(decision_tree_candidates(G1, G2))

def isomorphism_checker(G1, G2):
    """
    Checks which candidate mappings from Method 1 are actual isomorphisms.

    This function translates the edges of G1 using each mapping,
    and compares them to the edges in G2. If the edge sets match,
    the mapping is considered a valid isomorphism.

    Parameters:
        G1 (tuple): (V1, E1) — first graph
        G2 (tuple): (V2, E2) — second graph

    Returns:
        list: list of valid isomorphisms (mappings that preserve edge structure)
    """
    # Unpack the graph tuples into vertices and edges
    V1, E1 = G1  
    V2, E2 = G2 

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
            
            translated_edges.append(tuple(sorted((mapped_u, mapped_v))))  # Store the edge in normalized form, sorted for consistency
        
        # Check if the translated edges from G1 match the edges of G2
        if set(translated_edges) == edge_set_2:  # Compare the sets of edges to determine isomorphism
            isomorphisms.append(mapping)  # If edges are the same, this mapping is a valid isomorphism
            
    # Return the list of all valid graph isomorphisms found
    return isomorphisms 

# Example usage
print(isomorphism_checker(G1, G2))
