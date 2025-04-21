# Method Performance Analysis

## Worst: Brute Force (Method 3)
- It generates all n! possible vertex permutations
- Performs exhaustive checking regardless of graph structure
- Impractical beyond tiny graphs

**Performance**:
- 5 nodes → 120 candidates
- 8 nodes → 40,320 candidates 
- 10 nodes → 3,628,800 candidates
- ~300 matrix operations

## Best: Decision Tree (Methods 1+2)
The combined approach of Methods 1 and 2 proves most efficient because:
1. **Filtering**: Method 1 eliminates impossible mappings by:
   - Matching vertex degrees
   - Preserving degree sequences
2. **Verification**: Method 2 then:
   - Translates edges efficiently
   - Performs quick comparisons
   - Uses index-based operations

**Performance**:
For a typical 6-node graph:
- Brute force: 720 candidates
- Decision tree: Often <20 candidates
- Verification: Only needed for plausible mappings

**Exception Case**:
For polygon graphs (or similar) where all nodes have identical degrees, this method has n! performance (similar to brute-force) as all permutations become possible candidates.
