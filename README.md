# MA_MOES
# Multi-agent multi-objective ergodic coverage

This work contains a branch and bound approach to multi-agent multi-objective ergodic search. It aims to find the optimal allocation that minimizes the maximum individual ergodicity.

It also presents an approach to leverage similarity between information maps reduce the runtime of the algorithm. Further, a clustering approach based on minimal bounding sphere is also presented. Finally, a naive approach for post-processing trajectories followed by branch and bound is used to compute the minmax optimal allocation that considers inter-agent collisions.

# Setting up the environment
```
conda env create -f environment.yml
```

The algorithm is tested on randomly generated test cases with 4 agents and random start poses. 

To run the optimal branch and bound approach:
```
python BB_optimized.py
```

To run the branch and bound with similarity clustering:
```
python similarity_between_maps.py
```
