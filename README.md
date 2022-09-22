# sphenpy

Surrogate-assisted phenotypic niching (SPHEN) uses surrogate models to enhance the efficiency of quality diversity algorithms like MAP-Elites (ME) and Voronoi-Elites (VE). QD are generic optimization algorithms that produce diverse archives of optimized solutions. SPHEN was introduced in [1] where we used it to generate a diverse set of building footprints and air flows around those footprints, using the Lettuce Lattice-Boltzmann solver [2]. The footprints were optimized to reduce wind speed. The diversity of solutions was created based on the amount of turbulence they produced and the area of the footprint.

Content of this archive:
- surrogate-assisted QD
- with MAP-Elites or Voronoi-Elites at its core
- GPy for surrogate-assistance with Gaussian process regression
- Domains: Rastrigin, simple shape domain and shape optimization in 2D air flow

Feel free to use and adapt this code. Contributions to this repository are welcome.

[1] Hagg, A., Wilde, D., Asteroth, A., & Bäck, T. (2020, September). Designing air flow with surrogate-assisted phenotypic niching. In International Conference on Parallel Problem Solving from Nature (pp. 140-153). Springer, Cham.

[2] https://github.com/lettucecfd/lettuce

## Installation
Please use requirements.txt [TODO]
Also refer to https://github.com/jannessm/quadric-mesh-simplification for manual installation, because pip installation of quad_mesh_simplify does not work for some reason
