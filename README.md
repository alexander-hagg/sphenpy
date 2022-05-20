# sphenpy

This first release includes the main code for surrogate-assisted phenotypic niching (SPHEN). SPHEN uses surrogate models to enhance the efficiency of the quality diversity algorithm Voronoi-Elites (VE). VE is a generic optimization algorithm that produces a diverse archive of optimized solutions. SPHEN was introduced in [1] where we used it to generate a diverse set of building footprints and air flows around those footprints, using the Lettuce Lattice-Boltzmann solver [2]. The footprints were optimized to reduce wind speed. The diversity of solutions was created based on the amount of turbulence they produced and the area of the footprint.

Feel free to use and adapt this code. Contributions to this repository are welcome.

[1] Hagg, A., Wilde, D., Asteroth, A., & Bäck, T. (2020, September). Designing air flow with surrogate-assisted phenotypic niching. In International Conference on Parallel Problem Solving from Nature (pp. 140-153). Springer, Cham.

[2] https://github.com/lettucecfd/lettuce
