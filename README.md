# PhysicalModelsOfLivingSystems
Repository for the Physical Models Of Living Systems course od the Physics of Data master degree.
The lectures are given by UniPD professor Samir Suweiss.
In the Repo the weekly assigments and the final project will be uploaded as the semester proceed, the topics inspected up to now are:

- **Consumer Model**: with 1 species and 1 abiotic resource, more specifically the tasks were:
    - Solve the Quasi Stationary Approximation of the Consumer Resource Model with 1 species and 1 abiotic resource and compare it numerically with the                  full equation.
    - Write the Fokker Plank Equation associated to the stochastic logistic equation with environmental noise and solve the stationary solution.
  
- **Species Abundance Distribution**: 
  perform an analysis of the **Species Abundance Distribution (SAD)** of a dataset of species abundance in a community. 

  The dataset is composed by the number of individuals (the abundance) of 4283 species, sampled in the **1%** of the total area of a forest.
  The dataset is available in the file `RSA_sampled_1percent.xlsx`.

  In particular, from the data of this area, we want to infer the number of species at the whole scale (p=1).
  
- **Lotka-Volterra equations**: Describe the system known as the Lotka-Volterra equations that reproduces the dynamics of a predator-prey system. The first equation describes the dynamics of the prey population, while the second equation goes about the dynamics of the predator population. The parameter $\alpha$   stands for the growth rate of the prey population, while the parameter $c$ describes the death rate of the predator population. The parameter $\mathcal{p}$  is interpreted as the interaction between the predator and the prey.

  The tasks we will solve are the following:


   - Find the stationary solutions of the system.

   - Perform a linear stability analysis of the stationary solutions and find out wether we have stable solutions.
   - Simulate the equations of the system with different parameters trying to find a situation where we observe sustained oscillations. 
