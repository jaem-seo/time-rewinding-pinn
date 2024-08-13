# Time rewinding of fluid dynamics with PINN
- This repo shows a simple demonstration of time rewinding of fluid dynamics with PINN.
- Although the time-reversed simulation has been an ill-posed problem, PINN can provide a rough estimation of the past from the noisy observation.

# Note
- This is proof-of-concept work, and there are still issues with seed dependency. We recommend training on different seeds and choosing the one with the smallest loss, which is still practically reasonable.

# References
- J. Seo, "Past rewinding of fluid dynamics from noisy observation via physics-informed neural computing." [Phys. Rev. E 110 (2024) 025302](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.110.025302).
