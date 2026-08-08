What story do I want to tell with the work done in this repo?

We'll have a vignette that argues I've chosen a robust algorithm and arrived at a good solution. It should roughly go in this order:

- Start with naive approach: pick a random straight path (`full_path_optimizer`) and apply gradient descent until validation objective stops improving
  - probably 5-fold cross validation holding out 20-year periods as validation sets and generating bootstrapped paths contrained to those years
- Show that `--smooth` generates more-similar final paths
- Show that the bisection algorithm further improves convergence to the same path
- Internally tune hyperparameters until we're satisfied with the validation objective on the bisection algorithm
- Show that Huber curvature penalty produces more-reasonable paths with a tiny cost (it might improve validation performance, actually, so this might be part of the prebious bullet)
- Now train on the full dataset, which should be smooth and perform well
- Then show the same hyperparameters and algorithm does well compared to Fidelity and Vanguard in the retirement context


