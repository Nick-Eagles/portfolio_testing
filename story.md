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

Results we have so far:

- "Full-path optimizer" was marginally better on validation score than bisection algorithm, but the bisection algorithm was convergent to the same starting path regardless of initialization for a small validation cost. I proceeded with tuning the bisection algorithm at that point (see `consolidated_path_optimizer/tuning/report.md`)
- Learning rate 0.04 clearly improves training objective and sufficient for good convergence
- While some validation + path-similarity scores suggested as high as 6 bisections was beneficial, paths clearly looked jagged, overly complex, and generally unrealistic. Less than 4 clearly hurt performance in a way that looked like underfitting
- Curvature-penalty 0.00025 seemed to be a sweet spot of good score and lack of unrealistic jaggedness in the paths. The other curvature parameter was tuned subjectively (ability to control jaggedness at the proper length scale)
- `--early-stop` produced worse scores for good hyperparameters and had less predictable impact on optimization as a function of hyperparameters, so it was not pursued further.


