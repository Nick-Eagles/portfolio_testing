# Metric Screening

This screen perturbs one hyperparameter at a time around the default bisection
glide-path optimizer settings. It uses only the good starting path in each of
the five `year-cv` folds. The similarity metric is mean pairwise across-fold
distance among final paths; lower distance means the fold-specific optimized
paths agree more.

## Anchors

The single baseline is the current default hyperparameter set:

- `learning_rate = 0.040000`
- `bisections = 4`
- `gradient_steps = 15`
- `curvature_penalty = 0.000250`
- `early_stop = false`
- `smooth = true`
- `smoothing_strength = 0.900000`

Baseline start validation: `-0.012222`
Baseline final validation: `-0.010033`

Baseline start within-fold distance: `0.511487`
Baseline final across-fold distance: `0.396522`

The validation component is normalized by the baseline's own start-to-final
validation improvement. The similarity component is normalized by the
baseline's own start-to-final movement from within-fold initial-start
dispersion to final across-fold path distance. This makes both normalized
coordinates explicitly relative to the default hyperparameters under test.

The proposed weighted score is:

`validation_progress + 2 * similarity_progress`

where validation progress uses the baseline start-to-final validation range, and
similarity progress uses the within-start to final-across-fold distance range.
Because path distance is lower-is-better, its progress term is
direction-reversed.

## Top 10

| changed | experiment | learning_rate | early_stop | gradient_steps | bisections | curvature_penalty | smoothing_strength | mean_final_validation_canonical | final_across_fold_path_distance | validation_progress | similarity_progress | combined_score_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bisections = 5 | bisections_5 | 0.040000 | False | 15 | 5 | 0.000250 | 0.900000 | -0.009812 | 0.384671 | 1.100836 | 1.103087 | 3.307010 |
| bisections = 6 | bisections_6 | 0.040000 | False | 15 | 6 | 0.000250 | 0.900000 | -0.010171 | 0.381253 | 0.937122 | 1.132822 | 3.202765 |
| baseline | baseline | 0.040000 | False | 15 | 4 | 0.000250 | 0.900000 | -0.010033 | 0.396522 | 1.000000 | 1.000000 | 3.000000 |
| learning rate = 0.05 | learning_rate_005 | 0.050000 | False | 15 | 4 | 0.000250 | 0.900000 | -0.009987 | 0.399125 | 1.020788 | 0.977364 | 2.975516 |
| curvature = 0.0005 | curvature_0005 | 0.040000 | False | 15 | 4 | 0.000500 | 0.900000 | -0.010107 | 0.398547 | 0.966038 | 0.982389 | 2.930817 |
| smoothing = 0.8 | smooth_08 | 0.040000 | False | 15 | 4 | 0.000250 | 0.800000 | -0.010050 | 0.400168 | 0.992397 | 0.968290 | 2.928977 |
| curvature = 0 | curvature_0 | 0.040000 | False | 15 | 4 | 0.000000 | 0.900000 | -0.009940 | 0.403186 | 1.042345 | 0.942037 | 2.926419 |
| smoothing = 1.0 | smooth_10 | 0.040000 | False | 15 | 4 | 0.000250 | 1.000000 | -0.010083 | 0.401322 | 0.977061 | 0.958256 | 2.893573 |
| smoothing = 0.6 | smooth_06 | 0.040000 | False | 15 | 4 | 0.000250 | 0.600000 | -0.010035 | 0.403081 | 0.998932 | 0.942954 | 2.884841 |
| curvature = 0.00075 | curvature_00075 | 0.040000 | False | 15 | 4 | 0.000750 | 0.900000 | -0.010233 | 0.400071 | 0.908812 | 0.969133 | 2.847079 |

## All Results

| changed | experiment | learning_rate | early_stop | gradient_steps | bisections | curvature_penalty | smoothing_strength | mean_final_validation_canonical | final_across_fold_path_distance | validation_progress | similarity_progress | combined_score_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bisections = 5 | bisections_5 | 0.040000 | False | 15 | 5 | 0.000250 | 0.900000 | -0.009812 | 0.384671 | 1.100836 | 1.103087 | 3.307010 |
| bisections = 6 | bisections_6 | 0.040000 | False | 15 | 6 | 0.000250 | 0.900000 | -0.010171 | 0.381253 | 0.937122 | 1.132822 | 3.202765 |
| baseline | baseline | 0.040000 | False | 15 | 4 | 0.000250 | 0.900000 | -0.010033 | 0.396522 | 1.000000 | 1.000000 | 3.000000 |
| learning rate = 0.05 | learning_rate_005 | 0.050000 | False | 15 | 4 | 0.000250 | 0.900000 | -0.009987 | 0.399125 | 1.020788 | 0.977364 | 2.975516 |
| curvature = 0.0005 | curvature_0005 | 0.040000 | False | 15 | 4 | 0.000500 | 0.900000 | -0.010107 | 0.398547 | 0.966038 | 0.982389 | 2.930817 |
| smoothing = 0.8 | smooth_08 | 0.040000 | False | 15 | 4 | 0.000250 | 0.800000 | -0.010050 | 0.400168 | 0.992397 | 0.968290 | 2.928977 |
| curvature = 0 | curvature_0 | 0.040000 | False | 15 | 4 | 0.000000 | 0.900000 | -0.009940 | 0.403186 | 1.042345 | 0.942037 | 2.926419 |
| smoothing = 1.0 | smooth_10 | 0.040000 | False | 15 | 4 | 0.000250 | 1.000000 | -0.010083 | 0.401322 | 0.977061 | 0.958256 | 2.893573 |
| smoothing = 0.6 | smooth_06 | 0.040000 | False | 15 | 4 | 0.000250 | 0.600000 | -0.010035 | 0.403081 | 0.998932 | 0.942954 | 2.884841 |
| curvature = 0.00075 | curvature_00075 | 0.040000 | False | 15 | 4 | 0.000750 | 0.900000 | -0.010233 | 0.400071 | 0.908812 | 0.969133 | 2.847079 |
| gradient steps = 20 | gradient_steps_15 | 0.040000 | False | 20 | 4 | 0.000250 | 0.900000 | -0.010233 | 0.400960 | 0.908443 | 0.961403 | 2.831249 |
| gradient steps = 10 | gradient_steps_10 | 0.040000 | False | 10 | 4 | 0.000250 | 0.900000 | -0.010031 | 0.407547 | 1.001120 | 0.904107 | 2.809335 |
| learning rate = 0.06 | learning_rate_006 | 0.060000 | False | 15 | 4 | 0.000250 | 0.900000 | -0.010125 | 0.405850 | 0.957870 | 0.918866 | 2.795602 |
| gradient steps = 30 | gradient_steps_30 | 0.040000 | False | 30 | 4 | 0.000250 | 0.900000 | -0.010323 | 0.401186 | 0.867666 | 0.959436 | 2.786538 |
| curvature = 0.001 | curvature_001 | 0.040000 | False | 15 | 4 | 0.001000 | 0.900000 | -0.010333 | 0.402308 | 0.862941 | 0.949671 | 2.762284 |
| learning rate = 0.03 | learning_rate_003 | 0.030000 | False | 15 | 4 | 0.000250 | 0.900000 | -0.010132 | 0.409488 | 0.954971 | 0.887222 | 2.729414 |
| smooth = false | smooth_false | 0.040000 | False | 15 | 4 | 0.000250 | 0.000000 | -0.010189 | 0.415441 | 0.928895 | 0.835436 | 2.599766 |
| early_stop = true | early_stop_true | 0.040000 | True | 15 | 4 | 0.000250 | 0.900000 | -0.010433 | 0.420142 | 0.817256 | 0.794548 | 2.406351 |
| bisections = 3 | bisections_3 | 0.040000 | False | 15 | 3 | 0.000250 | 0.900000 | -0.010733 | 0.418160 | 0.680285 | 0.811791 | 2.303867 |
| learning rate = 0.02 | learning_rate_002 | 0.020000 | False | 15 | 4 | 0.000250 | 0.900000 | -0.010900 | 0.425283 | 0.603638 | 0.749826 | 2.103290 |

Best score in this screen: `bisections_5` with
`3.307`.

## Plots

- [Validation vs similarity](validation_vs_similarity.png)
- [Normalized components](normalized_components.pdf)
- [Top 10 weighted scores](top_10_weighted_scores.png)
- [Baseline paths](baseline/start_and_final_paths.png)

## Initial Read

This is a perturbation screen around a favored baseline, not a proof of a local
maximum in hyperparameter space. The default settings should still sit in a
good part of the tradeoff: perturbations that improve one coordinate should make
their cost in the other coordinate easy to see.
