# TODO

- Reimpl sokoban to use JSON/new API.
- Impl other bidir search algos?

# Notes
- I compute the loss over a whole batch (by creating a MergedTrajectory) and do an update (repeated grad_steps times).
- No point in shuffling the MergedTrajectory since we treat it as a batch anyways.
- Under this bootstrap training scheme (and same for the original implementation), the probability that the
  budget increases decreases as the number of problems increases, i.e. there is a trade-off between
  having to solve more problems and increasing the budget less frequently.
- With deterministic algos, for a fixed batch size and seed the order of problems will be the same.
  If we introduce non-determinism then just use a random generator local to the batchloader.
