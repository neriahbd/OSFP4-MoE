# Contributing

Keep this repository focused on explaining the three-fork integration. Link implementation changes to their owning fork and branch.

When updating the guide, verify linked paths and APIs against the documented revisions, update `revisions.tsv` deliberately, and distinguish implemented behavior from executed validation. Do not describe dense-model paper results as MoE results.

For a reproducible issue, include the three commit IDs, model and dataset revisions, format, optimizer, calibration settings, hardware, environment, backend, parallelism, command and relevant error. Remove credentials and private data from logs.

Before submitting documentation changes, check relative links, shell syntax with `bash -n scripts/checkout-forks.sh`, and whitespace with `git diff --check`. Run GPU tests and model evaluations when making new accuracy, performance, or hardware-support claims.

Changes to the three implementations must follow their own contribution and licensing policies.
