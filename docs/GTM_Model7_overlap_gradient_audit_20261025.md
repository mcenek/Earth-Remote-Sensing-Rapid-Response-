# Read-only training objective check

Written before the diagnostic. This does not authorize a new fit.

Measure BCE and unweighted positive overlap-loss gradients in the completed v8 update 2,400 model. Use only the first sixteen batches of its exact saved TRAIN draw schedule. Load the frozen normalizer and checkpoint through their saved identities. Do not update weights. No development or test arrays are opened.

Reject changed metadata, trainer, checkpoint, schedule or arrays; missing successful replay; non-TRAIN identity; incorrect four-plume/four-clear balance; nonfinite losses or gradients; and altered model tensors. Record batch losses, gradient L2 norms, weighted-overlap/BCE norm ratio at the actual 0.1 coefficient, and gradient cosine. Report all sixteen batches. This is a mechanism diagnostic, not evidence that another loss will improve development predictions. Save a repeatable script and receipt before deciding on another trial.
