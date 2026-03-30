#/bin/bash!
./jupyter-compute 16805 --jupyter_args="--ServerApp.allow_origin=* --IdentityProvider.token=3b30677a66c7a27927bb717f152b990f18bf8998c9644b1f" --nodes=1 --cpus-per-task=6 --mem=150G --time=03:00:00 --partition="epyc2" --qos="job_gratis" $@
