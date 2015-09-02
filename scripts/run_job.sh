#!/usr/bin/env bash
cat ../scripts/$1.sbatch  | sed "s/SBATCH_QUEUE/$SBATCH_QUEUE/g" | sed "s/SBATCH_MAIL/$SBATCH_MAIL/g"  | sbatch
