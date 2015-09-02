How to use these scripts:

Add this to your profile:

export SBATCH_MAIL=xyz@xyz.com
export SBATCH_QUEUE=queue_name

In the sbatch script to run, substitute:

SBATCH_MAIL for actual email,
and SBATCH_QUEUE for actual queue.

To run:

  cd build
   ../scripts/run_job.sh script-name


where script-name corresponds to a file of the name script-name.sbatch.

The run_job.sh textually replaces the template values with the actual
values and pipes this into the standard input of sbatch.

Example:

  cd build
  ../scripts/run_job.sh octree




