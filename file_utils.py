import os
import subprocess

def resolve_tmpdir():
    if 'SLURM_TMPDIR' in os.environ:
        return os.environ['SLURM_TMPDIR']
    else:
        result = subprocess.run(['squeue', '-u', 'georgeth', '-n', 'interactive'],
                                stdout=subprocess.PIPE)
        job_id = result.stdout.decode('utf-8').split('\n')[1].split(' ')[1]
        return f'/Tmp/slurm.{job_id}.0'
