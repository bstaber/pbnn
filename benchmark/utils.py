import getpass
import subprocess
import time
import uuid

from jinja2 import Template


def get_running_jobs_count():
    result = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "--noheader"],
        capture_output=True,
        text=True,
        check=True,
    )

    if result.stdout == "":
        return 0
    else:
        running_jobs_count = len(result.stdout.strip().split("\n"))
    return running_jobs_count


def barrier(jobs_limit: int = 1):
    while True:
        running_jobs = get_running_jobs_count()
        if running_jobs >= jobs_limit:
            print(f"{running_jobs} jobs still running. Waiting for jobs to complete...")
            time.sleep(60)
        else:
            break


def launch(
    template_name: str,
    template_folder="pipelines/templates",
    command="sbatch",
    **params,
):
    with open(f"{template_folder}/{template_name}.sh", "r") as f:
        template = Template(f.read())

    text = template.render(**params)

    unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    slurm_file = f"{template_name}_{unique_id}.slurm"
    with open(slurm_file, "w") as f:
        _ = f.write(text)

    res = subprocess.run([command, slurm_file], capture_output=True, text=True)

    return res
