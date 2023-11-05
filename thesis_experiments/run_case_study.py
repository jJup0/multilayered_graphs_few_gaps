import os
import subprocess


file_dir = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(file_dir, f"sge_log.txt")
sge_wrapper_file_path = os.path.join(file_dir, "case_study_wrapper.sh")

args = " ".join(
    [
        "qsub",
        f"-N case_study",
        "-l bc5 -l mem_free=8G -l h_vmem=8G",
        "-r y",
        f"-e {log_path} -o {log_path}",
        f"{sge_wrapper_file_path} long_timeout",
    ]
).split()

subprocess.run(args)
