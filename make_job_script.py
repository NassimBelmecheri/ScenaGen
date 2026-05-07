import itertools

frames = [10, 20, 30, 40, 60, 80, 100]
objects = [10, 20, 30, 40, 60, 80, 100]
seeds = [101, 102, 103]
refinements = 4

for f, o, s in itertools.product(frames, objects, seeds):
    job_name = f"scenegen_f{f}_o{o}_s{s}"

    cmd = f"uv run python3 ScenaGen_CLI.py --generate --num-objects {o} --num-frames {f} --seed {s} --refinements {refinements} --append"
    print(cmd)
