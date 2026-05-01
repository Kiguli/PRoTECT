"""
ARCH-COMP 2026 NLN -- PRoTECT benchmark runner
Runs all NLN benchmarks and writes results.csv to ./results/results.csv
(plural folder name; the portal verifier requires that exact path).
"""
import csv
import json
import os
import re
import subprocess
import sys
import time

# Output path. Folder name MUST be 'results' (plural) and file name
# MUST be 'results.csv' -- the portal verifier checks for exactly that
# pair at submit/results/results.csv.
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
RESULT_CSV = os.path.join(RESULT_DIR, 'results.csv')
BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), 'benchmarks')

# Solver: respect PROTECT_SOLVER env var; default to cvxopt in Docker
SOLVER = os.environ.get('PROTECT_SOLVER', 'cvxopt')

# Per-benchmark timeout. Final-submission cap; the validation pipeline
# tries MOSEK at all degrees first, then CVXOPT only if all MOSEK
# attempts fail (numerical issue / validation rejected).
TIMEOUT = 5000

# -----------------------------------------------------------------------
# Benchmark specifications: (benchmark_id, instance, script, extra_env)
# -----------------------------------------------------------------------
# Both spec versions are reported per family so the report can show the
# "we attempted the 2026 spec, fell back to the original" story:
#   * ROBE25/* (autocatalytic, 2026)        + ROBE21/* (rescaled, original)
#   * CVDP23_uncertain (b in [1,3], 5-D)    + CVDP23 (b=2 fixed) + CVDP22 (b=70)
#   * LOVO25 (Lotka-Volterra, 2026)         + LOVO21 (Lorenz, original)
# LALO20, SPRE22, TRAF22, TSPS25 are unchanged across the spec revision.
BENCHMARKS = [
    # 2026-spec attempts first.
    ('ROBE25', '1',     'ROBE25.py',           {'ROBE25_INSTANCE': '1'}),
    ('ROBE25', '2',     'ROBE25.py',           {'ROBE25_INSTANCE': '2'}),
    ('ROBE25', '3',     'ROBE25.py',           {'ROBE25_INSTANCE': '3'}),
    ('CVDP23', 'b_unc', 'CVDP23_uncertain.py', {}),
    ('CVDP23', 'b2',    'CVDP23.py',           {}),
    ('LOVO25', '',      'LOVO25.py',           {}),
    # Original-spec fallbacks.
    ('ROBE21', '1',     'ROBE21.py',           {'ROBE21_INSTANCE': '1'}),
    ('ROBE21', '2',     'ROBE21.py',           {'ROBE21_INSTANCE': '2'}),
    ('ROBE21', '3',     'ROBE21.py',           {'ROBE21_INSTANCE': '3'}),
    ('CVDP22', '',      'CVDP22.py',           {}),
    ('LOVO21', '',      'LOVO21.py',           {}),
    # Single-version benchmarks.
    ('LALO20', 'W001',  'LALO20.py',           {'LALO20_INSTANCE': 'W001'}),
    ('LALO20', 'W005',  'LALO20.py',           {'LALO20_INSTANCE': 'W005'}),
    ('LALO20', 'W01',   'LALO20.py',           {'LALO20_INSTANCE': 'W01'}),
    ('SPRE22', '',      'SPRE22.py',           {}),
    ('TRAF22', '',      'TRAF22.py',           {}),
    ('TSPS25', '',      'TSPS25.py',           {}),
]


def run_benchmark(script, extra_env):
    """Run a benchmark script and return (stdout, elapsed_time, timed_out)."""
    script_path = os.path.join(BENCHMARK_DIR, script)

    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    env['PROTECT_SOLVER'] = SOLVER
    env['PROTECT_RESULT_DIR'] = RESULT_DIR
    env.update(extra_env)

    wall_start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=TIMEOUT,
        )
        wall_end = time.time()
        return proc.stdout, wall_end - wall_start, False
    except subprocess.TimeoutExpired:
        return '', TIMEOUT, True


def parse_result(stdout):
    """
    Parse benchmark stdout and return (result_flag, elapsed_time).
    result_flag: 1 if a barrier certificate was found, 0 otherwise.
    elapsed_time: float seconds reported by the script itself.
    """
    # Extract elapsed time reported by the script
    m = re.search(r'elapsed time:\s*([\d.eE+\-]+)', stdout)
    elapsed = float(m.group(1)) if m else None

    # Detect success: result dict must contain 'barrier' key (non-error output)
    # An error output looks like: {'error': ..., 'b_degree': ...}
    # A success output contains: {'b_degree': ..., 'barrier': ..., 'gamma': ..., 'lambda': ...}
    if "'barrier'" in stdout and "'error'" not in stdout:
        result_flag = 1
    else:
        result_flag = 0

    return result_flag, elapsed


CSV_FIELDS = ['benchmark', 'instance', 'result', 'time', 'accuracy', 'timesteps',
              'b_degree', 'gamma', 'lambda', 'solver', 'sos_overall', 'barrier']


def write_csv(rows):
    with open(RESULT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_side_result(label):
    """Read the per-benchmark <label>.result.json side-channel file
    written by the benchmark script. Returns a dict (possibly empty)."""
    side_path = os.path.join(RESULT_DIR, f'{label}.result.json')
    if not os.path.isfile(side_path):
        return {}
    try:
        with open(side_path) as f:
            return json.load(f)
    except Exception:
        return {}


def csv_label(benchmark_id, instance):
    """Match the label used by each benchmark when writing its side
    JSON. Most benchmarks key by '<id>_<instance>' or '<id>'."""
    if instance:
        return f'{benchmark_id}_{instance}'
    return benchmark_id


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    rows = []
    write_csv(rows)  # create empty CSV up front so it always exists
    for (benchmark_id, instance, script, extra_env) in BENCHMARKS:
        label = f"{benchmark_id}/{instance}" if instance else benchmark_id
        print(f"Running {label} ...", flush=True)

        stdout, wall_time, timed_out = run_benchmark(script, extra_env)

        side = {}
        if timed_out:
            print(f"  TIMEOUT after {TIMEOUT}s")
            rows.append({
                'benchmark': benchmark_id,
                'instance':  instance,
                'result':    0,
                'time':      TIMEOUT,
                'accuracy':  '',
                'timesteps': '',
                'b_degree':  '',
                'gamma':     '',
                'lambda':    '',
                'solver':    '',
                'sos_overall': '',
                'barrier':   '',
            })
            write_csv(rows)  # persist timeout row immediately
            continue

        result_flag, elapsed = parse_result(stdout)
        reported_time = elapsed if elapsed is not None else round(wall_time, 4)
        side = load_side_result(csv_label(benchmark_id, instance))

        status = 'FOUND' if result_flag else 'NOT FOUND'
        print(f"  {status}  time={reported_time:.4f}s")

        # If validation marked the result as failed (sos_overall=='fail'
        # or 'error' set), force result_flag=0 even if 'barrier' was
        # printed -- a falsified-post-hoc certificate doesn't count.
        if side.get('sos_overall') == 'fail' or 'error' in side:
            result_flag = 0
            status = 'NOT FOUND (validation failed)'
            print(f'  override -> {status}')
        rows.append({
            'benchmark': benchmark_id,
            'instance':  instance,
            'result':    result_flag,
            'time':      round(reported_time, 4),
            'accuracy':  '',
            'timesteps': '',
            'b_degree':  side.get('b_degree', ''),
            'gamma':     side.get('gamma', ''),
            'lambda':    side.get('lambda', ''),
            'solver':    side.get('solver', ''),
            'sos_overall': side.get('sos_overall', ''),
            'barrier':   side.get('barrier', ''),
        })
        write_csv(rows)  # incremental write; survives a mid-run crash

    print(f"\nResults written to {RESULT_CSV}")


if __name__ == '__main__':
    main()
