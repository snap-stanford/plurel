import glob
import os
import shutil
import subprocess
import sysconfig

import pytest

from plurel.config import Choices, Config, DatabaseParams
from plurel.dataset import SyntheticDataset
from plurel.utils import set_random_seed
from rt.tasks import get_tasks_info, is_valid_db

# All tests share ~/scratch/ state via module-scoped fixtures, so they
# must run on the same xdist worker to avoid races.
pytestmark = pytest.mark.xdist_group("dev_run")

TAG = "pytest"
NUM_DBS = 10
NUM_TEST_DBS = 1
SEED_OFFSET = 0
SEEDS = list(range(SEED_OFFSET, SEED_OFFSET + NUM_DBS))

HOME = os.path.expanduser("~")
SCRATCH_RELBENCH = os.path.join(HOME, "scratch", "relbench")
SCRATCH_PRE = os.path.join(HOME, "scratch", "pre")
RUSTLER_DIR = os.path.realpath("rustler")

# Small tables so the test runs fast
SMALL_DB_PARAMS = DatabaseParams(
    num_tables_choices=Choices(kind="range", value=[3, 5]),
    num_rows_entity_table_choices=Choices(kind="range", value=[40, 80]),
    num_rows_activity_table_choices=Choices(kind="range", value=[100, 200]),
)

PRE_OUTPUT_FILES = [
    "text.json",
    "text_map.json",
    "column_index.json",
    "table_info.json",
    "nodes.rkyv",
    "offsets.rkyv",
    "p2f_adj.rkyv",
]


def _db_name(seed: int) -> str:
    return f"rel-synthetic-test-t{TAG}-s{seed}"


def _all_db_dirs():
    """Return all scratch dirs created by this test module."""
    dirs = []
    for seed in SEEDS:
        name = _db_name(seed)
        dirs.append(os.path.join(SCRATCH_RELBENCH, name))
        dirs.append(os.path.join(SCRATCH_PRE, name))
    return dirs


@pytest.fixture(scope="module")
def generated_dbs():
    """Generate all small DBs once, caching to ~/scratch/relbench/ for Rust.
    Cleans up all created directories when the module is done."""
    dbs = {}
    for seed in SEEDS:
        db_name = _db_name(seed)
        cache_dir = os.path.join(SCRATCH_RELBENCH, db_name)
        set_random_seed(0)
        dataset = SyntheticDataset(
            seed=seed,
            config=Config(
                database_params=SMALL_DB_PARAMS,
                cache_dir=cache_dir,
            ),
        )
        # get_db() saves parquet files to {cache_dir}/db/ for Rust to read
        dbs[seed] = dataset.get_db()

    yield dbs

    # cleanup
    for d in _all_db_dirs():
        if os.path.exists(d):
            shutil.rmtree(d)


def test_all_dbs_generated(generated_dbs):
    assert len(generated_dbs) == NUM_DBS
    for seed, db in generated_dbs.items():
        assert db is not None
        assert len(db.table_dict) > 0


def test_parquet_files_cached(generated_dbs):
    """Verify that get_db() wrote parquet files where Rust expects them."""
    for seed in SEEDS:
        db_dir = os.path.join(SCRATCH_RELBENCH, _db_name(seed), "db")
        assert os.path.isdir(db_dir), f"Missing cache dir: {db_dir}"
        parquets = glob.glob(os.path.join(db_dir, "*.parquet"))
        assert len(parquets) > 0, f"No parquet files in {db_dir}"


def test_valid_dbs_exist(generated_dbs):
    valid_count = sum(1 for db in generated_dbs.values() if is_valid_db(db))
    assert valid_count > 0, "No valid databases were generated"


def _cargo_env():
    """Build env dict so the Rust binary can find libpython at link time."""
    env = os.environ.copy()
    libdir = sysconfig.get_config_var("LIBDIR")
    # Embed the Python library path as an rpath in the binary so the
    # dynamic linker can find libpython at runtime.
    env["RUSTFLAGS"] = f"-C link-arg=-Wl,-rpath,{libdir}"
    return env


@pytest.fixture(scope="module")
def preprocessed_dbs(generated_dbs):
    """Run Rust `pre` on each generated DB."""
    env = _cargo_env()
    for seed in SEEDS:
        db_name = _db_name(seed)
        subprocess.run(
            ["pixi", "run", "cargo", "run", "--release", "--", "pre", db_name],
            cwd=RUSTLER_DIR,
            env=env,
            check=True,
        )
    return generated_dbs


def test_pre_output_exists(preprocessed_dbs):
    """Verify that Rust preprocessing created the expected output files."""
    for seed in SEEDS:
        pre_dir = os.path.join(SCRATCH_PRE, _db_name(seed))
        assert os.path.isdir(pre_dir), f"Missing pre dir: {pre_dir}"
        for fname in PRE_OUTPUT_FILES:
            fpath = os.path.join(pre_dir, fname)
            assert os.path.isfile(fpath), f"Missing pre output: {fpath}"
            assert os.path.getsize(fpath) > 0, f"Empty pre output: {fpath}"


def test_tasks_can_be_extracted(generated_dbs):
    """Verify that we can extract classification and regression tasks from valid DBs."""
    total_clf, total_reg = 0, 0
    for seed, db in generated_dbs.items():
        if not is_valid_db(db):
            continue
        db_name = _db_name(seed)
        for table_name in sorted(db.table_dict.keys()):
            info = get_tasks_info(db=db, db_name=db_name, table_name=table_name)
            assert "clf" in info and "reg" in info
            total_clf += len(info["clf"])
            total_reg += len(info["reg"])

    assert total_clf + total_reg > 0, "No tasks were extracted from any database"


def test_train_test_split(generated_dbs):
    """Verify that we can split DBs into train/test sets and collect tasks from each."""
    test_seeds = SEEDS[: 2 * NUM_TEST_DBS]
    train_seeds = SEEDS[2 * NUM_TEST_DBS :]
    num_train_dbs = NUM_DBS - 2 * NUM_TEST_DBS

    assert len(train_seeds) == num_train_dbs
    assert len(test_seeds) == 2 * NUM_TEST_DBS

    def collect_tasks(seed_list, max_count, per_db_task_limit=None):
        clf, reg = [], []
        db_count = 0
        for seed in seed_list:
            db = generated_dbs[seed]
            if not is_valid_db(db):
                continue
            db_name = _db_name(seed)
            db_clf, db_reg = [], []
            for table_name in sorted(db.table_dict.keys()):
                info = get_tasks_info(db=db, db_name=db_name, table_name=table_name)
                db_clf.extend(info["clf"])
                db_reg.extend(info["reg"])
            if per_db_task_limit:
                db_clf = db_clf[: per_db_task_limit // 2]
                db_reg = db_reg[: per_db_task_limit // 2]
            clf.extend(db_clf)
            reg.extend(db_reg)
            db_count += 1
            if db_count == max_count:
                break
        return clf, reg, db_count

    test_clf, test_reg, test_count = collect_tasks(test_seeds, NUM_TEST_DBS, per_db_task_limit=10)
    train_clf, train_reg, train_count = collect_tasks(train_seeds, num_train_dbs)

    # At least some DBs should be valid and produce tasks
    assert test_count + train_count > 0, "No valid DBs for train or test"

    # Tasks should be well-formed tuples of (db_name, table_name, col_name, [])
    for task in test_clf + test_reg + train_clf + train_reg:
        assert len(task) == 4
        db_name, table_name, col_name, extra = task
        assert db_name.startswith("rel-synthetic-test-t")
        assert isinstance(table_name, str)
        assert "feature" in col_name
        assert extra == []
