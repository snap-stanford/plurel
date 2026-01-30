<div align="center">
  <h1>PluRel</h1>
  <p>
Synthetic Data unlocks Scaling Laws for Relational Foundation Models
</p>
</div>
<br>


## 1. Setup

Setup the environment with [pixi](https://pixi.sh/latest/installation/)

```bash
$ pixi install
$ pixi run pip install -e .
```

Run pytest as a sanity check

```bash
$ pixi run pytest
```


### 2. Synthesize relational data from scratch

- The `SyntheticDataset` class can be used to create [relbench](https://github.com/snap-stanford/relbench) compatible dataset objects.
- It only requires a `seed` and a `Config` object that contains `database`, `scm` and `dag` level params for sampling. See example below.

```py
from plurel.dataset import SyntheticDataset
from plurel.config import Config

# create relbench compatible dataset
dataset = SyntheticDataset(seed=0, config=Config())

# create database which can be cached via relbench APIs
db = dataset.make_db()
```

The `Config` object is a collection of constants as well as `Choices` from which values can be sampled uniformly at random. For instance: the `scm_layout_choices` indicates that the layout can be selected at random from the four available choices. Similarly, `Choices` also supports sampling from a `"range"` of values (see `scm_col_node_perc_choices` in snippet below.)

```py
@dataclass(frozen=True)
class SCMParams:
    ...
    scm_layout_choices: Choices = Choices(
        kind="set",
        value=["ErdosRenyi", "BarabasiAlbert", "RandomTree", "ReverseRandomTree"],
    )
    scm_col_node_perc_choices: Choices = Choices(kind="range", value=[0.3, 0.9])
```


### 3. Synthesize from SQL schema

Prepare a `schema.sql` file:

```sql
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    status TEXT CHECK (status IN ('active', 'inactive', 'banned'))
);

CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    user_id BIGINT,
    amount DOUBLE,
    order_type TEXT CHECK (order_type IN ('online', 'instore')),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

Synthesize features with `plurel`

```py
from plurel.dataset import SyntheticDataset
from plurel.config import Config, DatabaseParams, Choices

config = Config(
    database_params=DatabaseParams(
        num_rows_entity_table_choices=Choices("range", [500, 1000]),
        num_rows_activity_table_choices=Choices("range", [5000, 10000]),
    ),
    schema_file="schema.sql", # pass the schema file to config
)

db = SyntheticDataset(seed=0, config=config).make_db()
```

PluRel populates the tables with data from diverse distributions

```py
# === synthesized orders table ===
   order_id  user_id    amount order_type                       date
0         0       27  0.450017     online 2014-09-08 00:00:00.000000
1         1      132  0.450033    instore 2014-09-08 02:58:31.964238
2         2       56  0.450052    instore 2014-09-08 05:57:03.928477
3         3      113  0.450071     online 2014-09-08 08:55:35.892716
4         4      127  0.450092    instore 2014-09-08 11:54:07.856955
```

Checkout notebooks in `examples/` for exploration!