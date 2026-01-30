<div align="center">
  <h1>PluRel</h1>
</div>
<br>


## Setup

Setup the rt environment with [pixi](https://pixi.sh/latest/installation/)

```bash
$ pixi install
```

Run pytest as a sanity check

```bash
$ pixi run pytest
```

## Getting Started

- The `SyntheticDataset` class can be used to create [relbench](https://github.com/snap-stanford/relbench) compatible dataset objects.
- It only requires a `seed` and a `Config` object that contains `database`, `scm` and `dag` level params for sampling. See example below.

```py
from plurel.dataset import SyntheticDataset
from plurel.config import Config

# create config with default values/choices for params
config = Config()
seed = 0

# create relbench compatible dataset
dataset = SyntheticDataset(seed=seed, config=config)

# create database which can be cached via relbench APIs
db = dataset.make_db()
```

### Config

The `Config` object is a collection of constants as well as `Choices` from which values can be sampled uniformly at random. For instance: the `scm_layout_choices` indicates that the layout can be selected at random from the four available choices. Similarly, `Choices` also supports sampling from a `"range"` of values (see `scm_col_node_perc_choices` in snippet below.)

```py
@dataclass(frozen=True)
class SCMParams:
    ...
    scm_layout_choices: Choices = Choices(
        kind="set",
        value=["ErdosRenyi", "BarabasiAlbert", "RandomTree", "ReverseRandomTree"],
    )
    scm_col_node_perc_choices: Choices = Choices(kind="range", value=[0.6, 0.9])
```
