# GRAEP

A composable HEP analysis framework.

## Table of contents

- [Input data](#input-data)

---

## Input data

You tell GRAEP which datasets to use, and it finds the files for you.
There are two steps.

1. **Build a fileset.** Write a small `queries.py`, wrap it in a spec
   object, and call `build_fileset(spec)`. You get back a Python dict
   listing every file URI grouped by process. No ROOT files are opened
   yet.
2. **Export to disk.** Call `export_fileset(...)` to open every file
   once, count events (and weighted events), and write
   `nanoaods.json` plus per-process JSONs. **The exported JSONs are
   what the rest of the pipeline reads.**

A working setup lives in
[`examples/example_opendata_cms/queries.py`](./examples/example_opendata_cms/queries.py).
The easiest way to learn the pattern is to copy it and edit.

A minimal usage looks like this:

```python
from examples.example_opendata_cms.config import config
from graep.inputs import build_fileset, export_fileset

fileset = build_fileset(config.inputs)
export_fileset(fileset, "examples/example_opendata_cms/datasets")
```

The `config` object is a `graep.config.main.Config` instance assembled
in
[`examples/example_opendata_cms/config.py`](./examples/example_opendata_cms/config.py).
That file imports each section spec from its own module and stitches
them into one `Config`.

### Writing your own queries

Each query callable returns a mapping from `ProcessInfo` to a dataset
path pattern (or a list of them):

```python
from graep.inputs import ProcessInfo

def signal_queries():
    return {
        ProcessInfo("signal", xsec=2.448):
            "/MyDataset_*/RunIISummer20UL16NanoAODv9-*/NANOAODSIM",
    }

def data_queries():
    return {
        # Multiple patterns merge into one fileset entry.
        ProcessInfo("data", xsec=1.0, is_data=True): [
            "/SingleMuon/Run2016G-UL2016_MiniAODv2_NanoAODv9-v*/NANOAOD",
            "/SingleMuon/Run2016H-UL2016_MiniAODv2_NanoAODv9-v*/NANOAOD",
        ],
    }
```

`ProcessInfo` carries:

- `name`: short identifier used as the fileset key.
- `xsec`: cross-section in pb (or any sentinel for data).
- `is_data`: set `True` for collision data so the export skips weight
  summation.

### Spec classes

Each resolver has a matching spec class. You construct an instance and
hand it to `build_fileset`. Two are available:

- **`graep.config.inputs.open_data.OpenDataPortalFilesetSpec`** wraps
  the CERN Open Data Portal resolver. No proxy needed.
- **`graep.config.inputs.rucio.RucioFilesetSpec`** wraps the rucio
  resolver. Needs an active VOMS proxy and the `examples` dependency
  group (`uv sync --group examples`).

Switching between them is one import change:

```python
from graep.config.inputs.rucio import RucioFilesetSpec

spec = RucioFilesetSpec(
    queries=[...],
    scope="atlas",      # rucio scope; defaults to "cms"
)
```

For rucio, refresh the proxy first:

```bash
voms-proxy-init -voms cms -rfc --valid 168:0
```

#### Common fields (every spec)

| Field | Default | Description |
|---|---|---|
| `queries` | _required_ | One or more zero-argument callables. Each returns a mapping of `ProcessInfo` to a dataset pattern (or list of patterns). Multiple callables are merged. |
| `cache_dir` | `/tmp/graep/.cache` | Where `build_fileset` writes its on-disk cache. |
| `force_refresh` | `False` | If `True`, bypass any cached entry and recompute the fileset. |
| `max_files_per_sample` | `None` | Per-process cap on the number of file URIs returned. `None` means no cap. |
| `variation` | `"nominal"` | Variation tag used to build the fileset key `"<process>__<variation>"`. |

#### `OpenDataPortalFilesetSpec` extras

| Field | Default | Description |
|---|---|---|
| `timeout` | `30.0` | HTTP timeout for portal requests, in seconds. |

#### `RucioFilesetSpec` extras

| Field | Default | Description |
|---|---|---|
| `scope` | `"cms"` | Rucio scope passed to `client.list_files`. |
| `veto_rules` | `[]` | Callables `(str -> bool)`. Resolved dataset names matching any rule are dropped before file listing. |

### Caching

`build_fileset(...)` saves the fileset dict on disk so re-running the
same call is instant. By default the cache lives at
`/tmp/graep/.cache/<hash>.json`; override with `cache_dir` on the spec.

The hash is over:

- the resolver class,
- the merged queries dict (process names, cross-sections, dataset
  patterns), and
- the keyword opts you pass (`variation`, `max_files_per_sample`).

Change any of those and the next call is a cache miss; GRAEP runs the
resolver again and writes a new entry. To force a recompute without
changing the inputs, set `force_refresh=True` on the spec (or pass
`force_refresh=True` to `build_fileset`).

`export_fileset(...)` does **not** cache. It writes its JSONs every
time you call it.

### Writing your own resolver

Two pieces are needed: a resolver class and a matching spec class.

1. Subclass `FilesetResolver` from
   [`src/graep/inputs/queries.py`](./src/graep/inputs/queries.py) and
   implement `resolve`. Look at
   [`src/graep/inputs/open_data.py`](./src/graep/inputs/open_data.py)
   for a worked example. The contract is:
   - Take a mapping of `ProcessInfo` to one or more dataset patterns.
   - Return a fileset dict with `files` and `metadata` per process.
   - Set `nevts` / `nevts_wt` to placeholder zeros; `export_fileset`
     fills in real values when files are opened.
   - Pass your work through `self._cached_resolve(...)` if you want
     the on-disk cache for free.
2. Subclass `FilesetSpec` from
   [`src/graep/config/inputs/base.py`](./src/graep/config/inputs/base.py)
   and implement `make_resolver` to construct your resolver.

Pass the new spec to `build_fileset` like any other. No registration
needed.
