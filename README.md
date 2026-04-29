# PyADRL

This is a python reinforcement learning project using PettingZoo and Ray Rlib.

## Protos

`Protos` is a Git submodule used for defining shared types across multiple languages.

It is responsible for:
- Enabling type-safe remote procedure calls (RPC)
- Providing a shared contract between components
- Serving as the communication backbone between the PyADRL library and the simulation environment


# How to Run

## Nix

If you have [Nix](https://nixos.org/download/) installed, the development environment is fully reproducible with no manual dependency setup.

### 1. Enter the Development Shell

```bash
nix develop
```

Alternatively automate this by installing direnv and allow it

```bash
direnv allow
```

This will automatically:
- Make `python`, `protobuf`, and `grpc` available
- Install missing python packages from pyproject.toml
- Initialize and update all Git submodules
- Print next steps as a reminder

---

## Manual setup

### 1. Initialize Git Submodules

Before building the project, make sure all Git submodules are initialized and updated:

```bash
git submodule init
git submodule update
```

### 2. Build the protobuf files

The .NET projects must be built before running the simulation:

```bash
sh build.sh
```

### 3. Enter virtual python environment

On Linux/Mac:
```bash
python3 -m venv venv
source ./venv/bin/activate
pip3 install .
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install .
```

---

## Learning a model
Launch the simulation environment by setting up and launching the project by following the instructions at [simulation](https://github.com/DAT-P8/simulation)

Training:
```bash
python3 train.py
```

Testing: 
```bash
python3 eval.py --restore {model_name}
```
