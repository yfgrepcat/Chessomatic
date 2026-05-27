# Chessomatic

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p project/bin
tmpdir=$(mktemp -d)
wget -O "$tmpdir/stockfish.tar" https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64.tar
tar -xf "$tmpdir/stockfish.tar" -C "$tmpdir"
mv "$tmpdir/stockfish/stockfish-ubuntu-x86-64" project/bin/stockfish
chmod +x project/bin/stockfish
rm -rf "$tmpdir"
```

## Run the app 

```bash
python project/gui/app.py
```

## Run training 

### Basic LinUCB : Stockfish niveau 3, 100 parties

```bash
python project/experiments/training.py --bandit-type basic_linucb  --total-games 100 --worker-id worker_p100_sf3_basic
```

### Neural LinUCB : Stockfish niveau 3, 100 parties

```bash
python project/experiments/training.py --bandit-type neural_linucb  --total-games 100 --worker-id worker_p100_sf3_neural
```

### Basic LinUCB : Stockfish niveau 3, 450 parties

```bash
python project/experiments/training.py --bandit-type basic_linucb  --total-games 450 --worker-id worker_p450_sf3_basic
```

### Neural LinUCB : Stockfish niveau 3, 450 parties

```bash
python project/experiments/training.py --bandit-type neural_linucb  --total-games 450 --worker-id worker_p450_sf3_neural
```

### Basic LinUCB : Stockfish niveau 10, 450 parties

```bash
python project/experiments/training.py --bandit-type basic_linucb  --total-games 450 --worker-id worker_p450_sf10_basic --stockfish-level 10
```

### Neural LinUCB : Stockfish niveau 10, 450 parties

```bash
python project/experiments/training.py --bandit-type neural_linucb  --total-games 450 --worker-id worker_p450_sf10_neural --stockfish-level 10
```

## Run Benchmark 

### Basic LinUCB

```bash
python project/experiments/benchmark.py --bandit-type basic_linucb --model-path project/models/worker_runtest_yo.npz 
```

### Neural LinUCB 

```bash
python project/experiments/benchmark.py --bandit-type neural_linucb --model-path project/models/worker_runtest_yo_neural.npz
```

## Run Analysis

### Basic LinUCB

```bash
./.venv/bin/python project/experiments/analysis.py   --log-file project/logs/games_worker_runtest_yo.jsonl 
```

### Neural LinUCB 

```bash
./.venv/bin/python project/experiments/analysis.py   --log-file project/logs/games_worker_runtest_yo_neural.jsonl 
```
