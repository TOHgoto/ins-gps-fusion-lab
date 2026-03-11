# Simulation & Test Results

This directory stores generated outputs from simulations, experiments, and test runs.

## Layout

```
results/
├── README.md           # This file
├── test_reports/       # Test execution results (JSON, XML, text)
└── figures/            # Plots and visualizations from experiments
```

## Regenerating Test Results

```bash
python scripts/run_tests.py
```

Or run pytest manually (results will go to `results/test_reports/` if configured in script):

```bash
pytest tests/ -v --tb=short --junitxml=results/test_reports/junit.xml
```
