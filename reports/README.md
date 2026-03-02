# Test Reports

This directory stores test execution results in a hierarchical structure.

## Layout

```
reports/
├── README.md           # This file
└── test_results/
    ├── summary.md      # Human-readable summary (by module)
    ├── summary.json    # Machine-readable JSON summary
    ├── junit.xml       # JUnit XML (for CI: Jenkins, GitHub Actions)
    └── pytest_output.txt  # Raw pytest verbose output
```

## Regenerating Reports

```bash
pytest tests/ -v --tb=short --junitxml=reports/test_results/junit.xml 2>&1 | tee reports/test_results/pytest_output.txt
```

Or use the convenience script (from repo root), which runs tests and regenerates all reports:

```bash
python scripts/run_tests.py
```
