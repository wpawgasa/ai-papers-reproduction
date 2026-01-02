# Papers Reproduction

A consolidated collection of implementation codebases used to replicate experimental results from AI research papers.

## Projects

| Paper | Directory | Status |
|-------|-----------|--------|
| [RL's Razor: Why Online RL Forgets Less](https://arxiv.org/abs/2509.04259) | [why_RL_forget_less/](why_RL_forget_less/) | ✅ Complete |

## Structure

Each subdirectory contains a self-contained reproduction of a specific paper:

```
papers-reproduction/
├── why_RL_forget_less/     # RL's Razor paper reproduction
│   ├── src/                # Source code
│   ├── notebooks/          # Experiment notebooks
│   └── README.md           # Paper-specific documentation
└── ...
```

## Getting Started

Each project has its own dependencies and setup instructions. Navigate to the specific project directory and follow its README.

### Example

```bash
cd why_RL_forget_less
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Contributing

When adding a new paper reproduction:

1. Create a new directory with a descriptive name
2. Include a `README.md` with paper link and reproduction details
3. Use `pyproject.toml` for dependency management
4. Add a Jupyter notebook demonstrating the key experiments
5. Update this README's project table
