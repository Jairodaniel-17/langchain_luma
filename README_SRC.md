# Development Instructions

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -e ".[dev,langchain]"
   ```

## Testing

Run tests with pytest:

```bash
pytest
```

## Linting

Run ruff:

```bash
ruff check .
```

## Packaging

Build the package:

```bash
python -m build
```
