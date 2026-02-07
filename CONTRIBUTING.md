# Contributing to Hybrid Quantum-AI Protein Folding

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions. We're all here to advance science and learn together.

## How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Use the issue template** and provide:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs. actual behavior
   - Environment details (OS, Python version, package versions)
   - Error messages and stack traces

### Suggesting Enhancements

1. **Open an issue** with the "enhancement" label
2. **Describe the motivation**: What problem does it solve?
3. **Propose implementation**: If you have ideas on how to implement it
4. **Consider alternatives**: Are there other ways to achieve the same goal?

### Pull Requests

#### Before You Start

1. **Fork the repository** and create a new branch from `main`
2. **Check existing PRs** to avoid duplicate work
3. **Open an issue first** for major changes to discuss the approach

#### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/hybrid-quantum-protein-folding.git
cd hybrid-quantum-protein-folding

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with test dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

#### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write clean, documented code**:
   - Follow PEP 8 style guidelines
   - Add docstrings to all functions and classes
   - Use type hints where appropriate
   - Keep functions focused and modular

3. **Add tests** for new functionality:
   ```python
   # tests/test_your_feature.py
   import pytest
   from hqpf.your_module import your_function

   def test_your_function():
       result = your_function(test_input)
       assert result == expected_output
   ```

4. **Run tests locally**:
   ```bash
   pytest tests/
   pytest tests/test_your_feature.py -v  # Specific test
   ```

5. **Update documentation** if needed:
   - Add docstrings
   - Update README.md if user-facing
   - Add examples to docs/ if introducing new features

6. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: brief description
   
   Longer explanation of what changed and why.
   Fixes #issue_number"
   ```

#### Submitting a PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Link to related issues
   - Screenshots/outputs if applicable

3. **Respond to review feedback**:
   - Be open to suggestions
   - Make requested changes
   - Push additional commits to the same branch

4. **Wait for CI checks** to pass:
   - All tests must pass
   - Code style checks must pass
   - Documentation must build successfully

## Code Style

### Python

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting (line length 100)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [flake8](https://flake8.pycqa.org/) for linting

```bash
# Format code
black hqpf/ tests/ examples/
isort hqpf/ tests/ examples/

# Check style
flake8 hqpf/ tests/ examples/
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_energy(structure: torch.Tensor, hamiltonian_params: Dict) -> float:
    """Calculate the energy of a protein structure.
    
    Args:
        structure: Coarse-grained protein structure of shape (N, 3) where N
            is the number of residues.
        hamiltonian_params: Dictionary of learned Hamiltonian parameters
            including 'J_local', 'K_contact', etc.
    
    Returns:
        Total energy as a scalar float.
    
    Raises:
        ValueError: If structure has invalid shape or NaN values.
    
    Example:
        >>> structure = torch.randn(20, 3)
        >>> params = {'J_local': torch.tensor([1.0, 2.0, ...])}
        >>> energy = calculate_energy(structure, params)
        >>> print(f"Energy: {energy:.4f}")
    """
    # Implementation
    pass
```

## Testing Guidelines

### Test Structure

```
tests/
├── test_models/
│   ├── test_hybrid_model.py
│   ├── test_quantum_hamiltonian.py
│   └── test_vqe.py
├── test_data/
│   └── test_coarse_grained.py
├── test_training/
│   └── test_trainer.py
└── fixtures/
    └── test_fixtures.py
```

### Writing Tests

1. **Unit tests**: Test individual functions in isolation
2. **Integration tests**: Test module interactions
3. **Use fixtures** for common test data:

```python
import pytest
import torch

@pytest.fixture
def sample_sequence():
    return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)

@pytest.fixture
def sample_structure():
    return torch.randn(10, 3)

def test_energy_calculation(sample_structure):
    energy = calculate_energy(sample_structure)
    assert isinstance(energy, float)
    assert not torch.isnan(torch.tensor(energy))
```

4. **Parametrize tests** for multiple scenarios:

```python
@pytest.mark.parametrize("n_residues,n_qubits", [(10, 10), (15, 15), (20, 20)])
def test_model_initialization(n_residues, n_qubits):
    model = HybridModel(n_residues=n_residues, n_qubits=n_qubits)
    assert model.n_residues == n_residues
    assert model.n_qubits == n_qubits
```

5. **Mock expensive operations**:

```python
from unittest.mock import patch, MagicMock

@patch('hqpf.quantum.backends.execute_circuit')
def test_vqe_with_mock(mock_execute):
    mock_execute.return_value = {'energy': -1.5}
    result = run_vqe(...)
    assert result['energy'] == -1.5
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=hqpf --cov-report=html

# Specific file
pytest tests/test_models/test_hybrid_model.py

# Specific test
pytest tests/test_models/test_hybrid_model.py::test_forward_pass

# Skip slow tests (e.g., quantum hardware)
pytest -m "not slow"

# Verbose output
pytest -v
```

## Documentation

### Types of Documentation

1. **Code comments**: Explain *why*, not *what*
2. **Docstrings**: API documentation for functions/classes
3. **README.md**: Project overview and quick start
4. **docs/**: Detailed guides and tutorials
5. **Examples**: Working code demonstrating usage

### Building Documentation

If using Sphinx (future enhancement):

```bash
cd docs/
make html
open _build/html/index.html
```

## Project-Specific Guidelines

### Quantum Computing

- **Mock quantum backends** in tests unless testing quantum-specific behavior
- **Document circuit depth and qubit requirements** for new quantum components
- **Use simulators** for CI/CD; reserve hardware for validation

### Protein Structure

- **Use standard formats**: PDB for structures, FASTA for sequences
- **Validate inputs**: Check sequence validity, structure dimensions
- **Document coordinate systems**: Specify lattice vs. Cartesian coordinates

### Performance

- **Profile before optimizing**: Use cProfile or line_profiler
- **Document computational complexity**: Add time/space complexity notes
- **Consider GPU acceleration**: Make models GPU-compatible where beneficial

## Review Process

### What Reviewers Look For

1. **Correctness**: Does the code do what it claims?
2. **Tests**: Are there adequate tests? Do they pass?
3. **Documentation**: Is the code well-documented?
4. **Style**: Does it follow project conventions?
5. **Performance**: Any obvious inefficiencies?
6. **Breaking changes**: Does it break existing functionality?

### Being a Good Reviewer

- Be constructive and respectful
- Ask questions to understand reasoning
- Suggest alternatives, don't demand changes
- Approve when it's "good enough," not "perfect"
- Respond in a timely manner

### Being a Good Contributor

- Respond to feedback promptly
- Don't take criticism personally
- Ask for clarification if needed
- Keep PRs focused and reasonably sized
- Update your branch if main advances

## Types of Contributions We Need

### High Priority

- [ ] Benchmarking on real proteins with experimental data
- [ ] Error mitigation techniques for quantum circuits
- [ ] Integration with additional quantum backends (IonQ, Rigetti)
- [ ] Fragment assembly algorithms for large proteins
- [ ] Visualization tools for energy landscapes

### Medium Priority

- [ ] Additional coarse-graining schemes
- [ ] Pre-trained models for common protein families
- [ ] Parallel training on multiple GPUs
- [ ] More comprehensive test coverage
- [ ] Tutorial notebooks

### Always Welcome

- Bug fixes
- Documentation improvements
- Performance optimizations
- Example scripts
- Test coverage increases

## Release Process

1. **Version numbering**: Follow [Semantic Versioning](https://semver.org/)
   - MAJOR.MINOR.PATCH (e.g., 1.2.3)
   - MAJOR: Breaking changes
   - MINOR: New features (backward compatible)
   - PATCH: Bug fixes

2. **Release checklist**:
   - [ ] All tests pass
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] Version bumped in setup.py
   - [ ] Git tag created
   - [ ] GitHub release created

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **Security issues**: Email maintainer directly (see README)
- **Contributing questions**: Ask in your PR or open a Discussion

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in academic papers if contributions are substantial

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to hybrid quantum-classical protein folding research! Every contribution, no matter how small, helps advance the field. 🚀🧬⚛️
