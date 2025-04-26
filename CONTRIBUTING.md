# Contributing to Image Stitching for Panorama Generation

Thank you for considering contributing to our project! We welcome contributions from everyone, whether it's adding new features, improving documentation, fixing bugs, or suggesting enhancements.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
   ```bash
   git clone https://github.com/YourUsername/image-stitching-panorama.git
   cd image-stitching-panorama
   ```
3. **Set up the development environment**
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

## How to Contribute

### Reporting Bugs

Before submitting a bug report:

1. Check the [issue tracker](https://github.com/YourUsername/image-stitching-panorama/issues) to see if the bug has already been reported
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the issue
   - Expected behavior and actual behavior
   - Screenshots if applicable
   - Any relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues:

1. Provide a clear, descriptive title
2. Describe the current behavior and explain the behavior you'd like to see
3. Explain why this enhancement would be useful

### Code Contributions

1. **Create a branch** for your feature or bugfix
   ```bash
   git checkout -b feature/your-feature-name
   ```
   
2. **Make your changes** and follow the coding conventions
   - Follow PEP 8 style guidelines for Python code
   - Write meaningful commit messages
   - Add or update tests as needed

3. **Run tests** to ensure your changes don't break existing functionality
   ```bash
   python test_panorama.py
   ```

4. **Push your changes** to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Submit a pull request** to the main repository
   - Provide a clear description of the changes
   - Reference any related issues

## Coding Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Write docstrings for all functions, classes, and methods
- Comment your code where necessary
- Keep functions and methods focused on a single responsibility
- Write tests for new features

## Testing

We use Python's unittest framework. Run tests using:

```bash
python test_panorama.py
```

## Pull Request Process

1. Update the README.md or documentation with details of changes if applicable
2. The PR should work on Python 3.7 and higher
3. Your PR will be reviewed by maintainers who may request changes
4. Once approved, your PR will be merged

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We expect all contributors to adhere to the following guidelines:

- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE). 