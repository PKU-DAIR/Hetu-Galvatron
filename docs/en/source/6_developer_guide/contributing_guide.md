## Contributing Guide

Welcome to the Hetu-Galvatron community! We're excited to have you contribute to advancing automatic distributed training for large-scale AI models.

> **Full Contributing Guide**: For the complete contributing guide with detailed setup instructions, coding standards, and community information, please see our [CONTRIBUTING.md](https://github.com/PKU-DAIR/Hetu-Galvatron/CONTRIBUTING.md) file.

### How to Contribute

#### Code Contributions

We welcome all types of code contributions:

##### High-Impact Areas
- **New Parallelism Strategies**: Implement novel parallel training methods
- **Hardware Support**: Add support for new GPU/TPU architectures
- **Performance Optimization**: Improve training efficiency and memory usage
- **New Architecture Models**: Such as multi-modal models, extending support beyond language models

##### Beginner-Friendly Tasks
- **Documentation**: Improve code comments and user guides
- **Bug Fixes**: Resolve issues labeled as `good first issue`
- **Testing**: Add unit tests and integration tests
- **Examples**: Create tutorials and example scripts
- **Hardware and Model Profiling**: Add profile data for new hardware and models

#### Non-Code Contributions

Your expertise is valuable beyond coding:

- **Documentation Translation**: Help make Galvatron accessible globally
- **Community Support**: Answer questions in issues and discussions
- **Tutorial Creation**: Write blog posts, videos, or workshops
- **Testing & Feedback**: Try new features and report your experience
- **Evangelism**: Present Galvatron at conferences or meetups

### Quick Start Guide

#### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/Hetu-Galvatron.git
cd Hetu-Galvatron

# Set up development environment
conda create -n galvatron-dev python=3.8
conda activate galvatron-dev

# Install in development mode
pip install -r requirements.txt
pip install -e .
```

#### Making Your First Contribution

```bash
# Create a new branch for your feature
git checkout -b feature/your-awesome-feature

# Make your changes
# ... edit files ...

# Test your changes
python -m pytest tests/

# Commit with clear message
git add .
git commit -m "[Runtime] feat: add awesome new feature"

# Push and create PR
git push origin feature/your-awesome-feature
```

#### Code Standards

##### Commit Messages
Similar to [Conventional Commits](https://www.conventionalcommits.org/):
```
[Modified Module]<type>(<scope>): <description>

Modified Module: Runtime, Search Engine, Profiler, Misc
Types: feat, fix, docs, style, refactor, test, chore
Example: feat(profiler): add GPU memory profiling support
```

##### Testing
- Write tests for new features
- Maintain test coverage above 80%
- Use pytest for testing framework
- Mock external dependencies

#### Newcomer's Guide - Try Hardware and Model Profiling

In the [models](https://github.com/PKU-DAIR/Hetu-Galvatron/tree/main/galvatron/models) folder, we provide some example models and provide the profiling information of the model's computation and memory, as well as the recommended parallel strategies in the configs folder. However, it is unrealistic to measure the corresponding profiling data for all models and hardware devices, so we encourage you to measure different hardware and models and submit PRs. The specific profiling method can be referred to the [Profiling with Galvatron](../3_quick_start/quick_start.html#profiling-with-galvatron) section.

### Documentation Guidelines

#### Documentation Types
- **API Documentation**: Docstrings for all public functions
- **User Guides**: Step-by-step tutorials
- **Developer Guides**: Technical implementation details
- **Examples**: Complete working code samples

#### Building Documentation Locally
```bash
# English documentation
cd docs/en
make html
open _build/html/index.html

# Chinese documentation
cd docs/zh_CN
make html
open _build/html/index.html
```

#### Writing Style
- Use clear, concise language
- Include code examples with expected output
- Add diagrams for complex concepts
- Keep Chinese and English versions synchronized

### Reporting Issues

#### Before Reporting
1. Check existing [issues](https://github.com/PKU-DAIR/Hetu-Galvatron/issues)
2. Search [discussions](https://github.com/PKU-DAIR/Hetu-Galvatron/discussions)
3. Try the latest version from main branch

#### Issue Templates

Mainly includes **Bug Report** and **Feature Request** templates, please refer to the issue submission interface.
