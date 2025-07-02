# Contributing to Hetu-Galvatron

Welcome to the Hetu-Galvatron project! We appreciate your contribution to the development of automatic distributed training systems.

## How to Contribute

### Code Contributions

#### High-Impact Areas
- **New Parallelism Strategies**: Implement novel parallel training methods
- **Hardware Support**: Add support for new GPU/TPU architectures
- **Performance Optimization**: Improve training efficiency and memory usage
- **New Architecture Models**: Such as multi-modal models, extending support beyond language models

#### Beginner-Friendly Tasks
- **Documentation**: Improve code comments and user guides
- **Bug Fixes**: Resolve issues labeled as `good first issue`
- **Testing**: Add unit tests and integration tests
- **Examples**: Create tutorials and example scripts
- **Hardware and Model Profiling**: Add profile data for new hardware and models

### Non-Code Contributions
- Documentation translation
- Tutorial creation
- Issue reporting
- Feature suggestions
- Community support

## Quick Start

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/PKU-DAIR/Hetu-Galvatron.git
cd Hetu-Galvatron

# Create virtual environment
conda create -n galvatron python=3.8
conda activate galvatron

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Development Workflow

```bash
# 1. Fork the repository to your personal account

# 2. Add upstream repository
git remote add upstream https://github.com/PKU-DAIR/Hetu-Galvatron.git

# 3. Create feature branch
git checkout -b feature/your-feature-name

# 4. Develop and commit
git add .
git commit -m "[Runtime] feat: add your feature description"

# 5. Push to your repository
git push origin feature/your-feature-name

# 6. Create Pull Request
```

### Code Standards

#### Commit Message Convention
Similar to [Conventional Commits](https://www.conventionalcommits.org/):
```
[Modified Module]<type>(<scope>): <description>

Modified Module: Runtime, Search Engine, Profiler, Misc
Types: feat, fix, docs, style, refactor, test, chore

Examples:
[Runtime] feat(core): add sequence parallelism support
[Profiler] fix: resolve CUDA memory leak issue
[Misc] docs(api): update model configuration guide
```

#### Testing Requirements
- Write tests for new features
- Maintain test coverage above 80%
- Use pytest as testing framework
- Mock external dependencies

## Newcomer's Guide - Try Hardware and Model Profiling

In the [models](https://github.com/PKU-DAIR/Hetu-Galvatron/tree/main/galvatron/models) folder, we provide some example models and provide the profiling information of the model's computation and memory, as well as the recommended parallel strategies in the configs folder. However, it is unrealistic to measure the corresponding profiling data for all models and hardware devices, so we encourage you to measure different hardware and models and submit PRs. The specific profiling method can be referred to the [Profiling with Galvatron](https://hetu-galvatron.readthedocs.io/en/latest/3_quick_start/quick_start.html#profiling-with-galvatron) section.

### How to Contribute Profiling Data

1. **Choose Hardware Platform**: Select GPU models or other hardware platforms we haven't covered yet
2. **Choose Model**: Select from existing models or add new model architectures
3. **Run Profiling**: Follow the documentation guide for computation and memory profiling
4. **Submit Data**: Submit profiling results as PR to the corresponding configs directory
5. **Verify Results**: Ensure accuracy and reproducibility of profiling data

This is a very beginner-friendly way to contribute, helping you become familiar with Galvatron's working principles while providing valuable data to the community.

## Documentation Contribution

### Documentation Structure
```
docs/
├── en/source/          # English documentation
├── zh_CN/source/       # Chinese documentation
├── imgs/               # Image resources
└── requirements.txt    # Documentation dependencies
```

### Building Documentation Locally

```bash
# English documentation
cd docs/en
make html

# Chinese documentation
cd docs/zh_CN
make html
```

### Documentation Writing Standards

- Use clear title hierarchy
- Include code examples and execution results
- Add necessary diagrams and flowcharts
- Keep Chinese and English versions synchronized

## Reporting Issues

### Before Reporting
1. Check existing [issues](https://github.com/PKU-DAIR/Hetu-Galvatron/issues)
2. Search [discussions](https://github.com/PKU-DAIR/Hetu-Galvatron/discussions)
3. Try the latest version from main branch

### Issue Templates

Mainly includes **Bug Report** and **Feature Request** templates, please refer to the issue submission interface.

## Contact Us

If you have any questions, feel free to contact us through the following channels:

- **Bug Reports**: [GitHub Issues](https://github.com/PKU-DAIR/Hetu-Galvatron/issues)
- **Feature Suggestions**: [GitHub Discussions](https://github.com/PKU-DAIR/Hetu-Galvatron/discussions)
- **Email Contact**: 
  - Xinyi Liu: xy.liu@stu.pku.edu.cn
  - Yujie Wang: alfredwang@pku.edu.cn
  - Shenhan Zhu: shenhan.zhu@pku.edu.cn

---

Thank you for your attention and contribution to Hetu-Galvatron! 