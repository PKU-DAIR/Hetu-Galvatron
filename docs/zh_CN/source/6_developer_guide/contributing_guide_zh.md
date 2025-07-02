## 贡献指南

欢迎加入 Hetu-Galvatron 社区！我们很兴奋能够与您一起推进大规模AI模型的自动分布式训练技术。

> **完整贡献指南**: 查看我们的 [CONTRIBUTING.md](https://github.com/PKU-DAIR/Hetu-Galvatron/CONTRIBUTING.md) 文件，了解详细的环境设置说明、编码标准和社区信息。

### 如何贡献

#### 代码贡献

我们欢迎各种类型的代码贡献：

##### 高影响力领域
- **新的并行策略**: 实现新颖的并行训练方法
- **硬件支持**: 为新的GPU/TPU架构添加支持
- **性能优化**: 提升训练效率和内存使用
- **新结构模型**: 如多模态模型等，扩展超越语言模型的支持

##### 新手友好任务
- **文档**: 改进代码注释和用户指南
- **Bug修复**: 解决标记为 `good first issue` 的问题
- **测试**: 添加单元测试和集成测试
- **示例**: 创建教程和示例脚本
- **硬件和模型测量**: 为新的硬件和模型添加测量数据

#### 非代码贡献

您的专业知识在编码之外同样宝贵：

- **文档翻译**: 帮助让Galvatron在全球范围内更易使用
- **社区支持**: 在问题和讨论中回答问题
- **教程创作**: 编写博客文章、视频或研讨会
- **测试反馈**: 试用新功能并报告您的体验
- **技术推广**: 在会议或聚会上展示Galvatron

### 快速开始指南

#### 开发环境设置

```bash
# Fork并克隆仓库
git clone https://github.com/your-username/Hetu-Galvatron.git
cd Hetu-Galvatron

# 设置开发环境
conda create -n galvatron-dev python=3.8
conda activate galvatron-dev

# 以开发模式安装
pip install -r requirements.txt
pip install -e .
```

#### 进行您的第一次贡献

```bash
# 为您的功能创建新分支
git checkout -b feature/your-awesome-feature

# 进行更改
# ... 编辑文件 ...

# 测试您的更改
python -m pytest tests/

# 提交并附上清晰的消息
git add .
git commit -m "[Runtime] feat: add awesome new feature"

# 推送并创建PR
git push origin feature/your-awesome-feature
```

#### 代码标准

##### 提交消息
类似于 [约定式提交](https://www.conventionalcommits.org/)：
```
[修改模块]<类型>(<范围>): <描述>

修改模块：Runtime, Search Engine, Profiler, Misc
类型: feat, fix, docs, style, refactor, test, chore
示例: feat(profiler): add GPU memory profiling support
```

##### 测试
- 为新功能编写测试
- 保持测试覆盖率在80%以上
- 使用pytest作为测试框架
- 模拟外部依赖

#### 新手上路——尝试进行硬件和模型测量

在[models](https://github.com/PKU-DAIR/Hetu-Galvatron/tree/main/galvatron/models)文件夹中，我们提供了一些示例模型，并在模型的configs文件夹中提供了模型的计算和内存测量信息，以及推荐的并行策略。但是，对于所有模型和硬件设备都测量出对应的测量数据是不现实的，因此我们鼓励您进行不同的硬件和模型测量，并提交PR。具体的测量方法可以参考[使用 Galvatron 进行性能分析](../3_quick_start/quick_start_zh.html#galvatron)章节。

### 文档指南

#### 文档类型
- **API文档**: 所有公共函数的文档字符串
- **用户指南**: 逐步教程
- **开发者指南**: 技术实现细节
- **示例**: 完整的工作代码样本

#### 本地构建文档
```bash
# 英文文档
cd docs/en
make html
open _build/html/index.html

# 中文文档
cd docs/zh_CN
make html
open _build/html/index.html
```

#### 写作风格
- 使用清晰、简洁的语言
- 包含代码示例和预期输出
- 为复杂概念添加图表
- 保持中英文版本同步

### 问题报告

#### 报告之前
1. 检查现有 [issues](https://github.com/PKU-DAIR/Hetu-Galvatron/issues)
2. 搜索 [discussions](https://github.com/PKU-DAIR/Hetu-Galvatron/discussions)
3. 尝试main分支的最新版本

#### 问题模板

主要包含**Bug报告**和**特性请求**两个问题模板，可以参考issue提交界面。