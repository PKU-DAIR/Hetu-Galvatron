.. Galvatron documentation master file, created by
   sphinx-quickstart on Sat Nov  9 18:33:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/PKU-DAIR/Hetu-Galvatron

Galvatron
=========

.. image:: https://img.shields.io/github/license/PKU-DAIR/Hetu-Galvatron
   :target: https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/LICENSE
   :alt: GitHub License

.. image:: https://img.shields.io/github/v/release/PKU-DAIR/Hetu-Galvatron
   :target: https://github.com/PKU-DAIR/Hetu-Galvatron/releases
   :alt: GitHub Release

.. image:: https://img.shields.io/pypi/v/hetu-galvatron
   :target: https://pypi.org/project/hetu-galvatron/
   :alt: PyPI - Version

.. image:: https://img.shields.io/readthedocs/hetu-galvatron
   :target: https://hetu-galvatron.readthedocs.io
   :alt: Read the Docs

.. image:: https://static.pepy.tech/badge/hetu-galvatron
   :target: https://pepy.tech/project/hetu-galvatron
   :alt: Downloads

.. image:: https://visitor-badge.laobi.icu/badge?page_id=PKU-DAIR.Hetu-Galvatron
   :alt: visitors

Galvatron 是一个为 Transformer 模型（包括大语言模型 LLMs）设计的自动分布式训练系统。它利用先进的自动并行技术提供卓越的训练效率。本仓库包含了 Galvatron-2 的官方实现，这是我们最新版本，增加了多项新特性。

**Galvatron GitHub:** https://github.com/PKU-DAIR/Hetu-Galvatron

.. toctree::
   :maxdepth: 2
   :caption: 目录
   
   概述 <1_overview/overview_zh>
   安装 <2_installation/installation_zh>
   快速入门 <3_quick_start/quick_start_zh>
   Galvatron 模型使用 <4_galvatron_model_usage/galvatron_model_usage_zh>
   搜索引擎使用 <5_search_engine_usage/search_engine_usage_zh>
   可视化 <7_visualization/visualization_zh>
   贡献指南与社区 <6_developer_guide/developer_guide_zh>


支持的并行策略
==============

+------------------------+------------------+------------------------+
| 策略                   | 类型             | 支持的变体             |
+========================+==================+========================+
| 数据并行 (DP)          | 基础             | 传统 DP                |
+------------------------+------------------+------------------------+
| 分片数据并行 (SDP)     | 内存高效         | ZeRO-1, ZeRO-2, ZeRO-3 |
+------------------------+------------------+------------------------+
| 流水线 (PP)            | 模型分割         | GPipe, 1F1B-flush      |
+------------------------+------------------+------------------------+
| 张量 (TP)              | 模型分割         | Megatron-LM 后端,      |
|                        |                  | flash-attn 后端        |
+------------------------+------------------+------------------------+
| 序列 (SP)              | 数据分割         | Megatron-SP, Ulysses   |
+------------------------+------------------+------------------------+
| 检查点 (CKPT)          | 内存高效         | 激活检查点             |
+------------------------+------------------+------------------------+

支持的模型
==========

+------------------+------------------+------------------------+
| 模型类型         | 架构             | 后端                   |
+==================+==================+========================+
| 大语言模型       | GPT              | Huggingface, flash-attn|
+------------------+------------------+------------------------+
| 大语言模型       | LLaMA            | Huggingface, flash-attn|
+------------------+------------------+------------------------+
| 大语言模型       | BERT             | Huggingface            |
+------------------+------------------+------------------------+
| 大语言模型       | T5               | Huggingface            |
+------------------+------------------+------------------------+
| 视觉模型         | ViT              | Huggingface            |
+------------------+------------------+------------------------+
| 视觉模型         | Swin             | Huggingface            |
+------------------+------------------+------------------------+


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`