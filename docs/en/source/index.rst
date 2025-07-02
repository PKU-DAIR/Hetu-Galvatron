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

Galvatron is an automatic distributed training system designed for Transformer models, including Large Language Models (LLMs). It leverages advanced automatic parallelism techniques to deliver exceptional training efficiency. This repository houses the official implementation of Galvatron-2, our latest version enriched with several new features.

**Galvatron GitHub:** https://github.com/PKU-DAIR/Hetu-Galvatron

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   Overview <1_overview/overview>
   Installation <2_installation/installation>
   Quick Start <3_quick_start/quick_start>
   Galvatron Model Usage <4_galvatron_model_usage/galvatron_model_usage>
   Search Engine Usage <5_search_engine_usage/search_engine_usage>
   Visualization(New Feature!) <7_visualization/visualization>
   Contributing & Community <6_developer_guide/developer_guide>

Supported Parallelism Strategies
================================

+------------------------+------------------+------------------------+
| Strategy               | Type             | Supported Variants     |
+========================+==================+========================+
| Data Parallelism (DP)  | Basic            | Traditional DP         |
+------------------------+------------------+------------------------+
| Sharded DP (SDP)       | Memory-Efficient | ZeRO-1, ZeRO-2, ZeRO-3 |
+------------------------+------------------+------------------------+
| Pipeline (PP)          | Model Split      | GPipe, 1F1B-flush      |
+------------------------+------------------+------------------------+
| Tensor (TP)            | Model Split      | Megatron-LM Style,     |
|                        |                  | flash-attn Style       |
+------------------------+------------------+------------------------+
| Sequence (SP)          | Data Split       | Megatron-SP, Ulysses   |
+------------------------+------------------+------------------------+
| Checkpointing (CKPT)   | Memory-Efficient | Activation Checkpoint  |
+------------------------+------------------+------------------------+

Supported Models
================

+------------------+------------------+------------------------+
| Model Type       | Architecture     | Backend                |
+==================+==================+========================+
| LLMs             | GPT              | Huggingface, flash-attn|
+------------------+------------------+------------------------+
| LLMs             | LLaMA            | Huggingface, flash-attn|
+------------------+------------------+------------------------+
| LLMs             | BERT             | Huggingface            |
+------------------+------------------+------------------------+
| LLMs             | T5               | Huggingface            |
+------------------+------------------+------------------------+
| Vision Models    | ViT              | Huggingface            |
+------------------+------------------+------------------------+
| Vision Models    | Swin             | Huggingface            |
+------------------+------------------+------------------------+


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
