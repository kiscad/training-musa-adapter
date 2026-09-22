# 架构与文档入口

更新日期：2026-09-22。

本项目使用 `training-musa-adaptor` 发行名、`training_musa_adaptor` Python 包名和
`TRAINING_MUSA_ADAPTOR_` 环境变量前缀。当前架构采用 `AttrPatch / HookPatch`、普通工厂和
显式 `PATCHES`；不引入 planner、binding/provider 或 MutationPlan 一类的注册框架。

## 当前维护入口

| 需要了解的内容 | 文档 |
|---|---|
| 目标、功能、安装、配置和排障 | [README](../README.md) / [English](../README.en.md) |
| 导入边界、所有权、撤销与新补丁开发 | [CONTRIBUTING](../CONTRIBUTING.md) |
| Coding agent 开发约束、验证与交付流程 | [AGENTS](../AGENTS.md) |
| 每项补丁的验证状态和已知缺口 | [PATCH_LEDGER](PATCH_LEDGER.md) |

## 架构依据与验证边界

CONTRIBUTING 说明机制和开发契约，AGENTS 规定开发流程，源码决定当前行为，
补丁台账记录实际验收范围；这些内容不一致时，
先记录差异并核实，不把设计目标写成已实现功能。历史验证证据不覆盖新版本测试，
也不自动成为新增功能。

后续机制变更在 CONTRIBUTING 和相关代码中说明必要性、兼容影响及验证，涉及开发流程时
同步更新 AGENTS。不要在这里维护另一套配置 schema、补丁目录清单或重复的逐 ID 状态表。
