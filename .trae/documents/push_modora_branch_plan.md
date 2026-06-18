# 推送 `modora` 新分支计划

## Summary

目标是把 `/home/zhanggaoyuan.225/modora/ruc-ov-eval` 当前本地代码推送到远端仓库 `origin` 的新分支 `modora`。

用户明确要求：

- 新分支名就叫 `modora`。
- 提交信息使用 `modora updates`。
- “全部提交”，但不要 push `config` 下的修改，防止 API key 泄露。

## Current State Analysis

已做只读检查：

- 仓库路径：`/home/zhanggaoyuan.225/modora/ruc-ov-eval`
- 当前本地分支：`gpy/modora`
- 远端：
  - `origin git@github.com:g121370451/ruc-ov-eval.git`
- 远端精确分支 `refs/heads/modora` 当前不存在。
- 当前工作区有大量未提交变更，共约 151 个 `git status --porcelain` 条目。
- 工作区包含配置相关变更，例如：
  - `ov_test/config_modora/versionrag_config.yaml`
  - `ov_test/config/**`
  - `ov_test/config_*/**`
- `versionrag_config.yaml` 当前包含明文 API key，因此配置目录必须从本次提交中排除。

## Proposed Changes

### 1. 创建本地新分支 `modora`

在当前工作区基础上创建新分支：

```bash
git checkout -b modora
```

原因：

- 用户要求推送到新分支 `modora`。
- 远端 `refs/heads/modora` 不存在，可以安全创建新分支。

### 2. 暂存非配置目录的全部变更

使用 Git pathspec exclude 暂存所有变更，但排除配置目录：

```bash
git add -A -- . \
  ':(exclude)ov_test/config/**' \
  ':(exclude)ov_test/config_*/**' \
  ':(exclude)ov_test/config_modora/**'
```

排除范围说明：

- `ov_test/config/**`
- `ov_test/config_*/**`
- `ov_test/config_modora/**`

这样会排除：

- `ov_test/config_modora/versionrag_config.yaml`
- `ov_test/config_hipporag/**`
- `ov_test/config_lightrag/**`
- `ov_test/config_pageindex/**`
- `ov_test/config_per_question*/**`
- `ov_test/config_sql_agent/**`
- 其他匹配 `ov_test/config_*` 的配置目录

原因：

- 用户明确要求不要 push config 下的修改。
- 当前配置文件里存在明文 API key，必须避免进入提交。

### 3. 提交前检查 staged 内容

提交前检查是否有配置文件被误暂存：

```bash
git diff --cached --name-only
git diff --cached --name-only | grep -E '^ov_test/config(_.*)?/' || true
```

验收标准：

- 第一条命令能看到将要提交的文件。
- 第二条命令不应输出任何配置目录文件。

如果第二条命令有输出，则停止，不提交，并重新调整暂存范围。

### 4. 提交

使用用户指定提交信息：

```bash
git commit -m "modora updates"
```

### 5. 推送到远端新分支

```bash
git push -u origin modora
```

预期结果：

- 远端创建 `origin/modora`。
- 本地 `modora` 分支设置 upstream 为 `origin/modora`。

## Assumptions & Decisions

- “不要 push config 下的修改”解释为：不提交 `ov_test/config/**`、`ov_test/config_*/**`、`ov_test/config_modora/**` 下的任何变更。
- `.env` 文件当前没有出现在 `git status --short` 输出里，说明未被 Git 跟踪或被忽略；本计划仍会通过 staged 检查确认不会误提交敏感配置。
- 不会使用 `git reset --hard`、`git checkout -- <path>` 等会丢弃工作区内容的命令。
- 不会清理或恢复配置目录的本地修改，只是不把它们纳入本次 commit/push。
- 如果 `git commit` 提示没有 staged changes，则停止并向用户说明。
- 如果 push 因权限、网络或远端拒绝失败，则停止并报告错误。

## Verification Steps

执行后检查：

```bash
git status --short --branch
git rev-parse --abbrev-ref HEAD
git ls-remote --heads origin refs/heads/modora
```

验收标准：

- 当前分支为 `modora`。
- `origin/modora` 存在。
- 已提交内容不包含 `ov_test/config/**`、`ov_test/config_*/**`、`ov_test/config_modora/**`。
- 本地配置目录修改仍保留在工作区，不被丢弃。
