---
name: "ascend-docs-updater"
description: "Updates Ascend NPU best practice docs from test cases. Invoke when user wants to update/regenerate docs, sync docs with test cases, or check doc-testcase consistency."
---

# Ascend 文档更新器

本技能通过运行 `generate_docs.py` 脚本更新 Ascend NPU 最佳实践文档，跟踪用例 commit ID，并展示 diff 摘要供用户确认。

## 何时调用

- 用户要求更新或重新生成 Ascend 最佳实践文档
- 用户想要同步文档与最新用例
- 用户想要检查文档与用例是否一致
- 用户提到每周文档更新或文档刷新

## 关键文件路径

- 生成脚本: `c:\Users\xujianzhao\Desktop\sglang\docs_new\docs\hardware-platforms\ascend-npus\best_practice\generate_docs.py`
- 用例目录: `c:\Users\xujianzhao\Desktop\Ascend\sglang\test\registered\ascend\performance\`
- 输出目录: `c:\Users\xujianzhao\Desktop\sglang\docs_new\docs\hardware-platforms\ascend-npus\best_practice\`
- 版本追踪: `c:\Users\xujianzhao\Desktop\sglang\docs_new\docs\hardware-platforms\ascend-npus\best_practice\doc_version.json`
- 锚点检查: `c:\Users\xujianzhao\Desktop\sglang\_check_anchors.js`

## 工作流程

### 第 1 步：记录当前用例 commit

读取 `doc_version.json` 获取 `last_testcase_commit`，该 commit ID 代表当前文档对应的用例版本。

同时获取用例仓库当前 HEAD 确认是否一致：
```
cd c:\Users\xujianzhao\Desktop\Ascend\sglang; git log -1 --format="%H %s"
```

如果当前 HEAD 与 `last_testcase_commit` 不一致，警告用户用例仓库可能被手动更新但未同步文档，询问是否继续（以 `last_testcase_commit` 为基线）。

### 第 2 步：拉取用例仓库

```
cd c:\Users\xujianzhao\Desktop\Ascend\sglang; git pull
```

记录拉取后的新 HEAD commit ID：
```
cd c:\Users\xujianzhao\Desktop\Ascend\sglang; git log -1 --format="%H %s"
```

### 第 3 步：拉取文档仓库

```
cd c:\Users\xujianzhao\Desktop\sglang; git pull my docs
```

### 第 4 步：对比用例变更

运行：
```
cd c:\Users\xujianzhao\Desktop\Ascend\sglang; git diff <last_testcase_commit>..HEAD --stat -- test/registered/ascend/performance/
```

然后获取详细 diff：
```
cd c:\Users\xujianzhao\Desktop\Ascend\sglang; git diff <last_testcase_commit>..HEAD -- test/registered/ascend/performance/
```

如果没有检测到变更，报告"自上次更新以来用例无变更，文档已是最新"并停止。

否则，以结构化形式总结用例变更：
- **新增文件**：新增的用例（新模型配置）
- **删除文件**：删除的用例
- **重命名文件**：文件重命名
- **修改文件**：参数变更，高亮关键差异（环境变量、命令行参数、benchmark 参数、类变更）

### 第 5 步：检查文档仓库工作区

运行：
```
cd c:\Users\xujianzhao\Desktop\sglang; git status docs_new/docs/hardware-platforms/ascend-npus/best_practice/
```

如果有未提交的 .mdx 文件变更，警告用户并询问是否继续（脚本会覆盖 .mdx 文件）。

### 第 6 步：运行生成脚本

执行：
```
cd c:\Users\xujianzhao\Desktop\sglang\docs_new\docs\hardware-platforms\ascend-npus\best_practice; python generate_docs.py
```

### 第 7 步：展示文档 diff 并交叉验证

脚本完成后，运行：
```
cd c:\Users\xujianzhao\Desktop\sglang; git diff --stat docs_new/docs/hardware-platforms/ascend-npus/best_practice/
```

然后：
```
cd c:\Users\xujianzhao\Desktop\sglang; git diff docs_new/docs/hardware-platforms/ascend-npus/best_practice/
```

**交叉验证**：对比用例变更（第 4 步）与文档变更（第 7 步），确认它们匹配：

1. **新增用例 → 新增文档段落**：每个新增的用例文件应在对应 .mdx 中产生新段落
2. **删除用例 → 删除文档段落**：每个删除的用例文件应从 .mdx 中删除对应段落
3. **修改用例 → 更新文档段落**：每个修改的用例文件应更新对应段落
4. **无意外文档变更**：文档变更应仅来自用例变更。如果文档有变更但用例无变更，标记为可疑。

**参数级验证**：对于新增和修改的用例，进一步验证文档中的参数与用例参数一致：

1. 使用 `generate_docs.py` 中的 `extract_config_from_file` 提取用例的环境变量和命令行参数
2. 在对应文档段落中搜索每个环境变量（`export KEY=`）和命令行参数（`--flag`），确认都已出现
3. 对于关键参数值（如 `--tp-size`、`--dp-size`、`--mem-fraction-static`、`--moe-a2a-backend`、`--deepep-mode`、`--speculative-*` 等），对比用例值与文档值是否一致
4. PD 分离模式分别验证 prefill 和 decode 的参数

以汇总表格形式展示交叉验证结果：

| 用例变更 | 文档变更 | 段落匹配 | 参数一致 | 状态 |
|---|---|---|---|---|
| 新增: test_npu_xxx.py | xxx.mdx 新增段落 | ✅ | ✅ | ✅ 匹配 |
| 修改: test_npu_yyy.py (环境变量变更) | yyy.mdx 更新环境变量 | ✅ | ⚠️ --tp-size 用例=8 文档=4 | ⚠️ 参数不一致 |
| 删除: test_npu_zzz.py | zzz.mdx 删除段落 | ✅ | - | ✅ 匹配 |
| (无) | www.mdx 意外变更 | - | - | ⚠️ 需排查 |

### 第 8 步：确认

询问用户确认变更：
- 用户批准：用新 commit ID 和当前日期更新 `doc_version.json`，变更保留在工作区
- 用户拒绝：运行 `git checkout -- docs_new/docs/hardware-platforms/ascend-npus/best_practice/` 回滚

### 第 9 步：更新版本追踪

用户确认后，更新 `doc_version.json`：
```json
{
  "last_testcase_commit": "<拉取后的新commit_id>",
  "last_update_date": "<今天日期>",
  "last_update_summary": "<变更摘要>"
}
```

## 注意事项

- 脚本是文档生成的唯一真相源
- best_practice 目录下所有 .mdx 文件都是全量生成的——不要手动编辑，否则会被覆盖
- 如果脚本输出不正确，修复脚本本身，而不是 .mdx 文件
- 每次成功更新文档后必须更新 `doc_version.json` 以维护追溯链
- 运行脚本前务必拉取两个仓库（用例 + 文档）以确保使用最新代码
