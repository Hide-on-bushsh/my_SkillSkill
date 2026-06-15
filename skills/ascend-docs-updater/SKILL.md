---
name: "ascend-docs-updater"
description: "Updates Ascend NPU best practice docs from test cases. Invoke when user wants to update/regenerate docs, sync docs with test cases, or check doc-testcase consistency."
---

# Ascend Docs Updater

This skill updates the Ascend NPU best practice documentation by running the `generate_docs.py` script and presenting a diff summary for user confirmation.

## When to Invoke

- User asks to update or regenerate Ascend best practice docs
- User wants to sync documentation with latest test cases
- User wants to check if docs are consistent with test cases
- User mentions weekly doc update or doc refresh

## Workflow

### Step 1: Pre-check

Before running the script, check the current state:

1. Run `git status` in `c:\Users\xujianzhao\Desktop\sglang` to see if there are uncommitted changes in the docs directory.
2. If there are uncommitted changes, warn the user and ask if they want to proceed (the script will overwrite .mdx files).

### Step 2: Run the generation script

Execute:
```
cd c:\Users\xujianzhao\Desktop\sglang\docs_new\docs\hardware-platforms\ascend-npus\best_practice; python generate_docs.py
```

### Step 3: Show diff summary

After the script completes, run:
```
cd c:\Users\xujianzhao\Desktop\sglang; git diff --stat docs_new/docs/hardware-platforms/ascend-npus/best_practice/
```

Then for each changed file, show a brief summary of what changed:
```
cd c:\Users\xujianzhao\Desktop\sglang; git diff docs_new/docs/hardware-platforms/ascend-npus/best_practice/
```

Present the diff to the user in a structured way:
- **New files**: List any newly generated .mdx files
- **Deleted sections**: List any configurations that were removed (test cases deleted)
- **Modified sections**: Highlight key changes:
  - Parameter changes (env vars, args, benchmark params)
  - New configurations added (new test cases)
  - Heading/anchor changes

### Step 4: Confirm

Ask the user to confirm the changes:
- If user approves: the changes are kept in the working directory, ready for commit
- If user rejects: run `git checkout -- docs_new/docs/hardware-platforms/ascend-npus/best_practice/` to revert

### Step 5: Anchor verification (optional)

If the user wants extra verification, run the anchor check:
```javascript
node _check_anchors.js
```

## Important Notes

- The script at `c:\Users\xujianzhao\Desktop\sglang\docs_new\docs\hardware-platforms\ascend-npus\best_practice\generate_docs.py` is the single source of truth for doc generation
- All .mdx files in the best_practice directory are fully generated - do NOT manually edit them as changes will be overwritten
- If the script output is incorrect, fix the script itself, not the .mdx files
- The script reads test cases from `c:\Users\xujianzhao\Desktop\Ascend\sglang\test\registered\ascend\performance\`
- Test case changes (parameter updates, file renames, additions, deletions) are all handled automatically by the script

## Key File Paths

- Generation script: `c:\Users\xujianzhao\Desktop\sglang\docs_new\docs\hardware-platforms\ascend-npus\best_practice\generate_docs.py`
- Test cases directory: `c:\Users\xujianzhao\Desktop\Ascend\sglang\test\registered\ascend\performance\`
- Output directory: `c:\Users\xujianzhao\Desktop\sglang\docs_new\docs\hardware-platforms\ascend-npus\best_practice\`
- Anchor checker: `c:\Users\xujianzhao\Desktop\sglang\_check_anchors.js`
