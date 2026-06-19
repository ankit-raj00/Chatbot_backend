---
name: code-review
description: >
  Use this skill when the user needs a code review for a file, function, or PR.
  Triggers: "review code", "check this file for bugs", "refactor this",
  "improve this code", "find bugs", "code quality". Do NOT use for writing
  new code from scratch (use the code agent normally).
metadata:
  agent: code
  triggers:
    - "review code"
    - "check for bugs"
    - "refactor this"
    - "improve this code"
    - "find bugs in"
    - "code quality"
    - "code review"
    - "review this function"
---
# Code Review

## Workflow

1. Analyze the provided code snippet or file carefully
2. Run the code with `run_python` tool if it's Python and execution would reveal issues
3. Output your review in strict categories
4. Provide exact code snippets showing how to fix each issue

## Review Categories (always use these headings)

### 🔴 Correctness
Logic errors, wrong outputs, edge cases not handled, off-by-one errors.

### 🔒 Security
SQL injection, XSS, hardcoded secrets, improper input validation, unsafe deserialization.

### ⚡ Performance
O(n²) loops that could be O(n), unnecessary DB calls in loops, missing indexes, memory leaks.

### 🎨 Style & Maintainability
Naming, dead code, overly complex functions, missing docstrings, duplication.

## Output Format

```
## Code Review

### 🔴 Correctness
**Issue**: [description]
**Location**: Line X
**Fix**:
```python
# Fixed code here
```

### 🔒 Security
[issues or "No security issues found"]

### ⚡ Performance
[issues or "No performance issues found"]

### 🎨 Style & Maintainability
[issues or "Code style looks good"]

## Summary
X issues found: Y critical, Z warnings, W suggestions.
```

## Rules

- Always provide the fixed code, not just the description
- If no issues in a category, say "No [category] issues found" — never skip the section
- Prioritize critical issues first
- Be specific: include line numbers and exact variable names
