"""Script to create the 4 custom skills not in the zip."""
from pathlib import Path

skills_base = Path('d:/Gemini Playgroun/vscodeground/chatbot/backend/skills/builtin')

custom_skills = {
    'analyze-data': {
        'frontmatter': '''---
name: analyze-data
description: >
  Use this skill when the user needs to analyze a CSV or Excel dataset.
  Triggers: "analyze data", "process csv", "examine excel file", "data analysis",
  "statistics on data", "what does this data show". Do NOT use for creating new
  spreadsheets (use spreadsheet-analyst).
metadata:
  agent: data
  triggers:
    - "analyze data"
    - "process csv"
    - "examine excel"
    - "data analysis"
    - "statistics on"
    - "what does this data"
    - "summarize data"
    - "insights from data"
---
''',
        'body': '''# Analyzing Data

## Workflow

1. Identify the file path from the user request
2. Call `analyze_data_file` tool with the file path and the user\'s specific question
3. Present findings clearly with statistics, key insights, and notable patterns

## What to Always Include

- **Baseline statistics**: row count, column names, data types
- **Key metrics**: mean, median, min, max for numeric columns
- **Data quality**: missing values, duplicates
- **Outliers**: values more than 2 standard deviations from mean
- **Answer the specific question**: directly address what the user asked

## Output Format

Present results as a structured summary:

```
Dataset Overview:
- Rows: X | Columns: Y
- Date range: ...

Key Statistics:
| Column | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|

Key Findings:
1. [Most important insight]
2. [Second insight]
3. [Third insight]

Answer to your question: [direct answer]
```

## Common Pitfalls

- Never just dump raw numbers without interpretation
- Always explain what the statistics mean in context
- If data has errors, report them clearly before analysis
'''
    },

    'code-review': {
        'frontmatter': '''---
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
''',
        'body': '''# Code Review

## Workflow

1. Analyze the provided code snippet or file carefully
2. Run the code with `run_python` tool if it\'s Python and execution would reveal issues
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
'''
    },

    'web-research': {
        'frontmatter': '''---
name: web-research
description: >
  Use this skill when the user requests deep research on a web topic.
  Triggers: "research topic", "search web for", "find articles about",
  "look up information on", "what is the latest on", "find recent news about".
  Do NOT use for answering from training data alone.
metadata:
  agent: chat
  triggers:
    - "research topic"
    - "search web"
    - "find articles"
    - "look up information"
    - "what is the latest"
    - "find recent news"
    - "web research"
    - "search the internet"
---
''',
        'body': '''# Web Research

## Workflow

1. Perform a web search using `tavily_search` to find initial sources
2. Identify the top 3-5 most relevant URLs from results
3. Synthesize the raw data into a cohesive, structured summary

## Output Format

Always structure research output as:

```
## Research: [Topic]

### Summary
[2-3 sentence overview of key findings]

### Key Findings
1. [Finding with source]
2. [Finding with source]
3. [Finding with source]

### Sources
- [Title](URL) — [one-line description]
- [Title](URL) — [one-line description]

### Last Updated
[Date of most recent source found]
```

## Rules

- **Never just list links** — synthesize and explain the information
- **Always cite sources** — include URL after each claim
- **Prioritize recency** — prefer sources from the last 6 months for fast-moving topics
- **Note conflicts** — if sources disagree, say so explicitly
- **Assess credibility** — prefer official docs, peer-reviewed papers, established news sources
'''
    },

    'generate-diagram': {
        'frontmatter': '''---
name: generate-diagram
description: >
  Use this skill when the user requests an architecture diagram, flowchart,
  sequence diagram, or any visual diagram. Triggers: "draw diagram",
  "create flowchart", "make mermaid", "architecture diagram", "sequence diagram",
  "class diagram", "ER diagram", "flow chart".
metadata:
  agent: document
  triggers:
    - "draw diagram"
    - "create flowchart"
    - "make mermaid"
    - "architecture diagram"
    - "sequence diagram"
    - "class diagram"
    - "er diagram"
    - "flow chart"
    - "create diagram"
    - "visualize flow"
---
''',
        'body': '''# Generating Mermaid Diagrams

## Workflow

1. Understand what the user wants to visualize
2. Choose the right diagram type (see below)
3. Output valid Mermaid.js syntax in a ```mermaid block
4. Keep diagrams focused — maximum 15-20 nodes

## Diagram Types

### Flowchart (most common)
```mermaid
flowchart TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E
```

### Sequence Diagram
```mermaid
sequenceDiagram
    participant Client
    participant API
    participant DB
    Client->>API: POST /chat
    API->>DB: Save message
    DB-->>API: OK
    API-->>Client: Stream response
```

### Class Diagram
```mermaid
classDiagram
    class Animal {
        +String name
        +makeSound()
    }
    class Dog {
        +fetch()
    }
    Animal <|-- Dog
```

### Entity Relationship
```mermaid
erDiagram
    USER ||--o{ CONVERSATION : has
    CONVERSATION ||--|{ MESSAGE : contains
```

## Critical Rules

- **Quote node labels with special chars**: `id["Label (with parens)"]`
- **No HTML tags in labels** — they break rendering
- **No special chars in message text**: avoid `:`, `{`, `}`, `;` in sequence messages
- **Max 15-20 nodes** for readable output
- **Always test** that the syntax is valid before presenting

## Mermaid Syntax Gotchas

- Sequence diagram arrows: `->>` (solid), `-->>` (dashed), `-x` (cross)
- Flowchart direction: `TD` (top-down), `LR` (left-right), `RL`, `BT`
- Node shapes: `[]` rectangle, `()` rounded, `{}` diamond, `(())` circle
- Never use `\\n` in node labels — use separate nodes instead
'''
    },
}

for skill_name, skill_data in custom_skills.items():
    skill_dir = skills_base / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = skill_data['frontmatter'] + skill_data['body']
    (skill_dir / 'SKILL.md').write_text(content, encoding='utf-8')
    print(f'Created: {skill_name}/SKILL.md')

print('All custom skills created!')
