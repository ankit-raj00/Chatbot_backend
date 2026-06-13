---
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
# Analyzing Data

## Workflow

1. Identify the file path from the user request
2. Call `analyze_data_file` tool with the file path and the user's specific question
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
