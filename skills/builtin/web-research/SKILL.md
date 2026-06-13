---
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
# Web Research

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
