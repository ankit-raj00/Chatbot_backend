---
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
# Generating Mermaid Diagrams

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
- Never use `\n` in node labels — use separate nodes instead
