---
title: Architecture
---

# Reporter — Architecture

Reporter turns cell history and metric samples into a compact performance
summary. It prepares one shared report model, classifies it with Analyzer, and
renders plain text or HyperText Markup Language (HTML) for notebook display. It
does not collect samples.

## Responsibilities

- Select a cell range and its matching performance samples.
- Compute duration, hardware limits, and ranked performance labels.
- Render the same information as terminal text or notebook HTML.
- Build a cell-labeled context for internal consumers.
- Degrade cleanly when rich notebook display is unavailable.

## Structure

### Class relationships

```mermaid
classDiagram
    direction LR

    class PerformanceReporter
    class ReportPrinter
    class ReportBuilder
    class ReportDisplayerProtocol {
        <<Protocol>>
        display()
    }
    class ReportDisplayer
    class UnavailableReportDisplayer

    PerformanceReporter *-- ReportPrinter
    PerformanceReporter *-- ReportDisplayerProtocol
    ReportPrinter --|> ReportBuilder
    ReportDisplayer --|> ReportBuilder
    ReportDisplayer ..|> ReportDisplayerProtocol
    UnavailableReportDisplayer ..|> ReportDisplayerProtocol
```

`build_performance_reporter()` constructs the facade and its two output
collaborators. `ReportPrinter` and `ReportDisplayer` receive the same
`PerformanceAnalyzer` instance.

### Report data flow

```mermaid
flowchart TB
    history[CellHistory]
    samples[Monitor.nodes]
    analyzer[PerformanceAnalyzer]

    prepare["ReportBuilder.prepare_report_data()"]
    model["Prepared report data<br/>cells · samples · tags · duration"]

    printer["ReportPrinter.print()"]
    displayer["ReportDisplayer.display()"]
    context["PerformanceReporter.build_context()"]

    terminal[Terminal text]
    template[Jinja2 report template]
    notebook[Notebook HTML]
    context_data[Context dictionary]

    history --> prepare
    samples --> prepare
    analyzer --> prepare
    prepare --> model

    model --> printer
    model --> displayer
    model --> context

    printer --> terminal
    displayer --> template
    template --> notebook
    context --> context_data

    classDef dependency fill:#e6f4ea,stroke:#34a853
    classDef pipeline fill:#f3e8fd,stroke:#9334e6
    classDef component fill:#e8f0fe,stroke:#4285f4
    classDef output fill:#f8f9fa,stroke:#80868b
    class history,samples,analyzer dependency
    class prepare,model pipeline
    class printer,displayer,context component
    class terminal,template,notebook,context_data output
```

The unavailable displayer follows the same `ReportDisplayerProtocol` contract,
but reports why rich output is disabled instead of entering the data pipeline.

## Design patterns

| Class | Pattern | Implementation role |
|---|---|---|
| `PerformanceReporter` | **Facade** | Exposes text, HTML, and AI-context operations over the reporting subsystem. |
| `UnavailableReportDisplayer` | **Null Object** | Preserves the display contract when rich notebook output is disabled. |
| `ReportDisplayerProtocol` | **Structural Subtyping** | Defines the display contract implemented by both real and unavailable displayers. |

| Method | Pattern | Implementation role |
|---|---|---|
| `ReportBuilder.prepare_report_data()` | **Processing Pipeline** | Applies range resolution, sample filtering, classification, and summary calculation in a fixed order. |
| `PerformanceReporter.attach()` | **Dependency Injection** | Rebinds the printer and displayer to a live or imported monitor after construction. |

| Function | Pattern | Implementation role |
|---|---|---|
| `build_performance_reporter()` | **Factory** | Constructs a reporter with one shared analyzer and the selected display implementation. |

## Key files

| File | Role |
|---|---|
| `adapters/reporter.py` | Prepares report data and coordinates text and HTML output |
| `adapters/analyzer.py` | Produces ranked performance labels |
| `templates/report/report.html` | Defines notebook report markup |
| `templates/report/styles.css` | Defines notebook report presentation |

## Notes

- With no explicit range, a report uses the last cell whose duration is at least
  one monitor interval.
- `build_context()` removes idle gaps and labels samples with their cell index.
- The printer and HTML displayer share one analyzer instance.
- Despite its name, `ReportBuilder` is a shared data-preparation base class, not
  an implementation of the **Builder** design pattern.
