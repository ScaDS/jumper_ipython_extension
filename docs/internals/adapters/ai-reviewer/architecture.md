---
title: Architecture
---

# AI Reviewer — Architecture

AI Reviewer turns recorded code and performance data into large language model
(LLM) analysis and ranked cell rewrites. It can benchmark each rewrite against
the original, repair a broken or divergent variant, and place a selected result
into the next notebook cell. It reuses Reporter and Monitor data; it does not
collect performance samples itself.

## Responsibilities

- Build review context from cell source, timings, metrics, tags, hardware, and
  installed packages.
- Compose strategy-controlled prompts for analysis, suggestions, refinement,
  and repair.
- Run fresh-review, benchmark-only, and apply workflows while retaining their
  state by run identifier (ID).
- Validate, replay, time, compare, and repair one-cell benchmark variants.
- Render text or notebook HyperText Markup Language (HTML) with code
  differences, measured verdicts, and resume commands.

## Structure

### System view

```mermaid
flowchart TB
    command["%perfmonitor_ai_review"] --> reviewer[AI Reviewer]
    reviewer --> workflows[Workflow graphs]
    reviewer --> state[Stored review state]

    workflows -.-> context[Context collection]
    context --> reporter[Reporter]
    context --> monitor[Monitor]

    workflows -.-> llm[Large language model]
    workflows -.-> benchmark[Benchmark]
    workflows -.-> display[Review display]
    workflows -.-> next[Next notebook cell]

    state -. resume input .-> workflows
    display --> card[Review card]
```

Solid arrows show ownership or a direct dependency. Dotted arrows show
collaborators used by particular paths, not one mandatory sequence. A fresh
review collects context and resolves a review strategy. Resume paths load
stored review state and enter benchmark or apply without collecting context
again.

| Part | Role in the workflow |
|---|---|
| AI Reviewer | Selects a workflow, coordinates collaborators, and retains review state by run identifier. |
| Context collection | Reads selected source code and existing performance data through Reporter and Monitor. |
| Review strategy | Selects which context sources and prompt rules the fresh review uses. |
| Workflow graphs | Coordinate fresh review, later benchmark, and apply without mixing their entry conditions. |
| Large language model (LLM) | Supplies analysis, suggestions, optional refinement, and repair code. |
| Benchmark | Validates every option and measures each version that can run. |
| Review display | Shows analysis, ranked code differences, measured verdicts, and follow-up commands. |

### LLM Context

To use smaller LLMs effectively, it's essential to clearly specify the task they need to perform. Therefore, our primary goal is to make context building system both flexible and convenient. The following schemes are aimed to give the user or the developer (choose your side) better view on how we approach LLM Context building in our project. **Source Query** - suggests an overall view at the context building in our project from the angle of request to the database. **Query Console** elaborates this idea and provides a fine-grained look at the picked sources and how they affect the resulting context.

#### Source Query

**Source Query** represents the general idea on Sources: Where **Root knowledge**: deterministic sources such as code and metrics, **Text sources**: text instructions and LLM generated analysis.

<div class="context-workbench-frame prompt-fullscreen-target" data-prompt-fullscreen>
  <button class="prompt-fullscreen-toggle" type="button" aria-label="Open diagram in fullscreen" aria-pressed="false">
    <svg class="panzoom-icon" viewBox="0 0 448 512" xmlns="http://www.w3.org/2000/svg">
      <path d="M32 32C14.3 32 0 46.3 0 64l0 96c0 17.7 14.3 32 32 32s32-14.3 32-32l0-64 64 0c17.7 0 32-14.3 32-32s-14.3-32-32-32L32 32zM64 352c0-17.7-14.3-32-32-32s-32 14.3-32 32l0 96c0 17.7 14.3 32 32 32l96 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l-64 0 0-64zM320 32c-17.7 0-32 14.3-32 32s14.3 32 32 32l64 0 0 64c0 17.7 14.3 32 32 32s32-14.3 32-32l0-96c0-17.7-14.3-32-32-32l-96 0zM448 352c0-17.7-14.3-32-32-32s-32 14.3-32 32l0 64-64 0c-17.7 0-32 14.3-32 32s14.3 32 32 32l96 0c17.7 0 32-14.3 32-32l0-96z"></path>
    </svg>
  </button>
  <div class="context-workbench">
  <input class="context-workbench__toggle" type="radio" name="context-request" id="context-request-analyze" checked>
  <input class="context-workbench__toggle" type="radio" name="context-request" id="context-request-suggest">
  <input class="context-workbench__toggle" type="radio" name="context-request" id="context-request-refine">

  <div class="context-workbench__tabs" role="tablist" aria-label="Request type">
    <label for="context-request-analyze" role="tab">Analyze request</label>
    <label for="context-request-suggest" role="tab">Suggest request</label>
    <label for="context-request-refine" role="tab">Refine request</label>
    <span class="context-workbench__scroll">Scroll to follow the pipeline →</span>
  </div>

  <div class="context-workbench__pane context-workbench__pane--analyze">
    <div class="context-pipeline">
      <section class="context-stage context-stage--sources">
        <header><span>01</span><strong>Sources</strong><small>one logical catalog</small></header>
        <div class="context-source-group">
          <b>Root knowledge</b>
          <div class="context-chips">
            <span>Cell code</span><span>Exact timing</span><span>Tags</span>
            <span>Metric summary</span><span>Raw metrics</span><span>Hardware</span>
          </div>
          <small><code>reporter.py</code> · monitor nodes · <code>collector.py</code></small>
        </div>
        <div class="context-source-group context-source-group--text">
          <b>Text sources</b>
          <div class="context-file"><code>analyze/template.md</code><span>fixed role · task · layout</span></div>
          <div class="context-file"><code>analyze/spec.yaml</code><span><em>given</em> · <em>hints</em> · enabled defaults</span></div>
        </div>
      </section>

      <div class="context-pipeline__arrow"><span>query</span></div>

      <section class="context-stage context-stage--query">
        <header><span>02</span><strong>Selection plan</strong><small>resolve before reading</small></header>
        <div class="context-rule"><b>Request profile</b><code>prompt_id = analyze</code></div>
        <div class="context-rule"><b>Strategy</b><span><code>--strategy</code> or default ID</span><small>matching toggles override <code>enabled</code></small></div>
        <div class="context-rule"><b>Scope</b><span><code>--cells</code> · <code>--level</code></span><small>limits runtime records only</small></div>
      </section>

      <div class="context-pipeline__arrow"><span>project</span></div>

      <section class="context-stage context-stage--messages">
        <header><span>03</span><strong>Message projections</strong><small>two independent branches</small></header>
        <div class="context-message context-message--system">
          <b>SystemMessage</b>
          <span>template + enabled <em>given</em> + enabled <em>hints</em></span>
          <small>role · input descriptions · bottleneck task</small>
        </div>
        <div class="context-message context-message--human">
          <b>HumanMessage</b>
          <span>non-empty enabled source values</span>
          <small>code · timing · tags · metrics · hardware</small>
        </div>
      </section>

      <div class="context-pipeline__arrow"><span>assemble</span></div>

      <section class="context-stage context-stage--request">
        <header><span>04</span><strong>LLM context</strong><small>Analyze request</small></header>
        <div class="context-envelope">
          <span>SystemMessage</span>
          <i>+</i>
          <span>HumanMessage</span>
        </div>
      </section>
    </div>
  </div>

  <div class="context-workbench__pane context-workbench__pane--suggest">
    <div class="context-pipeline">
      <section class="context-stage context-stage--sources">
        <header><span>01</span><strong>Sources</strong><small>one logical catalog</small></header>
        <div class="context-source-group">
          <b>Root knowledge</b>
          <div class="context-chips">
            <span>Cell code</span><span>Hardware</span><span>Installed packages</span>
          </div>
          <small><code>reporter.py</code> · monitor · <code>default.yaml</code> package names · <code>collector.py</code> installed versions</small>
        </div>
        <div class="context-source-group context-source-group--text">
          <b>Text sources</b>
          <div class="context-file context-file--runtime"><b>Bottleneck analysis</b><span>result of the preceding Analyze request</span></div>
          <div class="context-file"><code>suggest/template.md</code><span>fixed role · task · layout</span></div>
          <div class="context-file"><code>suggest/spec.yaml</code><span><em>given</em> · <em>rules</em> · note slot</span></div>
          <div class="context-file"><code>fragments/*.md</code><span>referenced rule text</span></div>
          <div class="context-file"><code>--note</code><span>optional runtime text · hard requirement</span></div>
        </div>
      </section>

      <div class="context-pipeline__arrow"><span>query</span></div>

      <section class="context-stage context-stage--query">
        <header><span>02</span><strong>Selection plan</strong><small>resolve before reading</small></header>
        <div class="context-rule"><b>Request profile</b><code>prompt_id = suggest</code></div>
        <div class="context-rule"><b>Strategy</b><span><code>--strategy</code> or default ID</span><small>matching toggles override <code>enabled</code>; <code>require_note</code> only validates</small></div>
        <div class="context-rule"><b>Text input</b><span><code>--note</code></span><small>queried with Text sources; does not rewrite strategy</small></div>
      </section>

      <div class="context-pipeline__arrow"><span>project</span></div>

      <section class="context-stage context-stage--messages">
        <header><span>03</span><strong>Message projections</strong><small>two independent branches</small></header>
        <div class="context-message context-message--system">
          <b>SystemMessage</b>
          <span>template + enabled <em>given</em> + enabled <em>rules</em> + note</span>
          <small>role · generation rules · output rules</small>
        </div>
        <div class="context-message context-message--human">
          <b>HumanMessage</b>
          <span>non-empty enabled source values</span>
          <small>analysis · code · hardware · package versions</small>
        </div>
      </section>

      <div class="context-pipeline__arrow"><span>assemble</span></div>

      <section class="context-stage context-stage--request">
        <header><span>04</span><strong>LLM context</strong><small>Suggest request</small></header>
        <div class="context-envelope">
          <span>SystemMessage</span>
          <i>+</i>
          <span>HumanMessage</span>
          <i>+</i>
          <span>Response schema</span>
        </div>
      </section>
    </div>
  </div>

  <div class="context-workbench__pane context-workbench__pane--refine">
    <div class="context-pipeline">
      <section class="context-stage context-stage--sources">
        <header><span>01</span><strong>Sources</strong><small>stored review + resume input</small></header>
        <div class="context-source-group">
          <b>Root knowledge</b>
          <div class="context-chips">
            <span>Selected option full code</span>
            <span>Other options' diffs</span>
          </div>
        </div>
        <div class="context-source-group context-source-group--text">
          <b>Text sources</b>
          <div class="context-file context-file--runtime"><b>Bottleneck analysis</b><span>stored result of the preceding Analyze request</span></div>
          <div class="context-file"><code>refine/template.md</code><span>fixed role · adjustment task · output contract</span></div>
          <div class="context-file"><code>refine/spec.yaml</code><span>enabled style rules</span></div>
          <div class="context-file"><code>fragments/*.md</code><span>referenced Python or R style text</span></div>
          <div class="context-file"><code>--note</code><span>required custom instruction</span></div>
        </div>
      </section>

      <div class="context-pipeline__arrow"><span>query</span></div>

      <section class="context-stage context-stage--query">
        <header><span>02</span><strong>Selection plan</strong><small>resume one stored review</small></header>
        <div class="context-rule"><b>Request profile</b><code>prompt_id = refine</code></div>
        <div class="context-rule"><b>Stored review</b><span><code>--resume RUN_ID</code></span><small>loads analysis, options, target code, and the original strategy toggles</small></div>
        <div class="context-rule"><b>Option</b><span><code>--select N</code></span><small>selects a title and full code; every other option becomes a diff against its own original target cell</small></div>
        <div class="context-rule"><b>Text input</b><span><code>--note</code></span><small>without it, resume applies directly and no Refine request is sent</small></div>
      </section>

      <div class="context-pipeline__arrow"><span>project</span></div>

      <section class="context-stage context-stage--messages">
        <header><span>03</span><strong>Message projections</strong><small>two independent branches</small></header>
        <div class="context-message context-message--system">
          <b>SystemMessage</b>
          <span>template + enabled style rules</span>
          <small>role · adjustment task · use-of-diffs constraint · code-only output</small>
        </div>
        <div class="context-message context-message--human">
          <b>HumanMessage</b>
          <span>stored values + derived diffs + new instruction</span>
          <small>analysis · other-option diffs · selected full code · <code>--note</code></small>
        </div>
      </section>

      <div class="context-pipeline__arrow"><span>assemble</span></div>

      <section class="context-stage context-stage--request">
        <header><span>04</span><strong>LLM context</strong><small>Refine request</small></header>
        <div class="context-envelope">
          <span>SystemMessage</span>
          <i>+</i>
          <span>HumanMessage</span>
        </div>
      </section>
    </div>
  </div>
</div>
</div>

!!! note "Messages are projections"

    HumanMessage and SystemMessage are not records in the source catalog but conventional logical parts of the prompt, coming from **LangChain** library.

##### Precedence

1. The graph step selects the Analyze, Suggest, or Refine profile.
2. `--strategy` selects a `strategies.yaml` entry; `default.yaml` supplies the
   ID only when the option is omitted.
3. The selected strategy's merged toggles override matching source and prompt
   item defaults.
4. Suggest consumes the bottleneck analysis from the preceding Analyze call,
   together with code, hardware, and installed package versions.
5. Scope options select fresh-review records. For Refine, `--resume` reloads the
   stored review and its strategy toggles; `--select` chooses one option.
   `--note` is the Human-message instruction that makes the Refine call run.

#### Query Interactive Console

Choose the request, strategy, and whether `--note` is present. The selected
sources, query plan, and returned `SystemMessage` and `HumanMessage` update
together.

<div class="query-console-material-shell prompt-fullscreen-target" data-prompt-fullscreen>
  <button class="prompt-fullscreen-toggle" type="button" aria-label="Open diagram in fullscreen" aria-pressed="false">
    <svg class="panzoom-icon" viewBox="0 0 448 512" xmlns="http://www.w3.org/2000/svg">
      <path d="M32 32C14.3 32 0 46.3 0 64l0 96c0 17.7 14.3 32 32 32s32-14.3 32-32l0-64 64 0c17.7 0 32-14.3 32-32s-14.3-32-32-32L32 32zM64 352c0-17.7-14.3-32-32-32s-32 14.3-32 32l0 96c0 17.7 14.3 32 32 32l96 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l-64 0 0-64zM320 32c-17.7 0-32 14.3-32 32s14.3 32 32 32l64 0 0 64c0 17.7 14.3 32 32 32s32-14.3 32-32l0-96c0-17.7-14.3-32-32-32l-96 0zM448 352c0-17.7-14.3-32-32-32s-32 14.3-32 32l0 64-64 0c-17.7 0-32 14.3-32 32s14.3 32 32 32l96 0c17.7 0 32-14.3 32-32l0-96z"></path>
    </svg>
  </button>
  <iframe
    id="query-console-material"
    src="../prompt-assembly/query-console-material-embed.html"
    title="Prompt assembly query console — Material style"
    loading="lazy">
  </iframe>
</div>

<script>
(() => {
  const frame = document.getElementById("query-console-material");
  if (!frame) return;

  const applyMaterialStyle = () => {
    const doc = frame.contentDocument;
    if (!doc) return;

    doc.querySelector("header")?.remove();
    if (doc.getElementById("query-console-material-style")) return;

    const style = doc.createElement("style");
    style.id = "query-console-material-style";
    style.textContent = `
      :root {
        --ground: #f8faf9;
        --surface: rgba(255, 255, 255, 0.88);
        --surface-sunk: rgba(255, 255, 255, 0.76);
        --ink: #202124;
        --ink-soft: #5f6368;
        --ink-faint: #80868b;
        --rule: rgba(32, 33, 36, 0.20);
        --accent: #3f51b5;
        --accent-soft: #eef0ff;
        --off: #b05f52;
        --off-soft: #faeae7;
        --source: #edf7ef;
        --query: #fff8df;
        --result: #f3edff;
        --human: #eaf2ff;
        --radius: 14px;
      }

      body {
        background:
          radial-gradient(circle at 10% 0%, rgba(52, 168, 83, 0.08), transparent 24rem),
          radial-gradient(circle at 90% 100%, rgba(124, 77, 255, 0.08), transparent 24rem),
          var(--ground);
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      }

      .page {
        min-width: 1080px;
        max-width: 1240px;
        padding: 18px;
        gap: 18px;
      }

      header { display: none !important; }

      .controls {
        position: sticky;
        top: 0;
        z-index: 5;
        padding: 12px 14px;
        border-radius: var(--radius);
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 8px 24px rgba(32, 33, 36, 0.08);
        backdrop-filter: blur(10px);
      }

      .controls__label {
        font-family: inherit;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: var(--ink-faint);
      }

      .chip {
        border-radius: 999px;
        padding: 0.36rem 0.72rem;
        font-family: inherit;
        font-size: 0.78rem;
        font-weight: 650;
        background: rgba(255, 255, 255, 0.72);
        transition: 150ms ease;
      }

      .chip:hover {
        border-color: var(--accent);
        color: var(--accent);
        transform: translateY(-1px);
      }

      .chip[aria-pressed="true"] {
        color: var(--accent);
        border-color: rgba(63, 81, 181, 0.52);
        background: var(--accent-soft);
        box-shadow: 0 2px 8px rgba(63, 81, 181, 0.10);
      }

      .stages {
        grid-template-columns: 1.05fr 0.85fr 1.1fr;
        gap: 16px;
        align-items: stretch;
      }

      .stage {
        overflow: hidden;
        border-radius: var(--radius);
        box-shadow: 0 6px 18px rgba(32, 33, 36, 0.07);
      }

      .stage:nth-child(1) { background: var(--source); }
      .stage:nth-child(2) { background: var(--query); }
      .stage:nth-child(3) { background: var(--result); }

      .stage__head {
        padding: 13px 14px;
        background: rgba(255, 255, 255, 0.42);
      }

      .stage__title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--ink);
        font-family: inherit;
        font-size: 0.82rem;
        font-weight: 750;
        letter-spacing: 0;
        text-transform: none;
      }

      .stage__title::before {
        display: grid;
        place-items: center;
        width: 1.75rem;
        height: 1.75rem;
        border-radius: 50%;
        color: #fff;
        background: var(--ink);
        font-size: 0.62rem;
        font-weight: 800;
      }

      .stage:nth-child(1) .stage__title::before { content: "01"; }
      .stage:nth-child(2) .stage__title::before { content: "02"; }
      .stage:nth-child(3) .stage__title::before { content: "03"; }

      .stage__count {
        font-family: inherit;
        font-size: 0.66rem;
      }

      .group {
        margin: 10px;
        padding: 11px;
        border: 1px solid var(--rule) !important;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.72);
      }

      .group + .group { margin-top: 0; }

      .group__name {
        color: var(--ink);
        font-family: inherit;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.025em;
        text-transform: none;
      }

      .row__flag { background: #34a853; }
      .row__id {
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
        font-size: 0.78rem;
        font-weight: 750;
      }
      .row__meta { font-size: 0.7rem; }

      .plan {
        margin: 10px;
        padding: 13px;
        border: 1px solid var(--rule);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.72);
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
        font-size: 0.75rem;
      }

      .plan .kw { color: #9a7100; font-weight: 750; }
      .plan pre { font-family: inherit; }

      .message {
        margin: 10px;
        padding: 12px;
        border: 1px solid var(--rule) !important;
        border-radius: 10px;
      }

      .message:nth-child(2) {
        border-left: 4px solid #7c4dff !important;
        background: rgba(243, 237, 255, 0.86);
      }

      .message:nth-child(3) {
        margin-top: 0;
        border-left: 4px solid #4285f4 !important;
        background: var(--human);
      }

      .message__role {
        color: var(--ink);
        font-family: inherit;
        font-weight: 750;
        letter-spacing: 0;
        text-transform: none;
      }

      .message__body {
        border: 1px solid rgba(32, 33, 36, 0.10);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.68);
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
        font-size: 0.72rem;
      }

      .message__body b { color: #6241c7; }

      .legend {
        padding: 10px 13px;
        border: 1px solid var(--rule);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.74);
      }

      .swatch { background: #34a853; }
      .swatch--off {
        background: transparent;
        border: 1px solid var(--off);
      }
      .swatch--none {
        background: var(--ink-faint);
        opacity: 0.55;
      }

      footer {
        padding: 13px;
        border: 1px solid rgba(63, 81, 181, 0.24);
        border-radius: 10px;
        background: rgba(232, 240, 254, 0.72);
        max-width: none;
      }

      footer code,
      .inline-code {
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
        font-weight: 700;
      }
    `;
    doc.head.appendChild(style);
  };

  frame.addEventListener("load", applyMaterialStyle);
  if (frame.contentDocument?.readyState === "complete") applyMaterialStyle();
})();
</script>

##### What the panels show

| Panel | Content |
|---|---|
| Sources | Root knowledge and Text-source rows, including whether this request selects each row |
| Query | Scope, selected identifiers, strategy overrides, and values that are never queried |
| Returned context | The rendered `SystemMessage` and the runtime values placed in `HumanMessage` |

Suggest receives the bottleneck analysis produced by Analyze. Refine reloads
that analysis and the stored options. `--select` supplies the selected title
and full code; every other option is converted to a diff against its own
original target cell; `--note` becomes the custom instruction. Refine inherits
the original review's strategy toggles.

### Workflows

```mermaid
flowchart TB
    START(["%perfmonitor_ai_review"]) --> kind{Arguments}

    kind -->|"fresh review (no resume)"| collect["Collect report context"]
    collect --> analyze["Analyze the bottleneck"]
    analyze --> suggest["Generate ranked options"]
    suggest --> gate{"--benchmark?"}
    gate -->|no| card["Display the review card"]
    gate -->|yes| bench["Benchmark the suggestions"]
    bench --> card

    kind -->|"--resume ID --benchmark"| bench
    kind -->|"--resume ID --select N"| chosen["Load the chosen option"]
    chosen --> note{"--note given?"}
    note -->|yes| refine["Rewrite it from the note"]
    refine --> next["Place code in the next cell"]
    note -->|no| next
```

- A fresh review collects recorded cell code and performance data, requests an
  analysis and ranked options, and optionally benchmarks them before display.
- A later benchmark reuses stored suggestions without repeating context
  collection, analysis, or suggestion calls.
- Apply loads one stored option, optionally refines it from a user note, and
  places the result in the next cell without running it.

### Benchmark

Benchmarking turns a proposal into a measured verdict. For a one-cell review,
it:

1. Measures the original cell as the baseline.
2. Validates every option and measures each version that can run.
3. Compares each result with the baseline and reports timing together with
   whether the result still matches.

Multi-cell reviews remain reviewable, but their benchmark step is skipped
because each target needs its own baseline and notebook state.

#### Replay paths

A replay runs the target cell in a separate operating-system process.

!!! note "State reconstruction"
    Before the target runs, the replay reconstructs the prefix: all recorded
    cells before the target. The replay path decides how often this work is
    repeated. The separate process isolates measurements from each other, not
    from the machine. Replayed code can still overwrite real files or repeat
    real network actions, such as submitting the same job twice.

```mermaid
flowchart TB
    bench[Benchmark] --> mode{Configured path}

    mode --> full["full<br/>replay the prefix every time"]
    mode --> fork["fork<br/>run the prefix once, copy the process"]
    mode --> dill["dill<br/>save the prefix state once, restore it"]

    fork --> check{Baseline agrees with full?}
    dill --> check
    check -->|no, or path unusable| fallback[Switch to full and restart]
    fallback --> full
    check -->|yes| target[Run one target cell]
    full --> target
    target --> outcome["Exported session → duration, metrics, result summaries"]
```

| Path | Language | Trade-off |
|---|---|---|
| `full` | Every supported language | The reference result; repeats all prefix work. |
| `fork` | Python | Reduces repeated setup, but copied memory can inflate memory figures and library worker threads may not survive the copy. |
| `dill` | Python | No process-copy effects; only safely saveable state survives, and the checkpoint can cost more than it saves. |

`fork` and `dill` accelerate Python benchmarks. By default, their baseline
result is checked once against `full`. If a path is unavailable, unsafe for the
recorded state, or produces a different baseline, the benchmark falls back to
`full`. Measurements already taken through the old path are discarded so one
verdict never mixes replay paths.

#### Measurement and repair concurrency

```mermaid
sequenceDiagram
    participant M as Measure loop (sequential)
    participant P as Repair pool (threads)
    participant L as LLM

    M->>M: baseline × runs
    M->>M: option 1 × runs ✓
    M->>M: option 2 → crash
    M->>P: submit repair(option 2)
    P->>L: fix prompt (network wait)
    M->>M: option 3 × runs ✓
    Note over M,P: option 2 is being repaired<br/>while option 3 is being timed
    L-->>P: fixed code
    P-->>M: re-queue option 2
    M->>M: option 2 × runs ✓
```

The LLM repairs option 2 while the measure loop times option 3, reducing total
benchmark time without distorting measurements. Measurements remain sequential
so options do not compete for processor, memory, or graphics-card resources.
Returned code re-enters the same queue and passes the checks again.

#### Repair loop

```mermaid
stateDiagram-v2
    direction TB

    [*] --> Syntax: fresh or repaired code
    Syntax --> Measure: valid with timing enabled
    Syntax --> Validated: valid with timing disabled
    Syntax --> Reject: invalid

    Measure --> Accepted: matches or unverified
    Measure --> Reject: crash or timeout
    Measure --> Remember: results differ
    Remember --> Reject: retain code and verdict

    Reject --> Repair: attempt available
    Reject --> Finalize: attempts exhausted
    Repair --> Syntax: code returned
    Repair --> Finalize: empty response or error

    Finalize --> Divergent: saved measurement exists
    Finalize --> Failed: no saved measurement

    Validated --> [*]
    Accepted --> [*]
    Divergent --> [*]
    Failed --> [*]
```

Syntax failures, crashes, timeouts, and changed results share one repair
budget. A repaired version is checked from the beginning. If repair is
exhausted, the final state depends on whether any version completed a
measurement. When timing is disabled with **`--skip-check run`**, valid syntax
settles as validated and invalid code follows the same repair path.

- **`Divergent`** — At least one version ran, but its result differed from the
  baseline. The benchmark keeps that real measurement and marks its speedup as
  unearned.
- **`Failed`** — No version completed a measurement before repair stopped
  because attempts were exhausted or the LLM returned no usable code. There is
  no timing verdict to report.

## Design patterns

| Class | Pattern | Implementation role |
|---|---|---|
| `AIReviewer` | **Facade** | Coordinates context collection, compiled graphs, benchmarking, display, and pending review state behind the service-facing review operations. |
| `AIReviewerProtocol` | **Structural Subtyping** | Defines the service contract shared by the available and unavailable reviewer implementations. |
| `UnavailableAIReviewer` | **Null Object** | Keeps the service dependency valid when the optional AI packages are absent and reports how to enable them. |
| `LanguageAdapter` | **Strategy** | Supplies language-specific syntax validation, output-name extraction, and replay rendering. |
| `ReplayStrategy` | **Strategy** | Selects how benchmark prefix state is reconstructed without changing result reading or scoring. |

| Method | Pattern | Implementation role |
|---|---|---|
| `AIReviewer.attach()` | **Dependency Injection** | Binds the live or imported monitor after reviewer construction. |
| `BenchmarkOrchestrator._drive()` | **Work Queue** | Serializes candidate processing while tracking concurrent repair futures and re-queuing returned code. |

| Function | Pattern | Implementation role |
|---|---|---|
| `build_ai_reviewer()` | **Factory** | Selects the real reviewer or its unavailable implementation according to installed dependencies. |
| `register_adapter()` | **Registry** | Maps a recorded cell language to its benchmark adapter. |
| `register_strategy()` | **Registry** | Maps a configured replay mode to its replay strategy. |

## Key files

| File | Role |
|---|---|
| `adapters/ai_reviewer/reviewer.py` | Exposes the adapter facade, stores pending reviews, and assembles benchmark dependencies |
| `adapters/ai_reviewer/agent/graph.py` | Defines the fresh-review, benchmark-only, and apply LangGraph workflows |
| `adapters/ai_reviewer/agent/nodes.py` | Implements prompt calls, benchmark integration, display, refinement, and apply nodes |
| `adapters/ai_reviewer/context/collector.py` | Builds strategy-controlled review context from Reporter and Monitor data |
| `adapters/ai_reviewer/strategy/strategies.yaml` | Declares named prompt and context strategies |
| `adapters/ai_reviewer/benchmark/orchestrator.py` | Owns sequential measurement, verdicts, retries, and the shared repair queue |
| `adapters/ai_reviewer/benchmark/runner.py` | Runs replay strategies and reads exported sessions through the normal reporting path |
| `adapters/ai_reviewer/benchmark/replay/base.py` | Defines the interchangeable prefix-state replay contract |
| `adapters/ai_reviewer/language/base.py` | Defines the per-language benchmark contract and capabilities |
| `adapters/ai_reviewer/ui/review_display.py` | Renders text and HTML diffs, verdicts, and resume commands |

## Notes

- Review state lives in the running Python process and is lost when it restarts.
- Benchmarking accepts one reviewed cell at a time.
- A replay is not the live notebook: interaction with the notebook user
  interface may not be reproducible under any replay path.
