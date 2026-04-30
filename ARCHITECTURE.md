# Architecture: Multi-Agent ML System with Fedot.Industrial

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DOCKER COMPOSE                               │
│                        (ml-network)                                 │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│  │   FRONTEND    │    │   BACKEND    │    │    FEDOT SERVER       │ │
│  │  (Streamlit)  │───▶│  (FastAPI)   │───▶│    (FastAPI)          │ │
│  │  :8502        │    │  :8001       │    │    :8000              │ │
│  └──────────────┘    └──────┬───────┘    └───────────┬───────────┘ │
│                             │                        │              │
│                             │                   ┌────▼────┐        │
│                      ┌──────▼───────┐           │ Fedot   │        │
│                      │   OLLAMA     │           │Industrial│        │
│                      │  (llama3.1)  │           │ Library  │        │
│                      │  :11434      │           └─────────┘        │
│                      └──────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Agent Orchestration Flow

```
                    ┌─────────────────────┐
                    │     USER INPUT      │
                    │  CSV + Task Type +  │
                    │  Fedot Config       │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │    ORCHESTRATOR     │
                    │  (orchestrator.py)  │
                    │                     │
                    │  1. Load CSV        │
                    │  2. DataProfiler    │
                    │  3. Build Context   │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────────────┐
              │          ITERATION LOOP                │
              │                                        │
              │   ┌──────────────────────────┐        │
              │   │   STEP 1: ARCHITECT      │        │
              │   │                          │        │
              │   │   Tools:                 │        │
              │   │   ├─ get_data_profile    │───┐    │
              │   │   ├─ get_available_ops   │   │    │
              │   │   ├─ propose_baselines   │   │    │
              │   │   ├─ propose_fedot_config│   │    │
              │   │   └─ mutate_fedot_config │   │    │
              │   │                          │   │    │
              │   │   Output:                │   │    │
              │   │   ├─ Baseline pipelines  │   │    │
              │   │   └─ FedotConfig         │   │    │
              │   └──────────┬───────────────┘   │    │
              │              │                   │    │
              │              ▼                   │    │
              │   ┌──────────────────────────┐   │    │
              │   │   STEP 2: ENGINEER       │   │    │
              │   │                          │   │    │
              │   │   Tools:                 │   │    │
              │   │   ├─ train_baseline ─────┤   │    │
              │   │   ├─ call_fedot ─────────┤───┤    │
              │   │   └─ compare_results     │   │    │
              │   │                          │   │    │
              │   │   Output:                │   │    │
              │   │   ├─ Baseline scores     │   │    │
              │   │   ├─ Fedot score         │   │    │
              │   │   └─ Comparison          │   │    │
              │   └──────────┬───────────────┘   │    │
              │              │                   │    │
              │              ▼                   │    │
              │   ┌──────────────────────────┐   │    │
              │   │   STEP 3: CRITIC         │   │    │
              │   │                          │   │    │
              │   │   Tools:                 │   │    │
              │   │   ├─ analyze_errors      │   │    │
              │   │   ├─ get_feature_import  │   │    │
              │   │   ├─ explain_model ──────┤───┤    │
              │   │   ├─ get_add_metrics ────┤───┘    │
              │   │   └─ gen_struct_feedback │        │
              │   │                          │        │
              │   │   Output:                │        │
              │   │   ├─ CriticFeedback      │        │
              │   │   │  ├─ winner           │        │
              │   │   │  ├─ fedot_changes ───┤──┐     │
              │   │   │  └─ should_stop      │  │     │
              │   └──────────┬───────────────┘  │     │
              │              │                  │     │
              │              │    ┌─────────────┘     │
              │              │    │ structured        │
              │              │    │ feedback           │
              │              │    │                    │
              │              ▼    ▼                    │
              │   ┌──────────────────────┐            │
              │   │  should_stop?        │            │
              │   │  YES ──▶ break       │            │
              │   │  NO  ──▶ next iter ──┤──▶ ARCHITECT
              │   └──────────────────────┘            │
              │                                        │
              └────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   STEP 4: SCRIBE    │
                    │   (called ONCE)     │
                    │                     │
                    │   Tools:            │
                    │   └─ generate_report│
                    │                     │
                    │   Output:           │
                    │   └─ ScribeReport   │
                    └─────────────────────┘
```

## Tool Calling Mechanism

```
┌─────────────────────────────────────────────────────────────┐
│                    BaseAgent.call_llm()                      │
│                                                              │
│   ┌──────────┐     ┌──────────────┐     ┌────────────────┐ │
│   │  System   │     │   Ollama     │     │  Tool          │ │
│   │  Prompt + │────▶│   llama3.1   │────▶│  Registry      │ │
│   │  User Msg │     │              │     │                │ │
│   │  + Tools  │     │  tool_calls? │     │ _tool_handlers │ │
│   └──────────┘     └──────┬───────┘     └───────┬────────┘ │
│                           │                      │          │
│                    ┌──────▼───────┐              │          │
│                    │  Has tool    │  YES         │          │
│                    │  calls?  ────┤─────────────▶│          │
│                    │              │   execute     │          │
│                    │  NO ─────┐   │   tool       │          │
│                    └──────────┘   │              │          │
│                           │       │     ┌────────▼───────┐ │
│                           ▼       │     │ Tool result    │ │
│                    ┌──────────┐   │     │ appended to    │ │
│                    │  Final   │   │     │ messages as    │ │
│                    │  text    │   │     │ role: "tool"   │ │
│                    │  response│   │     └────────┬───────┘ │
│                    └──────────┘   │              │          │
│                                   │     ┌────────▼───────┐ │
│                                   │     │ Call LLM again │ │
│                                   │     │ (next round)   │ │
│                                   │     └────────────────┘ │
│                                   │                         │
│                    max_rounds = 10                          │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Between Agents

```
┌──────────────┐                              ┌──────────────┐
│  DataContext  │──────────────────────────────│  Shared by   │
│              │                              │  ALL agents  │
│  X_train     │                              │              │
│  y_train     │                              │              │
│  X_val       │                              │              │
│  y_val       │                              │              │
│  csv_path    │                              │              │
│  task_type   │                              │              │
│  profile     │                              │              │
│  forecast_len│                              │              │
└──────────────┘                              └──────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Architect   │     │  Engineer    │     │     Critic       │
│  Result      │────▶│  Result      │────▶│     Feedback     │
│              │     │              │     │                  │
│ baselines:   │     │ baseline_    │     │ winner:          │
│  {lr, xgb..} │     │  results[]   │     │  "architect"     │
│              │     │              │     │                  │
│ fedot_config:│     │ fedot_result:│     │ suggested_fedot_ │
│  preset:auto │     │  score: 0.92 │     │  changes:        │
│  strategy:   │     │  metrics:{}  │     │  {timeout: 10,   │
│   kernel_    │     │              │     │   preset: best_, │
│   automl     │     │ comparison:  │     │   strategy:      │
│  task_params:│     │  best: fedot │     │   kernel_automl} │
│  {forecast:  │     │              │     │                  │
│    14}       │     │ fedot_config │     │ should_stop:     │
│              │     │  _used: {}   │     │  false           │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
                                          ┌────────▼─────────┐
                                          │  Back to         │
                                          │  ARCHITECT       │
                                          │  (next iteration)│
                                          └──────────────────┘
```

## External Service Interactions

```
┌─────────────────────────────────────────────────────┐
│                    BACKEND                           │
│                                                      │
│  Architect ──tool──▶ DataProfiler.profile()          │
│                      (local, no HTTP)                │
│                                                      │
│  Engineer  ──tool──▶ sklearn.fit() / .predict()      │
│            │         (local, no HTTP)                │
│            │                                         │
│            └─tool──▶ POST fedot-server:8000/mcp      │
│                      {datapath, target, problem,     │
│                       timeout, preset, strategy,     │
│                       task_params, available_ops...}  │
│                                                      │
│  Critic   ──tool──▶ POST fedot-server:8000/explain   │
│            │         {method: "point"|"shap"|...}    │
│            │                                         │
│            └─tool──▶ POST fedot-server:8000/get_metrics│
│                      {metric_names: ["rmse","mae"]}  │
│                                                      │
│  ALL agents ──LLM──▶ ollama:11434                    │
│                      model: llama3.1                 │
│                      + tools: [...]                  │
└─────────────────────────────────────────────────────┘
```

## Supported Task Types

```
┌────────────────────────────────────────────────────────────┐
│                    TASK TYPES                               │
│                                                             │
│  Standard ML:                                               │
│  ├─ classification ── sklearn baselines + Fedot            │
│  └─ regression ────── sklearn baselines + Fedot            │
│                                                             │
│  Time Series:                                               │
│  ├─ ts_forecasting ── Fedot only (needs forecast_length)   │
│  │   Models: ar, stl_arima, ets, nbeats, tcn, deepar      │
│  │   Strategies: forecasting_assumptions, exogenous        │
│  │                                                          │
│  ├─ ts_classification ── Fedot only                        │
│  │   Models: stat_clf, freq_clf, manifold_clf, inception   │
│  │   Preprocessing: wavelet, fourier, topological, rocket  │
│  │                                                          │
│  ├─ ts_regression ── Fedot only                            │
│  │   Models: stat_reg, freq_reg, manifold_reg, tst         │
│  │                                                          │
│  └─ anomaly_detection ── Fedot only                        │
│      Models: sst, iforest, one_class_svm, kalman           │
│      Models (DL): conv_ae_detector, lstm_ae_detector       │
└────────────────────────────────────────────────────────────┘
```

## Docker Services

```
┌─────────────────────────────────────────────────────┐
│  docker-compose.yml                                  │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   ollama     │  │ fedot-server│  │   backend   │ │
│  │             │  │             │  │             │ │
│  │ ollama/     │  │ Python 3.10 │  │ Python 3.10 │ │
│  │ ollama:     │  │ + Fedot.Ind │  │ + FastAPI   │ │
│  │ latest      │  │ + FastAPI   │  │ + sklearn   │ │
│  │             │  │             │  │ + xgboost   │ │
│  │ Model:      │  │ Port: 8000  │  │ + ollama    │ │
│  │ llama3.1    │  │             │  │             │ │
│  │ Port: 11434 │  │ Endpoints:  │  │ Port: 8001  │ │
│  │             │  │ /mcp        │  │             │ │
│  │             │  │ /explain    │  │ Endpoints:  │ │
│  │             │  │ /get_metrics│  │ /orchestrate│ │
│  │             │  │ /params     │  │ /health     │ │
│  │             │  │ /avail_ops  │  │ /config     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                      │
│  ┌─────────────┐                                    │
│  │  frontend   │  Shared volume: ./data:/app/data   │
│  │ Streamlit   │  Network: ml-network (bridge)      │
│  │ Port: 8502  │                                    │
│  └─────────────┘                                    │
└─────────────────────────────────────────────────────┘
```

## Architectural Decisions

### Double Serialization (MCP + HTTP)

Agent calls to Fedot.Industrial go through two serialization layers:
```
Agent → JSON-RPC (MCP stdio) → MCP Server → HTTP JSON → Fedot Server → FedotIndustrial
```

This is an accepted trade-off because:
- Fedot Server runs in a separate Docker container (heavy dependencies)
- MCP Server runs as subprocess in the backend container
- Without HTTP, we'd need to embed FedotIndustrial in the backend container (breaks isolation)
- Latency overhead is negligible compared to model training time (minutes)

### Tool Filtering per Agent

Each agent sees ONLY its tools via `ALLOWED_TOOLS`:
- Architect: 5 tools (profiling, baselines, config)
- Engineer: 3 tools (train, fedot, compare)
- Critic: 4 tools (errors, importance, explain, metrics)
- Scribe: 1 tool (report)

This prevents LLM confusion from seeing 15+ tools and improves tool selection accuracy.

### Cross-Iteration Memory

`DataContext.iteration_history` stores `IterationRecord` for each completed iteration:
- Best model and score
- Fedot config used
- Failed baselines
- Critic's winner and suggested changes

Architect receives this history in its prompt to avoid repeating failed approaches.

### Forced Feedback Merge

Orchestrator force-merges `CriticFeedback.suggested_fedot_changes` into Architect's FedotConfig
AFTER the Architect runs. This guarantees Critic feedback is applied even if the LLM ignores it.

### Stop Criteria

Two stop conditions:
1. Critic sets `should_stop=True` (score good enough)
2. Score-based: no improvement > 0.001 for 2 consecutive iterations
