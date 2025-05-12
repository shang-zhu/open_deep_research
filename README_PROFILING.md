# Profiling the Open Deep Research Agent

This document explains how to use the profiling features to analyze the performance of the Open Deep Research agent.

## How to Enable Profiling

You can enable profiling by passing the `--profile` flag when running the agent:

```bash
python src/together_open_deep_research.py --profile
```

By default, profiling data and visualizations will be saved to a directory called `profile_data`. You can specify a different directory using the `--profile-dir` option:

```bash
python src/together_open_deep_research.py --profile --profile-dir my_profiles
```

## Profiling Output

When profiling is enabled, the agent will:

1. Measure and record the execution time of each step in the research process
2. Generate visualizations of the profiling data
3. Print a summary of the timing information at the end of execution

### Generated Files

The following files will be created in the profile output directory:

1. `profile_data_{topic}.json` - Raw timing data in JSON format
2. `time_distribution_{topic}.png` - Pie chart showing the distribution of time across different steps
3. `step_durations_{topic}.png` - Bar chart showing the duration of each step
4. `iteration_performance_{topic}.png` - Line chart showing performance across iterations for iterative steps
5. `flame_graph_{topic}.png` - Flame graph visualization of the research process

### Console Output

At the end of execution, a summary of the profiling data will be printed to the console, including:

- Total execution time
- Breakdown of time by major steps, sorted by duration
- Percentage of total time spent in each step

## Understanding the Results

The profiling data can help identify performance bottlenecks in the research process:

1. **Time Distribution Chart**: Shows which steps consume the most time proportionally
2. **Step Durations Chart**: Shows the absolute time spent in each step
3. **Iteration Performance Chart**: Shows how the performance of iterative steps changes across iterations
4. **Flame Graph**: Visualizes the research process timeline grouped by operation type

### Flame Graph Visualization

The flame graph provides a visual representation of how time is spent during the research process:

- Operations are grouped by their main type (e.g., "initial", "evaluate", "filter", etc.)
- Each rectangle represents an individual operation
- The width of each rectangle represents the duration of the operation
- Operations of the same type share the same color 
- The height of each group corresponds to the relative time spent on that type of operation

This visualization helps identify:
- Which categories of operations consume the most time
- How time is distributed across different operation types
- The overall structure of the research process timeline

Common steps that may appear in the profiling data:

- `topic_clarification`: Time spent clarifying the research topic with the user
- `generate_initial_queries`: Time spent generating the initial set of research queries
- `initial_search`: Time spent performing the initial search
- `evaluate_research_iteration_X`: Time spent evaluating if more research is needed in iteration X
- `additional_search_iteration_X`: Time spent performing additional searches in iteration X
- `filter_results`: Time spent filtering and processing the search results
- `generate_answer_iteration_X`: Time spent generating the final answer in iteration X
- `total_execution_time`: Total time spent from start to finish

## Using Profiling for Optimization

The profiling data can be used to optimize the agent's performance:

1. Identify which steps take the most time
2. Consider using different models for steps that are particularly slow
3. Adjust the budget parameter to control the number of iterations
4. Modify the code to optimize slow steps
5. Experiment with different caching strategies for expensive operations

## Programmatic Usage

If you're using the DeepResearcher in your own code, you can enable profiling by passing the `profile_output_dir` parameter:

```python
from together_open_deep_research import DeepResearcher

researcher = DeepResearcher(
    budget=6,
    profile_output_dir="my_profiles"
)

result = researcher("What are the latest advances in quantum computing?")
```

The profiling data will be accessible through the `researcher.profiling_data` attribute. 