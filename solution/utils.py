import os
import re
import datetime
from typing import Dict, Any


def extract_code_from_response(response_text: str) -> str:
    if not response_text:
        return ""

    code_block_pattern = r'```(?:python)?\s*(.*?)\s*```'
    match = re.search(code_block_pattern, response_text, re.DOTALL)
    return match.group(1).strip() if match else response_text.strip()


def sanitise_filename(text: str) -> str:
    sanitised = re.sub(r'[^\w\s-]', '', text).strip()
    return re.sub(r'[-\s]+', '_', sanitised).lower()


class CodebaseGenerator:
    def __init__(self, pattern_name: str, task: str):
        self.pattern_name = pattern_name
        self.task = task
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder_name = f"generated/{pattern_name}_{self.timestamp}"

    def create_folder(self) -> str:
        os.makedirs(self.folder_name, exist_ok=True)
        return self.folder_name

    def write_python_file(self, filename: str, content: str) -> None:
        code = extract_code_from_response(content)
        if code:
            filepath = os.path.join(self.folder_name, f"{filename}.py")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)

    def write_text_file(self, filename: str, content: str) -> None:
        filepath = os.path.join(self.folder_name, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)


class SequentialCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        self.write_python_file("original_code", result.get('code', ''))
        self.write_python_file(
            "refactored_code", result.get('refactored_code', ''))

        if result.get('unit_tests'):
            self.write_python_file("unit_tests", result.get('unit_tests', ''))

        unit_tests_section = ""
        if result.get('unit_tests'):
            unit_tests_section = f"""

## Unit Tests
```python
{extract_code_from_response(result.get('unit_tests', 'No unit tests generated'))}
```"""

        files_generated = "- `original_code.py` - Initial implementation\n- `refactored_code.py` - Improved version based on review"
        if result.get('unit_tests'):
            files_generated += "\n- `unit_tests.py` - Comprehensive test suite"

        performance_section = ""
        if 'performance_metrics' in str(result):
            performance_section = f"""

## Performance Metrics
Execution timing analysis available in debug output."""

        audit_content = f"""# Sequential Workflow Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Sequential Workflow

## Original Code
```python
{extract_code_from_response(result.get('code', 'No code generated'))}
```

## Review Feedback
{result.get('review', 'No review available')}

## Refactored Code
```python
{extract_code_from_response(result.get('refactored_code', 'No refactored code available'))}
```{unit_tests_section}{performance_section}

## Files Generated
{files_generated}

---
*Generated using LangGraph Sequential Workflow Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(f"✅ Sequential codebase created in: {self.folder_name}/")


class ConditionalCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        code_list = result.get('code', [''])
        if isinstance(code_list, str):
            code_list = [code_list]

        final_code = code_list[-1] if code_list else ''
        self.write_python_file("final_code", final_code)

        previous_iterations_section = ""
        if len(code_list) > 1:
            previous_iterations_section = "\n## Previous Iterations\n\n"
            for i, code_version in enumerate(code_list[:-1]):
                iteration_label = "Original Code" if i == 0 else f"Iteration {i}"
                previous_iterations_section += f"""### {iteration_label}
```python
{extract_code_from_response(code_version)}
```

"""

        quality_metrics_section = self._build_quality_metrics_section(result)

        best_code_section = ""
        if result.get('best_code_index') is not None and result.get('best_lowest_score') is not None:
            best_code_section = f"""

## Best Code Selection
- **Best iteration:** {result['best_code_index'] + 1}
- **Best score:** {result['best_lowest_score']}/10
- **Selection method:** Highest scoring version across all iterations"""

        fast_track_section = ""
        if result.get('iteration_count', 0) == 0 and result.get('quality_score', 0) >= 9:
            fast_track_section = f"""

## Fast Track Activation
Initial code scored {result['quality_score']}/10 - skipped refactoring entirely."""

        audit_content = f"""# Conditional Routing Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Conditional Routing

## Final Code
```python
{extract_code_from_response(final_code)}
```

## Review Feedback
{result.get('review', 'No review available')}

{quality_metrics_section}{best_code_section}{fast_track_section}{previous_iterations_section}
## Files Generated
- `final_code.py` - Quality-approved implementation

---
*Generated using LangGraph Conditional Routing Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(
            f"✅ Conditional routing codebase created in: {self.folder_name}/")

    def _build_quality_metrics_section(self, result: Dict[str, Any]) -> str:
        if all(key in result for key in ['security_score', 'performance_score', 'readability_score']):
            security = result['security_score']
            performance = result['performance_score']
            readability = result['readability_score']
            lowest = result.get('lowest_score', min(
                security, performance, readability))

            def score_bar(score: int) -> str:
                filled = "█" * score
                empty = "░" * (10 - score)
                return f"{filled}{empty} ({score}/10)"

            return f"""## Quality Metrics (Multi-Criteria Evaluation)

| Criterion    | Score | Visual                    | Status |
|--------------|-------|---------------------------|--------|
| Security     | {security}/10  | {score_bar(security)}     | {'✅ Pass' if security >= 7 else '❌ Fail'} |
| Performance  | {performance}/10  | {score_bar(performance)}  | {'✅ Pass' if performance >= 7 else '❌ Fail'} |
| Readability  | {readability}/10  | {score_bar(readability)}  | {'✅ Pass' if readability >= 7 else '❌ Fail'} |
| **Overall**  | **{lowest}/10** | **{score_bar(lowest)}** | **{'✅ All Pass' if lowest >= 7 else '❌ Needs Work'}** |

**Evaluation Method:** Lowest score determines overall quality  
**Iterations:** {result.get('iteration_count', 0)}  
**Threshold:** 7/10 minimum for all criteria  

"""
        else:
            quality_score = result.get('quality_score', 'N/A')
            iteration_count = result.get('iteration_count', 0)

            return f"""## Quality Metrics
- **Score:** {quality_score}/10
- **Iterations:** {iteration_count}
- **Threshold:** 7/10

"""


class ParallelCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        self.write_python_file("main_code", result.get('code', ''))

        performance_comparison = ""
        if result.get('sequential_time') and result.get('parallel_time'):
            seq_time = result['sequential_time']
            par_time = result['parallel_time']
            speedup = seq_time / par_time if par_time > 0 else 0
            performance_comparison = f"""

## Performance Analysis
- **Sequential execution:** {seq_time:.2f}s
- **Parallel execution:** {par_time:.2f}s  
- **Speedup achieved:** {speedup:.2f}x"""

        synthesis_content = f"""# Code Analysis Synthesis Report

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Analysis Method:** Parallel Expert Review{performance_comparison}

## Executive Summary

{result.get('final_report', 'No synthesis report available')}

---
*Generated using LangGraph Parallel Processing Pattern*
"""
        self.write_text_file("SYNTHESIS_REPORT.md", synthesis_content)

        documentation_section = ""
        if result.get('documentation_analysis'):
            documentation_section = f"""

### Documentation Analysis
{result.get('documentation_analysis', 'No documentation analysis available')}"""

        error_handling_section = ""
        if any("failed" in str(result.get(key, "")) for key in ['security_analysis', 'performance_analysis', 'style_analysis', 'documentation_analysis']):
            error_handling_section = f"""

## Error Handling
Some agents encountered errors during execution but the workflow continued gracefully."""

        audit_content = f"""# Parallel Processing Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Parallel Processing

## Generated Code
```python
{extract_code_from_response(result.get('code', 'No code generated'))}
```

## Expert Analysis Reports

### Security Analysis
{result.get('security_analysis', 'No security analysis available')}

### Performance Analysis
{result.get('performance_analysis', 'No performance analysis available')}

### Style Analysis
{result.get('style_analysis', 'No style analysis available')}{documentation_section}{error_handling_section}

## Files Generated
- `main_code.py` - Analysed implementation
- `SYNTHESIS_REPORT.md` - **KEY DELIVERABLE:** Aggregated expert recommendations

---
*Generated using LangGraph Parallel Processing Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(
            f"✅ Parallel processing codebase created in: {self.folder_name}/")
        print(f"📊 Key deliverable: {self.folder_name}/SYNTHESIS_REPORT.md")


class SupervisorCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        self.write_python_file("main_code", result.get('code', ''))

        task_analysis_section = ""
        if result.get('task_type'):
            task_analysis_section = f"""

## Task Analysis
- **Task Type:** {result['task_type']}
- **Routing Strategy:** {'Priority security routing' if result['task_type'] == 'authentication' else 'Standard expert routing'}"""

        final_analysis_content = f"""# Expert Analysis & Recommendations

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Analysis Method:** Supervised Expert Consultation{task_analysis_section}

## Executive Summary

{result.get('final_analysis', 'No final analysis available')}

## Expert Consultation Process

**Agents Consulted:** {', '.join(result.get('completed_agents', []))}

### Supervisor Decision Log
{result.get('supervisor_notes', 'No supervisor decisions recorded')}

---
*Generated using LangGraph Supervisor Agents Pattern*
"""
        self.write_text_file("EXPERT_ANALYSIS.md", final_analysis_content)

        completed_agents = result.get('completed_agents', [])
        reports_section = ""

        if result.get('security_report'):
            context_note = " (with quality context)" if result.get(
                'quality_report') else ""
            reports_section += f"### Security Expert Report{context_note}\n{result['security_report']}\n\n"
        if result.get('quality_report'):
            reports_section += f"### Quality Expert Report\n{result['quality_report']}\n\n"
        if result.get('database_report'):
            reports_section += f"### Database Expert Report\n{result['database_report']}\n\n"

        supervisor_notes = "Supervisor coordinated expert consultation based on task analysis and code content."
        if result.get('task_type') == 'authentication':
            supervisor_notes += " Priority routing applied for authentication task - security expert consulted first."
        if result.get('database_report'):
            supervisor_notes += " Database expert added based on code analysis showing SQL/database operations."

        smart_routing_section = ""
        if result.get('database_report') or result.get('task_type') == 'authentication':
            smart_routing_section = f"""

## Smart Routing Features
- **Content-based routing:** Database expert consulted based on code analysis
- **Task-type prioritisation:** {'Security-first routing for authentication tasks' if result.get('task_type') == 'authentication' else 'Standard routing applied'}
- **Expert collaboration:** Security expert reviewed quality findings"""

        audit_content = f"""# Supervisor Agents Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Supervisor Agents
**Agents Consulted:** {', '.join(completed_agents)}

## Generated Code
```python
{extract_code_from_response(result.get('code', 'No code generated'))}
```

## Supervisor Decision Process
{supervisor_notes}

## Individual Expert Reports

{reports_section}{smart_routing_section}

## Files Generated
- `main_code.py` - Expert-reviewed implementation  
- `EXPERT_ANALYSIS.md` - **KEY DELIVERABLE:** Synthesised expert recommendations

---
*Generated using LangGraph Supervisor Agents Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(f"✅ Supervisor agents codebase created in: {self.folder_name}/")
        print(f"🎯 Key deliverable: {self.folder_name}/EXPERT_ANALYSIS.md")


class EvaluatorCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        self.write_python_file("final_code", result.get(
            'final_code', result.get('code', '')))

        current_eval = result.get('current_evaluation', {})
        final_score = current_eval.get('quality_score', 'N/A')
        complexity_score = current_eval.get('complexity_score', 'N/A')
        final_feedback = current_eval.get('feedback', 'No feedback available')
        iteration_count = result.get('iteration_count', 0)
        plateau_count = result.get('plateau_count', 0)

        complexity_section = ""
        if complexity_score != 'N/A':
            complexity_section = f"""
- **Complexity Score:** {complexity_score}/10 (10 = simple)
- **Combined Score:** {(final_score + complexity_score) / 2 if isinstance(final_score, int) and isinstance(complexity_score, int) else 'N/A'}/10"""

        optimisation_features = ""
        optimization_methods = []
        if result.get('performance_focused'):
            optimization_methods.append("Performance-targeted optimisation")
        if plateau_count > 0:
            optimization_methods.append("Plateau detection enabled")
        if result.get('history'):
            optimization_methods.append("Progress chart generation")

        if optimization_methods:
            optimisation_features = f"""

## Advanced Optimisation Features
{chr(10).join(f"- {method}" for method in optimization_methods)}"""

        history_section = f"""## Optimisation Summary
- **Total Iterations:** {iteration_count}
- **Final Quality Score:** {final_score}/10{complexity_section}
- **Plateau Count:** {plateau_count}
- **Final Feedback:** {final_feedback}
- **Completion Reason:** {'Quality threshold reached' if isinstance(final_score, int) and final_score >= 8 else 'Plateau detected' if plateau_count >= 2 else 'Max iterations reached' if iteration_count >= 3 else 'Evaluator determined completion'}

"""

        chart_note = ""
        if result.get('history') and len(result['history']) > 1:
            chart_note = "\n📊 **Progress chart saved as `optimisation_progress.png`**"

        audit_content = f"""# Evaluator-Optimiser Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Evaluator-Optimiser
**Total Iterations:** {iteration_count}
**Final Score:** {final_score}/10

## Final Code
```python
{extract_code_from_response(result.get('final_code', result.get('code', 'No code generated')))}
```

{history_section}{optimisation_features}

## Files Generated
- `final_code.py` - Iteratively optimised implementation{chart_note}

---
*Generated using LangGraph Evaluator-Optimiser Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(
            f"✅ Evaluator-optimiser codebase created in: {self.folder_name}/")


class OrchestratorCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        self.write_python_file("final_code", result.get('final_result', ''))

        subtasks_section = ""
        if result.get('subtasks'):
            subtasks_section = "\n## Task Breakdown\n\n"
            for i, subtask in enumerate(result['subtasks'], 1):
                if isinstance(subtask, dict):
                    deps = ", ".join(subtask.get('dependencies', [])) if subtask.get(
                        'dependencies') else "None"
                    priority = subtask.get('priority', 'N/A')
                    subtasks_section += f"""### Subtask {i}: {subtask.get('name', f'Task {i}')}
**Type:** {subtask.get('type', 'Unknown')}  
**Priority:** {priority}  
**Dependencies:** {deps}  
**Description:** {subtask.get('description', 'No description')}

"""
                else:
                    subtasks_section += f"""### Subtask {i}
{subtask}

"""

        worker_specialisation_section = ""
        worker_types = set()
        if result.get('worker_outputs'):
            for output in result['worker_outputs']:
                if output.startswith('FRONTEND'):
                    worker_types.add('Frontend')
                elif output.startswith('BACKEND'):
                    worker_types.add('Backend')
                elif output.startswith('DATABASE'):
                    worker_types.add('Database')
                elif output.startswith('TESTING'):
                    worker_types.add('Testing')

        if worker_types:
            worker_specialisation_section = f"""

## Worker Specialisation
**Specialised workers used:** {', '.join(sorted(worker_types))}"""

        dependency_handling_section = ""
        if result.get('subtasks') and any(subtask.get('dependencies') for subtask in result.get('subtasks', [])):
            dependency_handling_section = f"""

## Dependency Management
**Dependency-aware execution:** Subtasks executed in correct order based on dependencies"""

        validation_section = ""
        if result.get('validation_result'):
            validation = result['validation_result']
            validation_section = f"""

## Integration Validation
- **Can combine:** {validation.get('can_combine', 'Unknown')}
- **Issues found:** {len(validation.get('issues', []))}
- **Suggestions:** {len(validation.get('suggestions', []))}"""

        worker_outputs_section = ""
        if result.get('worker_outputs'):
            worker_outputs_section = "\n## Worker Outputs\n\n"
            for i, output in enumerate(result['worker_outputs'], 1):
                worker_outputs_section += f"""### Worker {i} Output
```python
{extract_code_from_response(output)}
```

"""

        orchestrator_report = f"""# Orchestrator Process Report

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Analysis Method:** Dynamic Task Decomposition{worker_specialisation_section}{dependency_handling_section}{validation_section}

## Executive Summary

The orchestrator successfully broke down the complex task into {len(result.get('subtasks', []))} manageable subtasks, executed them through specialised workers, and synthesised the results into a cohesive solution.

## Process Overview

1. **Task Analysis**: Orchestrator analysed the input requirements
2. **Dynamic Decomposition**: Created {len(result.get('subtasks', []))} specialised subtasks
3. **Dependency Resolution**: Executed subtasks in correct order
4. **Specialised Execution**: Workers processed subtasks independently
5. **Integration Validation**: Checked compatibility before synthesis
6. **Result Synthesis**: Combined worker outputs into final solution

{subtasks_section}

---
*Generated using LangGraph Orchestrator-Worker Pattern*
"""
        self.write_text_file("ORCHESTRATOR_REPORT.md", orchestrator_report)

        audit_content = f"""# Orchestrator-Worker Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Orchestrator-Worker
**Subtasks Created:** {len(result.get('subtasks', []))}
**Workers Executed:** {len(result.get('worker_outputs', []))}

## Final Code
```python
{extract_code_from_response(result.get('final_result', 'No code generated'))}
```

{subtasks_section}{worker_outputs_section}## Files Generated
- `final_code.py` - Synthesised final implementation
- `ORCHESTRATOR_REPORT.md` - **KEY DELIVERABLE:** Orchestration process breakdown

---
*Generated using LangGraph Orchestrator-Worker Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(
            f"✅ Orchestrator-worker codebase created in: {self.folder_name}/")
        print(f"🎯 Key deliverable: {self.folder_name}/ORCHESTRATOR_REPORT.md")


class ProductionCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        self.write_python_file("production_code", result.get(
            'refactored_code', result.get('code', '')))

        metrics_section = f"""## Production Metrics
- **Session ID:** {result.get('session_id', 'N/A')}
- **Execution Time:** {result.get('execution_time', 0):.2f}s
- **Retry Count:** {result.get('retry_count', 0)}
- **Human Approval Required:** {result.get('human_approval_needed', False)}
"""

        error_section = ""
        if result.get('error_log'):
            error_section = f"""## Error Log
```
{result['error_log']}
```

"""

        audit_content = f"""# Production Ready Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Production Ready

## Production Code
```python
{extract_code_from_response(result.get('refactored_code', result.get('code', 'No code generated')))}
```

{metrics_section}

## Review Feedback
{result.get('review', 'No review available')}

{error_section}## Files Generated
- `production_code.py` - Production-ready implementation

---
*Generated using LangGraph Production Ready Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(f"✅ Production ready codebase created in: {self.folder_name}/")
