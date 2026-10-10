from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .graph import CurriculumGraph, Exam, POLICIES, StudentState, WeightPolicy


@dataclass
class PlanResult:
    algorithm: str
    selected_units: List[str]
    daily_schedule: List[List[str]]
    unassigned_units: Dict[str, str]
    total_cost: int
    total_value: int
    budget_limit: int
    dependency_violations: int = 0
    budget_violations: int = 0
    solver_status: str = "FEASIBLE"
    latency_ms: float = 0.0
    unit_costs: Dict[str, int] = field(default_factory=dict)
    unit_values: Dict[str, int] = field(default_factory=dict)

    @property
    def today_units(self) -> List[str]:
        if not self.daily_schedule:
            return []
        seen: List[str] = []
        for u in self.daily_schedule[0]:
            if u not in seen:
                seen.append(u)
        return seen


def allocate_daily_slots(
    cg: CurriculumGraph,
    ordered_units: List[str],
    daily_budget: List[int],
    state: StudentState,
    policy: WeightPolicy = POLICIES["Default"],
) -> List[List[str]]:
    """
    Allocates selected units (in topological order) across daily slot budgets.
    Units with cost > 1 (e.g., difficulty=3) can be split across multiple days,
    and a dependent unit only starts after all slots of its prerequisites are completed.
    """
    schedule: List[List[str]] = [[] for _ in daily_budget]
    if not ordered_units or not daily_budget:
        return schedule

    day_idx = 0
    for uid in ordered_units:
        slots_needed = cg.get_unit_cost(uid, state, policy)
        while slots_needed > 0 and day_idx < len(daily_budget):
            if len(schedule[day_idx]) < daily_budget[day_idx]:
                schedule[day_idx].append(uid)
                slots_needed -= 1
            else:
                day_idx += 1

    return schedule


def prune_orphan_zero_value_prereqs(
    cg: CurriculumGraph,
    selected_set: Set[str],
    targets: Set[str],
    state: StudentState,
    policy: WeightPolicy,
) -> Set[str]:
    """
    Removes any selected unit that has 0 value (or is not a target and has 0 value)
    and has no selected dependent requiring it.
    """
    pruned = set(selected_set)
    changed = True
    while changed:
        changed = False
        for uid in list(pruned):
            val = cg.get_unit_value(uid, state, policy)
            has_selected_dependent = any(
                succ in pruned for succ in cg.graph.successors(uid)
            )
            if val <= 0 and not has_selected_dependent:
                pruned.remove(uid)
                changed = True
    return pruned


def build_unassigned_reasons(
    cg: CurriculumGraph,
    closure_units: List[str],
    selected_set: Set[str],
    state: StudentState,
    blocked_units: Optional[Set[str]] = None,
) -> Dict[str, str]:
    reasons: Dict[str, str] = {}
    blocked = blocked_units or set()
    for uid in closure_units:
        if uid in selected_set:
            continue
        if uid in blocked:
            reasons[uid] = "선수 단원 평가 미통과로 인한 학습 시작 보류"
            continue
        missing_reqs = [
            r for r in cg.units[uid].requires if r not in state.mastered and r not in selected_set
        ]
        if missing_reqs:
            reasons[uid] = f"선수 조건({', '.join(missing_reqs)}) 미배정 및 예산 부족"
        else:
            reasons[uid] = "가용 학습 예산 대비 중요도/가치 후순위로 이번 계획에서 제외"
    return reasons


class TopologicalPlanner:
    """
    Algorithm A (Baseline):
    Iterates through the prerequisite closure in topological order and selects units
    sequentially as long as prerequisites are met and budget remains.
    """

    def plan(
        self,
        cg: CurriculumGraph,
        exam: Exam,
        state: StudentState,
        policy: WeightPolicy = POLICIES["Default"],
        blocked_units: Optional[Set[str]] = None,
    ) -> PlanResult:
        t0 = time.perf_counter()
        blocked = blocked_units or set()
        closure = cg.get_required_closure(exam.targets, state.mastered)

        budget_rem = exam.total_budget
        selected_set: Set[str] = set()

        for uid in closure:
            if uid in blocked:
                continue
            u = cg.units[uid]
            # Check all prerequisites are either mastered or already selected
            if not all(r in state.mastered or r in selected_set for r in u.requires):
                continue
            cost = cg.get_unit_cost(uid, state, policy)
            if cost <= budget_rem:
                selected_set.add(uid)
                budget_rem -= cost

        selected_set = prune_orphan_zero_value_prereqs(
            cg, selected_set, set(exam.targets), state, policy
        )
        ordered = cg.topological_order(selected_set)
        daily_schedule = allocate_daily_slots(
            cg, ordered, exam.daily_budget, state, policy
        )
        audit = cg.verify_plan(ordered, daily_schedule, exam.daily_budget, state, policy)

        total_cost = sum(cg.get_unit_cost(u, state, policy) for u in ordered)
        total_value = sum(cg.get_unit_value(u, state, policy) for u in ordered)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return PlanResult(
            algorithm="Topological",
            selected_units=ordered,
            daily_schedule=daily_schedule,
            unassigned_units=build_unassigned_reasons(
                cg, closure, selected_set, state, blocked
            ),
            total_cost=total_cost,
            total_value=total_value,
            budget_limit=exam.total_budget,
            dependency_violations=int(audit["dependency_violations"]),
            budget_violations=int(audit["budget_violations"]),
            solver_status="FEASIBLE",
            latency_ms=latency_ms,
            unit_costs={u: cg.get_unit_cost(u, state, policy) for u in ordered},
            unit_values={u: cg.get_unit_value(u, state, policy) for u in ordered},
        )
