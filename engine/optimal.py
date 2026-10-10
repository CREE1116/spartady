from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

from .graph import CurriculumGraph, Exam, POLICIES, StudentState, WeightPolicy
from .planner import (
    PlanResult,
    allocate_daily_slots,
    build_unassigned_reasons,
    prune_orphan_zero_value_prereqs,
)


class OptimalPlanner:
    """
    Algorithm C (Exact Optimization via OR-Tools CP-SAT):
    Maximizes sum(w_i * x_i) subject to:
      1) sum(c_i * x_i) <= B
      2) x_i <= x_j + m_j  for all prerequisites j of i
    """

    def __init__(self, time_limit_seconds: float = 5.0) -> None:
        self.time_limit_seconds = time_limit_seconds

    def plan(
        self,
        cg: CurriculumGraph,
        exam: Exam,
        state: StudentState,
        policy: WeightPolicy = POLICIES["Default"],
        blocked_units: Optional[Set[str]] = None,
        previous_selected: Optional[Set[str]] = None,
    ) -> PlanResult:
        t0 = time.perf_counter()
        blocked = blocked_units or set()
        closure = cg.get_required_closure(exam.targets, state.mastered)

        model = cp_model.CpModel()
        x: Dict[str, cp_model.IntVar] = {
            uid: model.NewBoolVar(f"x_{uid}") for uid in closure
        }

        # Blocked units constraint
        for uid in closure:
            if uid in blocked:
                model.Add(x[uid] == 0)

        # Prerequisite constraint: x_i <= x_j + m_j
        for uid in closure:
            u = cg.units[uid]
            for req_id in u.requires:
                m_j = 1 if req_id in state.mastered else 0
                if m_j == 0:
                    if req_id in x:
                        model.Add(x[uid] <= x[req_id])
                    else:
                        model.Add(x[uid] == 0)

        # Budget constraint: sum(c_i * x_i) <= B
        costs = {uid: cg.get_unit_cost(uid, state, policy) for uid in closure}
        values = {uid: cg.get_unit_value(uid, state, policy) for uid in closure}

        model.Add(sum(costs[uid] * x[uid] for uid in closure) <= exam.total_budget)

        # Objective: Primary = maximize sum(w_i * x_i).
        # Secondary tie-breaker = avoid picking 0-value orphan prerequisites and minimize cost on ties
        scale = 1000
        prev_set = previous_selected or set()
        obj_terms = []
        for uid in closure:
            stability_bonus = 1 if uid in prev_set else 0
            cost_penalty = costs[uid] if len(closure) <= 60 else (1 if values[uid] == 0 else 0)
            coeff = values[uid] * scale - cost_penalty + stability_bonus
            obj_terms.append(coeff * x[uid])

        model.Maximize(sum(obj_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.num_search_workers = 1  # Deterministic single-thread execution
        status_code = solver.Solve(model)
        status_name = solver.StatusName(status_code)

        selected_set: Set[str] = set()
        if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for uid in closure:
                if solver.BooleanValue(x[uid]):
                    selected_set.add(uid)

        selected_set = prune_orphan_zero_value_prereqs(
            cg, selected_set, set(exam.targets), state, policy
        )
        ordered = cg.topological_order(selected_set)
        daily_schedule = allocate_daily_slots(
            cg, ordered, exam.daily_budget, state, policy
        )
        audit = cg.verify_plan(ordered, daily_schedule, exam.daily_budget, state, policy)

        total_cost = sum(costs[u] for u in ordered)
        total_value = sum(values[u] for u in ordered)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return PlanResult(
            algorithm="Optimal",
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
            solver_status=status_name,
            latency_ms=latency_ms,
            unit_costs={u: costs[u] for u in ordered},
            unit_values={u: values[u] for u in ordered},
        )


class ExhaustiveOptimalPlanner:
    """
    Brute-force 2^N exact subset enumerator for small graphs (N <= 18)
    to independently verify the CP-SAT solver formulation (Section 4).
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
        if len(closure) > 20:
            raise ValueError(
                f"Exhaustive enumeration only intended for N <= 20 (got {len(closure)})"
            )

        costs = {uid: cg.get_unit_cost(uid, state, policy) for uid in closure}
        values = {uid: cg.get_unit_value(uid, state, policy) for uid in closure}

        best_subset: Set[str] = set()
        best_key: Tuple[int, int] = (-1, 0)  # (total_value, -total_cost)

        for r in range(len(closure) + 1):
            for combo in itertools.combinations(closure, r):
                cand = set(combo)
                if cand & blocked:
                    continue
                total_c = sum(costs[u] for u in cand)
                if total_c > exam.total_budget:
                    continue

                # Check prerequisite validity
                valid = True
                for u in cand:
                    for req in cg.units[u].requires:
                        if req not in state.mastered and req not in cand:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue

                total_v = sum(values[u] for u in cand)
                cand_key = (total_v, -total_c)
                if cand_key > best_key:
                    best_key = cand_key
                    best_subset = cand

        ordered = cg.topological_order(best_subset)
        daily_schedule = allocate_daily_slots(
            cg, ordered, exam.daily_budget, state, policy
        )
        audit = cg.verify_plan(ordered, daily_schedule, exam.daily_budget, state, policy)

        total_cost = sum(costs[u] for u in ordered)
        total_value = sum(values[u] for u in ordered)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return PlanResult(
            algorithm="Exhaustive",
            selected_units=ordered,
            daily_schedule=daily_schedule,
            unassigned_units=build_unassigned_reasons(
                cg, closure, best_subset, state, blocked
            ),
            total_cost=total_cost,
            total_value=total_value,
            budget_limit=exam.total_budget,
            dependency_violations=int(audit["dependency_violations"]),
            budget_violations=int(audit["budget_violations"]),
            solver_status="EXACT_ENUMERATION",
            latency_ms=latency_ms,
            unit_costs={u: costs[u] for u in ordered},
            unit_values={u: values[u] for u in ordered},
        )
