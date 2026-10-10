from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx


@dataclass(frozen=True)
class Unit:
    id: str
    difficulty: int  # 하=1, 중=2, 상=3
    importance: int  # 선수전용=0, 하=1, 중=2, 상=3
    requires: Tuple[str, ...] = field(default_factory=tuple)
    title: str = ""
    explicit_cost: Optional[int] = None
    explicit_value: Optional[int] = None


@dataclass
class StudentState:
    mastered: Set[str] = field(default_factory=set)
    partial: Dict[str, int] = field(default_factory=dict)  # unit_id -> completed slots
    failed: Set[str] = field(default_factory=set)

    def copy(self) -> "StudentState":
        return StudentState(
            mastered=set(self.mastered),
            partial=dict(self.partial),
            failed=set(self.failed),
        )


@dataclass
class Exam:
    targets: List[str]
    daily_budget: List[int]

    @property
    def total_budget(self) -> int:
        return sum(self.daily_budget)


@dataclass(frozen=True)
class WeightPolicy:
    name: str
    difficulty_map: Dict[int, int]
    importance_map: Dict[int, int]
    use_explicit_if_present: bool = True


POLICIES: Dict[str, WeightPolicy] = {
    "Linear": WeightPolicy(
        name="Linear",
        difficulty_map={1: 1, 2: 2, 3: 3},
        importance_map={0: 0, 1: 1, 2: 2, 3: 3},
        use_explicit_if_present=False,
    ),
    "Steep": WeightPolicy(
        name="Steep",
        difficulty_map={1: 1, 2: 2, 3: 4},
        importance_map={0: 0, 1: 1, 2: 3, 3: 6},
        use_explicit_if_present=False,
    ),
    "Flat": WeightPolicy(
        name="Flat",
        difficulty_map={1: 1, 2: 2, 3: 3},
        importance_map={0: 0, 1: 1, 2: 1, 3: 1},
        use_explicit_if_present=False,
    ),
    "Priority-first": WeightPolicy(
        name="Priority-first",
        difficulty_map={1: 1, 2: 2, 3: 3},
        # Lexicographic tier weighting: importance=3 dominates importance=2, which dominates importance=1
        importance_map={0: 0, 1: 1, 2: 20, 3: 400},
        use_explicit_if_present=False,
    ),
    "Default": WeightPolicy(
        name="Default",
        difficulty_map={1: 1, 2: 2, 3: 3},
        importance_map={0: 0, 1: 1, 2: 2, 3: 3},
        use_explicit_if_present=True,
    ),
}


class CyclicDependencyError(ValueError):
    """Raised when the curriculum graph contains a cycle in REQUIRES relations."""


class CurriculumGraph:
    """Directed Acyclic Graph (DAG) representation of Curriculum Units using NetworkX."""

    def __init__(self, units: Iterable[Unit], name: str = "curriculum_graph") -> None:
        self.name = name
        self.units: Dict[str, Unit] = {}
        self.graph = nx.DiGraph()

        for u in units:
            self.units[u.id] = u
            self.graph.add_node(u.id, unit=u)

        for u in self.units.values():
            for req_id in u.requires:
                if req_id not in self.units:
                    raise KeyError(f"Unit '{u.id}' requires unknown unit '{req_id}'")
                # Edge direction: prerequisite (req_id) -> dependent (u.id)
                self.graph.add_edge(req_id, u.id)

        self.validate_dag()

    @classmethod
    def from_json(cls, path: str | Path) -> Tuple["CurriculumGraph", List[str]]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        units = [
            Unit(
                id=item["id"],
                title=item.get("title", item["id"]),
                difficulty=int(item["difficulty"]),
                importance=int(item["importance"]),
                requires=tuple(item.get("requires", [])),
                explicit_cost=item.get("cost"),
                explicit_value=item.get("value"),
            )
            for item in data["units"]
        ]
        default_targets = data.get("default_targets", [u.id for u in units])
        return cls(units, name=data.get("name", Path(path).stem)), default_targets

    def validate_dag(self) -> None:
        if not nx.is_directed_acyclic_graph(self.graph):
            cycle = nx.find_cycle(self.graph)
            raise CyclicDependencyError(f"Cycle detected in curriculum graph: {cycle}")

    def get_ancestors(self, unit_id: str) -> Set[str]:
        """Returns all transitive prerequisites of unit_id."""
        return nx.ancestors(self.graph, unit_id)

    def get_descendants(self, unit_id: str) -> Set[str]:
        """Returns all transitive dependents that require unit_id."""
        return nx.descendants(self.graph, unit_id)

    def get_required_closure(
        self, target_ids: Iterable[str], mastered: Optional[Set[str]] = None
    ) -> List[str]:
        """Returns all unmastered units required to reach target_ids (including targets), in topological order."""
        mastered_set = mastered or set()
        closure: Set[str] = set()
        pending = list(target_ids)
        while pending:
            uid = pending.pop()
            if uid not in self.units:
                raise KeyError(f"Unknown target unit '{uid}'")
            if uid in mastered_set or uid in closure:
                continue
            closure.add(uid)
            pending.extend(self.units[uid].requires)
        return self.topological_order(closure)

    def topological_order(self, subset_ids: Optional[Iterable[str]] = None) -> List[str]:
        """Deterministic topological sort using unit ID as lexicographical tie-breaker."""
        if subset_ids is None:
            sub = self.graph
        else:
            sub = self.graph.subgraph(set(subset_ids))
        return list(nx.lexicographical_topological_sort(sub))

    def get_unit_cost(
        self,
        unit_id: str,
        state: Optional[StudentState] = None,
        policy: WeightPolicy = POLICIES["Default"],
    ) -> int:
        u = self.units[unit_id]
        if state and unit_id in state.mastered:
            return 0
        if policy.use_explicit_if_present and u.explicit_cost is not None:
            base_cost = int(u.explicit_cost)
        else:
            base_cost = policy.difficulty_map.get(u.difficulty, u.difficulty)

        if state and unit_id in state.partial and unit_id not in state.failed:
            done = state.partial[unit_id]
            return max(1, base_cost - done)
        return base_cost

    def get_unit_value(
        self,
        unit_id: str,
        state: Optional[StudentState] = None,
        policy: WeightPolicy = POLICIES["Default"],
    ) -> int:
        u = self.units[unit_id]
        if state and unit_id in state.mastered:
            return 0
        if policy.use_explicit_if_present and u.explicit_value is not None:
            return int(u.explicit_value)
        return policy.importance_map.get(u.importance, u.importance)

    def verify_plan(
        self,
        selected_order: List[str],
        daily_schedule: List[List[str]],
        daily_budget: List[int],
        state: StudentState,
        policy: WeightPolicy = POLICIES["Default"],
    ) -> Dict[str, object]:
        """Audits a generated plan against H1 (Dependency Violation) and H2 (Budget Violation)."""
        dep_violations: List[str] = []
        budget_violations: List[str] = []

        selected_set = set(selected_order)
        seen_in_order: Set[str] = set(state.mastered)

        # 1. Check prerequisite closure & ordering in selected_order
        for uid in selected_order:
            u = self.units[uid]
            for req in u.requires:
                if req not in seen_in_order:
                    dep_violations.append(
                        f"Order/Closure violation: '{uid}' requires '{req}', which is neither mastered nor precedes '{uid}'."
                    )
            seen_in_order.add(uid)

        # 2. Check total budget and daily slot budgets
        total_cost = sum(self.get_unit_cost(uid, state, policy) for uid in selected_set)
        total_budget = sum(daily_budget)
        if total_cost > total_budget:
            budget_violations.append(
                f"Total cost ({total_cost}) exceeds total budget ({total_budget})."
            )

        for day_idx, day_slots in enumerate(daily_schedule):
            cap = daily_budget[day_idx] if day_idx < len(daily_budget) else 0
            if len(day_slots) > cap:
                budget_violations.append(
                    f"Day {day_idx + 1} allocated {len(day_slots)} slots, exceeding daily budget {cap}."
                )

        # 3. Check day-by-day slot prerequisite completion
        completed_slots: Dict[str, int] = {}
        available_mastered: Set[str] = set(state.mastered)

        for day_idx, day_slots in enumerate(daily_schedule):
            for uid in day_slots:
                u = self.units[uid]
                for req in u.requires:
                    if req not in available_mastered:
                        dep_violations.append(
                            f"Schedule violation on Day {day_idx + 1}: '{uid}' scheduled before prerequisite '{req}' finished."
                        )
                completed_slots[uid] = completed_slots.get(uid, 0) + 1
                if completed_slots[uid] >= self.get_unit_cost(uid, state, policy):
                    available_mastered.add(uid)

        return {
            "dependency_violations": len(dep_violations),
            "budget_violations": len(budget_violations),
            "dep_violation_details": dep_violations,
            "budget_violation_details": budget_violations,
            "is_valid": len(dep_violations) == 0 and len(budget_violations) == 0,
        }


def generate_synthetic_graph(
    archetype: str,
    num_units: int,
    seed: int = 42,
) -> Tuple[CurriculumGraph, List[str], List[int]]:
    """Generates reproducible benchmark graphs for Section 6 (Experiment D)."""
    rng = random.Random(seed)
    units: List[Unit] = []
    ids = [f"U{i:04d}" if num_units >= 1000 else f"U{i:03d}" for i in range(1, num_units + 1)]

    for idx, uid in enumerate(ids):
        diff = rng.choice([1, 2, 3])
        imp = rng.choice([1, 2, 3])
        reqs: List[str] = []

        if archetype == "linear":
            if idx > 0:
                reqs = [ids[idx - 1]]
        elif archetype == "parallel":
            reqs = []
        elif archetype == "shared_prereq":
            # First 15% are foundational hubs; remaining depend on 1~2 hubs
            num_hubs = max(1, num_units // 7)
            if idx < num_hubs:
                imp = 1
                diff = 1
                reqs = []
            else:
                reqs = rng.sample(ids[:num_hubs], k=min(2, num_hubs))
        elif archetype == "deep":
            # Multi-layer chain of width 3
            width = 3
            layer = idx // width
            if layer > 0:
                prev_layer_start = (layer - 1) * width
                prev_candidates = ids[prev_layer_start : prev_layer_start + width]
                reqs = rng.sample(prev_candidates, k=min(2, len(prev_candidates)))
                if layer >= (num_units // width) - 2:
                    imp = 3
        elif archetype == "wide":
            # Many independent branches of depth 2~3
            branch_len = 3
            pos_in_branch = idx % branch_len
            if pos_in_branch > 0:
                reqs = [ids[idx - 1]]
        elif archetype == "tight_budget":
            # General DAG with moderate density
            if idx > 0:
                window = ids[max(0, idx - 8) : idx]
                k = rng.randint(0, min(2, len(window)))
                reqs = sorted(rng.sample(window, k=k))
        else:
            raise ValueError(f"Unknown archetype: {archetype}")

        units.append(
            Unit(
                id=uid,
                title=f"{archetype}_{uid}",
                difficulty=diff,
                importance=imp,
                requires=tuple(reqs),
            )
        )

    cg = CurriculumGraph(units, name=f"{archetype}_{num_units}_s{seed}")
    total_cost = sum(cg.get_unit_cost(u.id) for u in units)

    # Budget ratio: 35% for tight_budget, 55% for standard benchmarks
    ratio = 0.35 if archetype == "tight_budget" else 0.55
    target_budget = max(3, int(total_cost * ratio))

    # Split into daily budgets of 3~4 slots per day
    daily_budget: List[int] = []
    rem = target_budget
    while rem > 0:
        slot = min(4, rem)
        daily_budget.append(slot)
        rem -= slot

    return cg, ids, daily_budget
