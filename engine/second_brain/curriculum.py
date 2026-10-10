"""Graph-only curriculum projection and replanning. No model is reachable here."""
from collections import defaultdict
from dataclasses import asdict
import networkx as nx
from ..graph import CurriculumGraph, CyclicDependencyError, Unit, Exam, StudentState
from ..optimal import OptimalPlanner


def project(graph,include_candidates=False,min_weight=0.2,costs=None,values=None,completed=()):
    if not 0<=min_weight<=1:raise ValueError('Invalid evidence threshold')
    costs,values=costs or {},values or {}
    chunks={c['id']:c for c in graph['chunks']}
    if (set(costs)|set(values))-chunks.keys():raise ValueError('Unknown unit cost/value ID')
    if any(type(v) is not int or v<=0 for v in costs.values()):raise ValueError('Unit costs must be positive integer slots')
    if any(type(v) is not int or v<0 for v in values.values()):raise ValueError('Unit values must be nonnegative integers')
    taught=defaultdict(set);difficulty=defaultdict(list)
    for o in graph['occurrences']:
        if o['role']=='teaches':
            taught[o['concept_id']].add(o['bundle_id'])
            difficulty[o['bundle_id']].append(o['difficulty'])
    unit_cost={id:costs.get(id,max(difficulty[id],default=1)) for id in chunks}
    completed=set(completed)
    if completed-chunks.keys():raise ValueError('Unknown completed unit')
    # Each prerequisite concept needs one teaching resource, not every occurrence.
    # Preserve alternatives separately so repeated explanations are not AND-ed.
    groups=defaultdict(list)
    for e in graph['chunk_edges']:
        if include_candidates and e['status']!='REJECTED' and e['weight']>=min_weight and e['direction_margin']>0:
            for concept in e.get('concept_ids') or ['EDGE:'+e['id']]:
                if e['target_id'] in taught[concept]:continue
                groups[(e['target_id'],concept)].append(dict(e,
                    required_concept_id=concept,projection_origin='OUTCOME_PREREQUISITE'))
    for e in graph['relations']:
        if e['kind']!='REQUIRES' or e['status']=='REJECTED':continue
        if e['status']!='ACCEPTED' and not include_candidates:continue
        for a in taught[e['source_id']]:
            for b in taught[e['target_id']]:
                if a==b:continue
                # If both are explained in the same target, no external lesson is required.
                if b in taught[e['source_id']]:continue
                groups[(b,e['source_id'])].append({'id':e['id'],'source_id':a,'target_id':b,
                    'status':e['status'],'weight':1 if e['status']=='ACCEPTED' else 0.7,
                    'required_concept_id':e['source_id'],
                    'projection_origin':'CONCEPT_REQUIRES','evidence':e['evidence']})
    # Original graph may cycle. Only the planner projection must be a DAG.
    # Accepted cycles are errors; candidate cycle conflicts are explicitly preserved.
    selected={};quarantined=[];choices=[];dag=nx.DiGraph();dag.add_nodes_from(chunks)
    order=sorted(groups,key=lambda g:(not any(e['status']=='ACCEPTED' for e in groups[g]),
                                     -max(e['weight'] for e in groups[g]),g))
    for target,concept in order:
        options=groups[(target,concept)]
        accepted=any(e['status']=='ACCEPTED' for e in options)
        if accepted:options=[e for e in options if e['status']=='ACCEPTED']
        unique={}
        for e in options:
            a=e['source_id']
            if a not in unique or e['weight']>unique[a]['weight']:unique[a]=e
        options=sorted(unique.values(),key=lambda e:(e['source_id'] not in completed,
            0 if e['source_id'] in completed else unit_cost[e['source_id']],
            -e['weight'],e['source_id'],e['id']))
        chosen=None
        for e in options:
            a,b=e['source_id'],e['target_id']
            if nx.has_path(dag,b,a):
                quarantined.append(dict(e,reason='CANDIDATE_CYCLE_CONFLICT' if not accepted else 'RESOURCE_CYCLE_CONFLICT'))
                continue
            chosen=a;dag.add_edge(a,b)
            edge=selected.setdefault((a,b),dict(e,required_concept_ids=[],supporting_assertion_ids=[]))
            if concept not in edge['required_concept_ids']:edge['required_concept_ids'].append(concept)
            if e['id'] not in edge['supporting_assertion_ids']:edge['supporting_assertion_ids'].append(e['id'])
            break
        if chosen is None and accepted:
            raise CyclicDependencyError('Accepted prerequisite cycle in knowledge graph')
        choices.append({'target_id':target,'prerequisite_concept_id':concept,'chosen_source_id':chosen,
            'alternatives':[{'source_id':e['source_id'],'remaining_cost':0 if e['source_id'] in completed else unit_cost[e['source_id']],
                             'weight':e['weight']} for e in options]})
    units=[Unit(id=id,title=c['title'],difficulty=max(difficulty[id],default=1),
        importance=1,requires=tuple(sorted(dag.predecessors(id))),
        explicit_cost=unit_cost[id],explicit_value=values.get(id,1))
        for id,c in sorted(chunks.items())]
    return CurriculumGraph(units,'second_brain'),{'mode':'RESEARCH_CANDIDATES' if include_candidates else 'ACCEPTED_RELATIONS',
        'selected_edges':list(selected.values()),'quarantined_edges':quarantined,'resource_alternatives':choices,
        'resource_selection':'completed first, then least local cost; heuristic, not globally optimal OR planning',
        'source_graph_status':graph['status'],
        'cost_units':'relative learning slots unless caller supplies measured costs',
        'default_values':'equal importance; caller may supply exam priorities'}


def plan(graph,daily_budget,completed=(),targets=None,include_candidates=False,
         min_weight=0.2,costs=None,values=None,previous_selected=None):
    if not isinstance(daily_budget,list) or any(type(v) is not int or v<0 for v in daily_budget):
        raise ValueError('Daily budgets must be nonnegative integer slots')
    completed=set(completed)
    cg,projection=project(graph,include_candidates,min_weight,costs,values,completed)
    if completed-cg.units.keys():raise ValueError('Unknown completed unit')
    targets=sorted(cg.units) if targets is None else list(targets)
    if set(targets)-cg.units.keys():raise ValueError('Unknown target unit')
    targets=[id for id in targets if id not in completed]
    state=StudentState(mastered=completed)
    result=OptimalPlanner().plan(cg,Exam(targets,daily_budget),state,
        previous_selected=set(previous_selected or []))
    return dict(asdict(result),projection=projection,
        audit=cg.verify_plan(result.selected_units,result.daily_schedule,daily_budget,state))
