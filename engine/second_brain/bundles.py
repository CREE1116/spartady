"""Courseware anchors, and transcript/note alignment from displayed-slide intervals."""
from collections import defaultdict

KINDS = {'transcript', 'note'}

def make_bundles(slides, displays=(), supplements=()):
    anchors = {s['id']: s for s in slides}
    if len(anchors) != len(slides):
        raise ValueError('Duplicate slide IDs')
    timelines = defaultdict(list)
    for frame in displays:
        if frame['slide_id'] not in anchors or frame['start_ms'] >= frame['end_ms']:
            raise ValueError('Invalid displayed-slide interval')
        timelines[frame['lecture_id']].append(frame)
    for frames in timelines.values():
        frames.sort(key=lambda f: f['start_ms'])
        if any(a['end_ms'] > b['start_ms'] for a,b in zip(frames, frames[1:])):
            raise ValueError('Displayed-slide intervals must not overlap')
    attached = defaultdict(list)
    unresolved = []
    source_ids = set(anchors)
    for source in supplements:
        if source['id'] in source_ids or source['kind'] not in KINDS:
            raise ValueError('Duplicate source ID or invalid supplement kind')
        source_ids.add(source['id'])
        start, end = source['start_ms'], source['end_ms']
        if start > end:
            raise ValueError('Reversed supplement time range')
        frames = timelines.get(source['lecture_id'], [])
        matches = [f for f in frames if (f['start_ms'] <= start < f['end_ms'])
                   and (end <= f['end_ms'])]
        if len(matches) != 1:
            # Cross-slide speech is retained for review; never copied wholesale to both.
            unresolved.append(dict(source, alignment_status='UNRESOLVED'))
            continue
        anchor=anchors[matches[0]['slide_id']]
        attached[anchor['id']].append(dict(source, alignment_status='TIME_ANCHORED',
            linked_courseware_id=anchor['id'],linked_document=anchor['document'],linked_page=anchor['page']))
    bundles = []
    for slide in slides:
        if not slide['text'].strip():
            raise ValueError('Empty courseware anchor')
        anchor = dict(slide, kind='courseware')
        bundles.append({'id': slide['id'], 'title': slide.get('title',''),
                        'document': slide['document'], 'page': slide['page'],
                        'sources': [anchor] + attached[slide['id']],
                        'boundary_strategy': 'slide_anchor_with_time_linked_context',
                        'interpretation': 'SOURCE_MEMBERSHIP_NOT_PREREQUISITE'})
    return bundles, unresolved
