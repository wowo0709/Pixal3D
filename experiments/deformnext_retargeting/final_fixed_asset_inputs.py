"""Require actual inputs for every DEV128 case, including the old prefix32.

The historical extension packet declares128 sample IDs but stores only the
additional96 assets. It must never be treated as a complete input catalogue.
"""
from pathlib import Path
import re

from .dev128_pipeline_contract import validate_population
from .selected_pipeline_model_contract import require


SCHEMA = 'paired-DEV128-complete-asset-inputs-v1'
ROLES = ('target_rgb','source_rgb_9','source_rgb_10','source_rgb_11',
         'GT_rgb_9','GT_rgb_10','source_geometry','target_vertices',
         'transport_cache','transport_cache_pbr','pbr_dump')


def validate_packet(packet):
    require(packet.get('schema') == SCHEMA and packet.get('no_TEST_data') is True,
            'explicit complete128 development input packet required; additional96 is not complete')
    ids,clusters,records = packet['sample_ids'],packet['source_clusters_by_sample'],packet['records']
    require(len(records) == 128 and [a['sample_id'] for a in records] == ids,
            'all128 actual input records in the original panel order required')
    validate_population([a['row'] for a in records],ids,clusters)
    for a in records:
        require(a['source_cluster_id'] == clusters[a['sample_id']]
                and a['row']['sample_id'] == a['sample_id']
                and set(a['inputs']) == set(a['canonical_files']) == set(ROLES),
                'each actual sample/source needs all eleven original input roles')
        canonical = a['canonical_files']; row = a['row']
        expected = dict(target_rgb=row['target_rgb_input'],source_geometry=row['source_geometry'],
            target_vertices=row['target_vertices'],transport_cache=row['transport_cache'],
            transport_cache_pbr=row['transport_cache_pbr'])
        expected.update(zip(('source_rgb_9','source_rgb_10','source_rgb_11'),row['source_images']))
        expected.update(zip(('GT_rgb_9','GT_rgb_10'),row['evaluation_target_images']))
        require(len(row['evaluation_target_images']) == 2
                and all(canonical[k] == v for k,v in expected.items()),
                'actual source/target/camera paths must match the saved validation row')
        for role,spec in a['inputs'].items():
            require(Path(spec['path']).is_absolute()
                    and spec['canonical_path'] == canonical[role]
                    and spec['canonical_path'].startswith('/file4/')
                    and isinstance(spec['SHA256'],str) and re.fullmatch('[0-9a-f]{64}',spec['SHA256'])
                    and type(spec['bytes']) is int and spec['bytes'] > 0,
                    'explicit actual local/canonical path, file SHA256 and size required')
        rgb = [a['inputs'][k]['path'] for k in ROLES[:4]]
        require(a['RGB_paths'] == rgb and len(set(rgb)) == 4,
                'same four actual edited-target/original-source RGB inputs required')
        views = [dict(camera_id=cid,input_camera_index=cid-8,
            GT_RGBA=a['inputs']['target_rgb' if cid==8 else 'GT_rgb_'+str(cid)]['path'])
            for cid in (8,9,10)]
        require(a['render_views'] == views, 'same actual three target/source render references required')
    manifest = packet['validation_manifest']
    require(Path(manifest['path']).is_absolute() and isinstance(manifest['SHA256'],str)
            and re.fullmatch('[0-9a-f]{64}',manifest['SHA256']),
            'explicit unchanged validation manifest identity required')
    return ids
