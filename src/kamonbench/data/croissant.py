from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from kamonbench.generator.assets import ChargeAssets, default_basic_charges_path, load_basic_charges


def load_jsonl(filepath: Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    with filepath.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def create_translation_map(parsing_translation_data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    translation_map: dict[str, dict[str, Any]] = {}
    for item in parsing_translation_data:
        desc = item['description']
        translation_map[desc] = {'analysis': item.get('analysis', []), 'translation': item.get('translation', '')}
    return translation_map


def validate_translation_map(all_images: list[dict[str, Any]], translation_map: dict[str, dict[str, Any]]) -> None:
    missing = []
    empty_analysis = []
    empty_translation = []
    for desc in sorted({img['description'] for img in all_images}):
        if desc not in translation_map:
            missing.append(desc)
            continue
        item = translation_map[desc]
        if not item.get('analysis'):
            empty_analysis.append(desc)
        if not item.get('translation'):
            empty_translation.append(desc)

    problems = []
    for label, values in [
        ('missing labels', missing),
        ('empty analysis', empty_analysis),
        ('empty translation', empty_translation),
    ]:
        if values:
            examples = ', '.join(repr(x) for x in values[:5])
            problems.append(f'{label}: {len(values)} examples ({examples})')

    if problems:
        raise ValueError(
            'Incomplete parsing_translation.jsonl for --label-mode jp-en; '
            + '; '.join(problems)
            + '. Provide external labels for every generated description, or use --label-mode program.'
        )


PROGRAM_MODIFIER_TO_ID: dict[str, int] = {'': 0, '覗き': 1, '豆': 2, '三つ盛り': 3, '尻合せ三つ': 4, '頭合せ三つ': 5}
IMAGE_FILE_SET_ID = 'image-files'
ARCHIVE_FILE_OBJECT_ID = 'kamon-bench-archive'


def _charge_id_from_path(path: str) -> str:
    return Path(path).stem


def _compact_description(text: str) -> str:
    return ''.join(text.split())


def _tokens_to_analysis(tokens: list[str]) -> list[dict[str, Any]]:
    return [{'expr': t} for t in tokens]


def _program_tokens_for_components(
    assets: ChargeAssets, *, base_term: str, container_terms: list[str]
) -> tuple[str, dict[str, list[str]]]:
    motif_path = assets.motifs.get(base_term)
    if not motif_path:
        raise KeyError(f'Unknown motif term in synthetic.jsonl: {base_term}')
    motif_id = _charge_id_from_path(motif_path)
    component_tokens: dict[str, list[str]] = {base_term: [f'M:{motif_id}']}

    for term in container_terms:
        container_path = assets.containers.get(term)
        if not container_path:
            raise KeyError(f'Unknown container term in synthetic.jsonl: {term}')
        container_id = _charge_id_from_path(container_path)
        component_tokens[term] = [f'C:{container_id}']

    return motif_id, component_tokens


def create_program_token_map(synthetic_data: list[dict[str, Any]], assets: ChargeAssets) -> dict[str, dict[str, Any]]:
    token_map: dict[str, dict[str, Any]] = {}

    for item in synthetic_data:
        desc = item.get('description', '')
        if not desc:
            continue
        images = item.get('images', [])
        if not images or not isinstance(images, list):
            raise ValueError(f'Expected images list for description {desc!r}')

        first_img = images[0]
        base_motif = first_img.get('base_motif', {})
        base_term = base_motif.get('description', '') if isinstance(base_motif, dict) else ''
        if not base_term:
            raise ValueError(
                f'Missing base_motif.description for description {desc!r}. '
                'Generate synthetic data with --save-components.'
            )

        containers_list = first_img.get('containers', [])
        container_terms = []
        if isinstance(containers_list, list):
            for c in containers_list:
                if not isinstance(c, dict):
                    continue
                term = c.get('description', '')
                if term:
                    container_terms.append(term)

        compact_container_terms = [_compact_description(t) for t in container_terms]
        compact_base_term = _compact_description(base_term)

        expected_prefix = ('に'.join(compact_container_terms) + 'に') if compact_container_terms else ''
        if expected_prefix and not desc.startswith(expected_prefix):
            raise ValueError(f'Unexpected description format for {desc!r}: expected prefix {expected_prefix!r}')

        remainder = desc[len(expected_prefix) :] if expected_prefix else desc
        if not remainder.endswith(compact_base_term):
            raise ValueError(
                f'Unexpected description format for {desc!r}: expected suffix base_term={compact_base_term!r}, '
                f'got {remainder!r}'
            )
        modifier = remainder[: -len(compact_base_term)]
        if modifier not in PROGRAM_MODIFIER_TO_ID:
            raise ValueError(f'Unknown modifier in description {desc!r}: {modifier!r}')

        motif_id, component_tokens = _program_tokens_for_components(
            assets, base_term=base_term, container_terms=container_terms
        )

        composite_tokens = [component_tokens[t][0] for t in container_terms]
        composite_tokens.append(f'X:{PROGRAM_MODIFIER_TO_ID[modifier]}')
        composite_tokens.append(f'M:{motif_id}')

        token_map[desc] = {'analysis': _tokens_to_analysis(composite_tokens), 'translation': ' '.join(composite_tokens)}
        for term, toks in component_tokens.items():
            token_map.setdefault(term, {'analysis': _tokens_to_analysis(toks), 'translation': ' '.join(toks)})

    return token_map


def extract_all_images(synthetic_data: list[dict[str, Any]]):
    all_images: list[dict[str, Any]] = []
    image_id_counter = 0
    image_groups: dict[str, str] = {}

    for item in synthetic_data:
        main_description = item['description']

        for img in item.get('images', []):
            main_path = img['path']

            base_motif_path = img.get('base_motif', {}).get('path')
            container_paths = [c['path'] for c in img.get('containers', [])]

            all_images.append(
                {
                    'id': image_id_counter,
                    'image_path': main_path,
                    'description': main_description,
                    'is_composite': True,
                    'base_motif_id': None,
                    'container_ids': [],
                    'base_motif_path': base_motif_path,
                    'container_paths': container_paths,
                    'main_image_path': main_path,
                }
            )
            image_groups[main_path] = main_path
            main_image_id = image_id_counter
            image_id_counter += 1

            if base_motif_path:
                base_motif_desc = img['base_motif'].get('description', '')
                all_images.append(
                    {
                        'id': image_id_counter,
                        'image_path': base_motif_path,
                        'description': base_motif_desc,
                        'is_composite': False,
                        'base_motif_id': None,
                        'container_ids': [],
                        'base_motif_path': None,
                        'container_paths': [],
                        'main_image_path': main_path,
                    }
                )
                image_groups[base_motif_path] = main_path
                all_images[main_image_id]['base_motif_id'] = image_id_counter
                image_id_counter += 1

            for container in img.get('containers', []):
                container_path = container['path']
                container_desc = container.get('description', '')
                all_images.append(
                    {
                        'id': image_id_counter,
                        'image_path': container_path,
                        'description': container_desc,
                        'is_composite': False,
                        'base_motif_id': None,
                        'container_ids': [],
                        'base_motif_path': None,
                        'container_paths': [],
                        'main_image_path': main_path,
                    }
                )
                image_groups[container_path] = main_path
                all_images[main_image_id]['container_ids'].append(image_id_counter)
                image_id_counter += 1

    return all_images, image_groups


def assign_splits(
    all_images: list[dict[str, Any]],
    image_groups: dict[str, str],  # legacy, unused
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    del image_groups
    main_image_paths = sorted({img['main_image_path'] for img in all_images})

    if seed is None:
        seed = len(main_image_paths)
    random.seed(seed)

    shuffled_main_images = main_image_paths.copy()
    random.shuffle(shuffled_main_images)

    total = len(shuffled_main_images)
    train_size = int(total * train_ratio)
    dev_size = int(total * dev_ratio)

    split_assignment: dict[str, str] = {}
    for i, main_path in enumerate(shuffled_main_images):
        if i < train_size:
            split_assignment[main_path] = 'train'
        elif i < train_size + dev_size:
            split_assignment[main_path] = 'dev'
        else:
            split_assignment[main_path] = 'test'

    for img in all_images:
        img['split'] = split_assignment[img['main_image_path']]

    return all_images


def create_croissant_metadata(
    all_images: list[dict[str, Any]],
    translation_map: dict[str, dict[str, Any]],
    dataset_name: str = 'KamonBench',
    dataset_description: str = 'Synthetic Kamon (Japanese family crest) image dataset',
    version: str = '1.0.0',
    label_mode: str = 'jp-en',
    archive_url: str | None = None,
    archive_sha256: str | None = None,
) -> dict[str, Any]:
    if bool(archive_url) != bool(archive_sha256):
        raise ValueError('archive_url and archive_sha256 must be provided together')

    records: list[dict[str, Any]] = []
    for img in all_images:
        desc = img['description']
        trans_data = translation_map.get(desc, {})

        record = {
            'id': img['id'],
            'image_path': img['image_path'],
            'description': desc,
            'analysis': trans_data.get('analysis', []),
            'translation': trans_data.get('translation', ''),
            'is_composite': img['is_composite'],
            'split': img['split'],
        }

        if img['is_composite']:
            components: list[int] = []
            if img['base_motif_id'] is not None:
                components.append(img['base_motif_id'])
            components.extend(img['container_ids'])
            if components:
                record['component_ids'] = components

        records.append(record)

    if label_mode == 'program':
        translation_description = 'Whitespace-tokenized program label text'
        analysis_description = 'Structured program label tokens'
    else:
        translation_description = 'English translation of the description'
        analysis_description = 'Dependency analysis of the description'

    image_file_set = {
        '@type': 'cr:FileSet',
        '@id': IMAGE_FILE_SET_ID,
        'name': IMAGE_FILE_SET_ID,
        'description': 'All PNG Kamon images under dataset01/',
        'encodingFormat': 'image/png',
        'includes': 'dataset01/*.png',
    }
    if archive_url and archive_sha256:
        image_file_set['containedIn'] = {'@id': ARCHIVE_FILE_OBJECT_ID}
        distribution = [
            {
                '@type': 'cr:FileObject',
                '@id': ARCHIVE_FILE_OBJECT_ID,
                'name': Path(archive_url).name or 'kamon_bench.zip',
                'description': 'Archive containing the released KamonBench dataset files',
                'contentUrl': archive_url,
                'encodingFormat': 'application/zip',
                'sha256': archive_sha256,
            },
            image_file_set,
        ]
    else:
        distribution = [image_file_set]

    return {
        '@context': {
            '@language': 'en',
            '@vocab': 'https://schema.org/',
            'sc': 'https://schema.org/',
            'cr': 'http://mlcommons.org/croissant/',
            'dct': 'http://purl.org/dc/terms/',
            'rai': 'http://mlcommons.org/croissant/RAI/',
            'prov': 'http://www.w3.org/ns/prov#',
            'citeAs': 'cr:citeAs',
            'fileSet': 'cr:fileSet',
            'includes': 'cr:includes',
        },
        '@type': 'sc:Dataset',
        'name': dataset_name,
        'description': dataset_description,
        'version': version,
        'url': 'https://huggingface.co/datasets/SakanaAI/KamonBench',
        'dct:conformsTo': ['http://mlcommons.org/croissant/1.0', 'http://mlcommons.org/croissant/RAI/1.0'],
        'license': 'https://creativecommons.org/licenses/by-nc/4.0/',
        'citeAs': 'KamonBench: A Grammar-Based Dataset for Evaluating Compositional Factor Recovery in Vision-Language Models',
        'datePublished': '2026-05-01',
        'prov:wasDerivedFrom': [
            {
                '@id': 'https://github.com/Rebolforces/kamondataset',
                'prov:label': 'Rebolforces Kamon dataset',
                'sc:license': 'No license declared in source repository',
            }
        ],
        'prov:wasGeneratedBy': [
            {
                '@type': 'prov:Activity',
                'prov:label': 'Synthetic Kamon generation',
                'prov:type': 'Synthetic data generation',
                'prov:description': (
                    'Rendered synthetic Kamon images from source motif assets using the KamonBench grammar '
                    'and rule-based image composition.'
                ),
            },
            {
                '@type': 'prov:Activity',
                'prov:label': 'Kamon description annotation',
                'prov:type': 'Annotation',
                'prov:description': (
                    'Assigned KDL descriptions, Japanese analysis labels, English translations, and deterministic '
                    'program labels used by the released Croissant variants.'
                ),
            },
        ],
        'rai:dataCollection': (
            'The released dataset was generated synthetically from source motif assets from '
            'https://github.com/Rebolforces/kamondataset. The generator composes base motifs, optional '
            'containers, and a limited set of modifiers into rendered PNG Kamon images.'
        ),
        'rai:dataCollectionType': 'Synthetic image generation from source motif assets and rule-based composition.',
        'rai:dataCollectionRawData': (
            'The source assets are Kamon motif images from https://github.com/Rebolforces/kamondataset. '
            'The release contains generated PNG images and Croissant metadata.'
        ),
        'rai:dataAnnotationProtocol': (
            'Composite records include KDL descriptions. Japanese analysis and English translation labels are '
            'externally supplied annotations; program-label Croissant variants derive non-linguistic labels '
            'deterministically from saved generator components.'
        ),
        'rai:dataUseCases': (
            'Research benchmark for image-to-structure prediction, image-to-KDL generation, compositional '
            'generalization, factor recognition, controlled recombination, and representation analysis.'
        ),
        'rai:dataLimitations': (
            'The data are synthetic and do not cover the full distribution of real historical or professionally '
            'rendered Kamon. The released generator uses a limited grammar, one level of containment, a fixed '
            'set of containers and modifiers, and source motif assets from one repository.'
        ),
        'rai:dataBiases': (
            'The dataset reflects the coverage, style, and taxonomy of the source motif assets and the '
            'KamonBench generator. It should not be treated as representative of all real Kamon traditions, '
            'historical periods, regions, or rendering styles.'
        ),
        'rai:personalSensitiveInformation': (
            'The released dataset contains synthetic crest renderings and labels, not personal or sensitive '
            'information.'
        ),
        'rai:dataSocialImpact': (
            'The intended positive impact is to provide a culturally grounded benchmark for compositional '
            'vision-language research involving Japanese Kamon. No obvious negative societal impacts were '
            'identified, but the dataset should not be used as an authoritative cultural or historical catalogue.'
        ),
        'rai:hasSyntheticData': True,
        'distribution': distribution,
        'recordSet': [
            {
                '@type': 'cr:RecordSet',
                '@id': 'images',
                'name': 'images',
                'description': 'All Kamon images including main images, base motifs, and containers',
                'field': [
                    {
                        '@type': 'cr:Field',
                        '@id': 'images/id',
                        'name': 'id',
                        'description': 'Unique image identifier',
                        'dataType': 'sc:Integer',
                    },
                    {
                        '@type': 'cr:Field',
                        '@id': 'images/image_path',
                        'name': 'image_path',
                        'description': 'Path to the image file',
                        'dataType': 'sc:Text',
                    },
                    {
                        '@type': 'cr:Field',
                        '@id': 'images/image',
                        'name': 'image',
                        'description': 'Image content',
                        'dataType': 'sc:ImageObject',
                        'source': {'fileSet': {'@id': IMAGE_FILE_SET_ID}, 'extract': {'fileProperty': 'content'}},
                    },
                    {
                        '@type': 'cr:Field',
                        '@id': 'images/description',
                        'name': 'description',
                        'description': 'Japanese description of the Kamon',
                        'dataType': 'sc:Text',
                    },
                    {
                        '@type': 'cr:Field',
                        '@id': 'images/translation',
                        'name': 'translation',
                        'description': translation_description,
                        'dataType': 'sc:Text',
                    },
                    {
                        '@type': 'cr:Field',
                        '@id': 'images/analysis',
                        'name': 'analysis',
                        'description': analysis_description,
                        'dataType': 'sc:Text',
                        'repeated': True,
                    },
                    {
                        '@type': 'cr:Field',
                        '@id': 'images/is_composite',
                        'name': 'is_composite',
                        'description': 'Whether this image is a composite of other images',
                        'dataType': 'sc:Boolean',
                    },
                    {
                        '@type': 'cr:Field',
                        '@id': 'images/component_ids',
                        'name': 'component_ids',
                        'description': 'IDs of component images (base motif and containers) for composite images',
                        'dataType': 'sc:Integer',
                        'repeated': True,
                        'references': {'@id': 'images/id'},
                    },
                    {
                        '@type': 'cr:Field',
                        '@id': 'images/split',
                        'name': 'split',
                        'description': 'Dataset split (train, dev, or test)',
                        'dataType': 'sc:Text',
                    },
                ],
                'data': records,
            }
        ],
    }


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('convert-to-croissant', help='Convert JSONL into Croissant.')
    p.add_argument('--dataset-dir', type=Path, default=Path('dataset01'))
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--label-mode', choices=['program', 'jp-en'], required=True)
    p.add_argument('--basic-charges-csv', type=Path, default=default_basic_charges_path())
    p.add_argument('--train-ratio', type=float, default=0.8)
    p.add_argument('--dev-ratio', type=float, default=0.1)
    p.add_argument('--test-ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--name', type=str, default='KamonBench')
    p.add_argument('--archive-url', type=str, default=None)
    p.add_argument('--archive-sha256', type=str, default=None)
    p.add_argument(
        '--description',
        type=str,
        default='Synthetic Kamon (Japanese family crest) image dataset with hierarchical composition structure',
    )
    p.add_argument('--version', type=str, default='1.0.0')
    p.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    if args.output is None:
        args.output = args.dataset_dir / 'kamon_croissant.json'

    if abs(args.train_ratio + args.dev_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError('Split ratios must sum to 1.0')

    synthetic_path = args.dataset_dir / 'synthetic.jsonl'
    if not synthetic_path.exists():
        raise FileNotFoundError(synthetic_path)

    synthetic_data = load_jsonl(synthetic_path)
    translation_path = args.dataset_dir / 'parsing_translation.jsonl'
    label_mode = args.label_mode

    all_images, image_groups = extract_all_images(synthetic_data)
    seed = args.seed if args.seed is not None else len({img['main_image_path'] for img in all_images})
    all_images = assign_splits(all_images, image_groups, args.train_ratio, args.dev_ratio, args.test_ratio, seed)

    if label_mode == 'jp-en':
        if not translation_path.exists():
            raise FileNotFoundError(translation_path)
        translation_map = create_translation_map(load_jsonl(translation_path))
        validate_translation_map(all_images, translation_map)
    else:
        assets = load_basic_charges(args.basic_charges_csv)
        translation_map = create_program_token_map(synthetic_data, assets)

    metadata = create_croissant_metadata(
        all_images,
        translation_map,
        args.name,
        args.description,
        args.version,
        label_mode=label_mode,
        archive_url=args.archive_url,
        archive_sha256=args.archive_sha256,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    return 0
