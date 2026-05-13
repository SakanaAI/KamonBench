#!/usr/bin/env python3
"""Generate HTML visualization for Kamon model outputs."""

import argparse
import base64
import html
import os
from dataclasses import dataclass, replace

import jsonlines

from kamonbench import defaults
from kamonbench.metrics import character_error_rate, token_error_rate
from kamonbench.text import normalize_japanese_for_comparison

DEFAULT_REPORT_OUTPUT = 'visualization.html'


@dataclass(frozen=True)
class ReportConfig:
    input_jsonl: str | None = None
    output: str | None = None
    model: str = 'vgg'
    use_translation: bool = False


@dataclass(frozen=True)
class ReportMetrics:
    total: int
    normalized_string_correct: int
    insertions: int
    deletions: int
    substitutions: int
    normalized_ref_chars: int
    token_insertions: int
    token_deletions: int
    token_substitutions: int
    normalized_ref_tokens: int

    @property
    def string_correct(self) -> int:
        return self.normalized_string_correct

    @property
    def string_accuracy(self) -> float:
        return self.normalized_string_accuracy

    @property
    def normalized_string_accuracy(self) -> float:
        return self.normalized_string_correct / self.total if self.total else 0.0

    @property
    def edit_distance(self) -> int:
        return self.insertions + self.deletions + self.substitutions

    @property
    def character_error_rate(self) -> float:
        return self.edit_distance / self.normalized_ref_chars if self.normalized_ref_chars else 0.0

    @property
    def token_edit_distance(self) -> int:
        return self.token_insertions + self.token_deletions + self.token_substitutions

    @property
    def token_error_rate(self) -> float:
        return self.token_edit_distance / self.normalized_ref_tokens if self.normalized_ref_tokens else 0.0


def resolve_report_config(config: ReportConfig) -> ReportConfig:
    input_jsonl = config.input_jsonl or os.path.join(
        defaults.output_dir(config.model, use_translation=config.use_translation),
        defaults.DEFAULT_INFERENCE_RESULT_NAME,
    )
    output = config.output or DEFAULT_REPORT_OUTPUT
    return replace(config, input_jsonl=input_jsonl, output=output)


def preprocess_text_for_comparison(text):
    """Preprocess text for character edit distance comparison.

    Args:
        text: Input text string

    Returns:
        Preprocessed text with whitespace removed and converted to hiragana
    """
    return normalize_japanese_for_comparison(text)


def calculate_character_error_rate(reference: str, predicted: str) -> tuple[int, int, int, float]:
    """Calculate character-level edit distance and error rate.

    Args:
        reference: Ground truth string
        predicted: Predicted string

    Returns:
        Tuple of (insertions, deletions, substitutions, error_rate)
    """
    return character_error_rate(reference, predicted)


def calculate_token_error_rate(reference: str, predicted: str) -> tuple[int, int, int, float]:
    return token_error_rate(reference, predicted)


def include_in_visual_report(item: dict) -> bool:
    image_path = item.get('image', '')
    return '_base' not in image_path and '_container' not in image_path


def is_program_label(text: str) -> bool:
    tokens = text.split()
    return bool(tokens) and all(token.startswith(('C:', 'X:', 'M:')) for token in tokens)


def is_program_report(data: list[dict]) -> bool:
    return bool(data) and all(is_program_label(item.get('reference', '')) for item in data)


def calculate_report_metrics(data: list[dict]) -> ReportMetrics:
    insertions = deletions = substitutions = 0
    normalized_string_correct = normalized_ref_chars = 0
    token_insertions = token_deletions = token_substitutions = normalized_ref_tokens = 0

    for item in data:
        reference = item.get('reference', '')
        predicted = item.get('predicted', '')
        ins, dels, subs, _ = calculate_character_error_rate(reference, predicted)
        insertions += ins
        deletions += dels
        substitutions += subs

        tok_ins, tok_dels, tok_subs, _ = calculate_token_error_rate(reference, predicted)
        token_insertions += tok_ins
        token_deletions += tok_dels
        token_substitutions += tok_subs

        ref_norm = preprocess_text_for_comparison(reference)
        pred_norm = preprocess_text_for_comparison(predicted)
        normalized_string_correct += int(ref_norm == pred_norm)
        normalized_ref_chars += len(ref_norm)
        normalized_ref_tokens += len(reference.split())

    return ReportMetrics(
        total=len(data),
        normalized_string_correct=normalized_string_correct,
        insertions=insertions,
        deletions=deletions,
        substitutions=substitutions,
        normalized_ref_chars=normalized_ref_chars,
        token_insertions=token_insertions,
        token_deletions=token_deletions,
        token_substitutions=token_substitutions,
        normalized_ref_tokens=normalized_ref_tokens,
    )


def format_count_rate(count: int, total: int) -> str:
    rate = count / total if total else 0.0
    return f'{count}/{total} = {rate:.6f}'


def create_aligned_display(reference: str, predicted: str, translation: str = '') -> str:
    """Create HTML for aligned display of reference and predicted strings."""
    ref_html = html.escape(reference)
    pred_html = html.escape(predicted)
    trans_html = html.escape(translation) if translation else ''

    translation_line = (
        f"""
        <div class="translation-line">
            <span class="label">EN:</span>
            <span class="text translation">{trans_html}</span>
        </div>
    """
        if translation
        else ''
    )

    return f"""
    <div class="alignment-container">
        <div class="reference-line">
            <span class="label">REF:</span>
            <span class="text">{ref_html}</span>
        </div>
        <div class="predicted-line">
            <span class="label">PRD:</span>
            <span class="text">{pred_html}</span>
        </div>
        {translation_line}
    </div>
    """


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string for embedding in HTML."""
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            # Determine image format from extension
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
            }.get(ext, 'image/jpeg')
            return f'data:{mime_type};base64,{img_base64}'
    except Exception as e:
        print(f'Warning: Could not load image {image_path}: {e}')
        return ''


def generate_html_page(jsonl_file: str, output_file: str) -> None:
    """Generate HTML page from JSONL file."""

    data = []
    with jsonlines.open(jsonl_file) as reader:
        for item in reader:
            if include_in_visual_report(item):
                data.append(item)

    if not data:
        print('No data found in the JSONL file.')
        return

    metrics = calculate_report_metrics(data)
    program_report = is_program_report(data)
    if program_report:
        headline_label = 'TER'
        headline_rate = metrics.token_error_rate
        edit_stats_html = f"""
                <div class="stat-item">
                    <div class="stat-value">{metrics.token_insertions}</div>
                    <div>Insertions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{metrics.token_deletions}</div>
                    <div>Deletions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{metrics.token_substitutions}</div>
                    <div>Substitutions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{metrics.normalized_ref_tokens}</div>
                    <div>Total Ref Tokens</div>
                </div>
        """
    else:
        headline_label = 'CER'
        headline_rate = metrics.character_error_rate
        edit_stats_html = f"""
                <div class="stat-item">
                    <div class="stat-value">{metrics.insertions}</div>
                    <div>Insertions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{metrics.deletions}</div>
                    <div>Deletions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{metrics.substitutions}</div>
                    <div>Substitutions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{metrics.normalized_ref_chars}</div>
                    <div>Total Ref Chars</div>
                </div>
        """

    # Find cases where prediction = reference AND no training images
    perfect_predictions_no_training = []
    nenit = 0
    for idx, item in enumerate(data):
        reference = item.get('reference', '')
        predicted = item.get('predicted', '')
        train_images_reference = item.get('train_images_reference', [])
        train_images_predicted = item.get('train_images_predicted', [])

        if len(train_images_reference) == 0 and len(train_images_predicted) == 0:
            nenit += 1

        if preprocess_text_for_comparison(reference) == preprocess_text_for_comparison(predicted):
            if len(train_images_reference) == 0 and len(train_images_predicted) == 0:
                perfect_predictions_no_training.append((idx + 1, reference))

    nit_cor = len(perfect_predictions_no_training)
    nit_rate = nit_cor / nenit if nenit else 0.0
    print(f'Acc: {format_count_rate(metrics.normalized_string_correct, metrics.total)}')
    print(f'Acc_NIT: {nit_cor}/{nenit} = {nit_rate:.6f}')
    if program_report:
        print(f'TER: {metrics.token_error_rate:.6f}')
    print(f'CER: {metrics.character_error_rate:.6f}')
    print(f'Total examples not in training: {nenit}')

    # Generate navigation links
    navigation_html = ''
    if perfect_predictions_no_training:
        navigation_html = f"""
        <div class="navigation-section">
            <h3>Perfect Predictions with No Training Support ({len(perfect_predictions_no_training)} cases)</h3>
            <p>These are cases where the model correctly predicted the reference description, but there were no training images with the same description:</p>
            <div class="navigation-links">
                {''.join([f'<a href="#example-{idx}" class="nav-link">Example {idx}: {html.escape(desc)}</a>' for idx, desc in perfect_predictions_no_training])}
            </div>
        </div>
        """

    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kamon Model Output Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin: -20px -20px 30px -20px;
            border-radius: 0 0 15px 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            margin: 0;
            font-size: 2em;
            text-align: center;
        }}

        .cer-stats {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            margin-top: 15px;
            border-radius: 10px;
            text-align: center;
        }}

        .cer-stats h2 {{
            margin: 0 0 10px 0;
            font-size: 1.5em;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }}

        .stat-item {{
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 8px;
        }}

        .stat-value {{
            font-size: 1.3em;
            font-weight: bold;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .item {{
            background: white;
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }}

        .item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }}

        .image-container {{
            text-align: center;
            margin-bottom: 15px;
        }}

        .main-image {{
            max-width: 300px;
            max-height: 300px;
            border: 2px solid #ddd;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .alignment-container {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #667eea;
        }}

        .reference-line, .predicted-line, .translation-line {{
            margin: 5px 0;
            font-family: 'Courier New', monospace;
            font-size: 16px;
        }}

        .reference-line .label {{
            display: inline-block;
            width: 40px;
            font-weight: bold;
            color: #28a745;
        }}

        .predicted-line .label {{
            display: inline-block;
            width: 40px;
            font-weight: bold;
            color: #dc3545;
        }}

        .translation-line .label {{
            display: inline-block;
            width: 40px;
            font-weight: bold;
            color: #6c757d;
        }}

        .translation {{
            font-style: italic;
            color: #495057;
        }}

        .text {{
            background: white;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}

        .training-images {{
            margin-top: 20px;
        }}

        .training-images h4 {{
            margin-bottom: 10px;
            color: #666;
            font-size: 0.9em;
        }}

        .training-images-container {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}

        .training-image {{
            max-width: 80px;
            max-height: 80px;
            border: 1px solid #ddd;
            border-radius: 6px;
            opacity: 0.8;
            transition: opacity 0.2s ease;
        }}

        .training-image:hover {{
            opacity: 1;
        }}

        .item-number {{
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 15px;
        }}

        .error-info {{
            font-size: 0.8em;
            color: #666;
            margin-top: 10px;
            font-style: italic;
        }}

        .error-info.perfect {{
            color: #0066cc;
            font-weight: bold;
        }}

        .navigation-section {{
            background: white;
            margin: 20px -20px 30px -20px;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .navigation-section h3 {{
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}

        .navigation-links {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}

        .nav-link {{
            display: block;
            padding: 8px 12px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            text-decoration: none;
            color: #495057;
            font-size: 0.9em;
            transition: all 0.2s ease;
        }}

        .nav-link:hover {{
            background: #e9ecef;
            border-color: #667eea;
            color: #667eea;
            transform: translateY(-1px);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Kamon Model Output Visualization</h1>
        <div class="cer-stats">
            <h2>{headline_label}: {headline_rate:.6f} ({headline_rate * 100:.1f}%)</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{format_count_rate(metrics.normalized_string_correct, metrics.total)}</div>
                    <div>Acc</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{format_count_rate(nit_cor, nenit)}</div>
                    <div>Acc_NIT</div>
                </div>
                {edit_stats_html}
                <div class="stat-item">
                    <div class="stat-value">{metrics.total}</div>
                    <div>Total Examples</div>
                </div>
            </div>
        </div>
    </div>

    {navigation_html}

    <div class="container">
"""

    # Add each item
    for idx, item in enumerate(data):
        reference = item.get('reference', '')
        predicted = item.get('predicted', '')
        image_path = item.get('image', '')
        train_images_reference = item.get('train_images_reference', [])
        train_images_predicted = item.get('train_images_predicted', [])
        translation = item.get('translation', '')

        # Convert main image to base64
        main_image_data = image_to_base64(image_path) if image_path else ''

        # Create alignment display
        alignment_html = create_aligned_display(reference, predicted, '' if program_report else translation)

        # Training images handling
        training_images_html = ''

        # Check if reference and predicted are identical
        if reference == predicted:
            # When identical, combine the images and show with unified label
            all_training_images = list(set(train_images_reference + train_images_predicted))  # Remove duplicates
            if all_training_images:
                img_tags = []
                for train_img_path in all_training_images:
                    train_img_data = image_to_base64(train_img_path)
                    if train_img_data:
                        img_tags.append(
                            f'<img src="{train_img_data}" class="training-image" alt="Training image matching both reference and prediction">'
                        )

                if img_tags:
                    training_images_html = f"""
                    <div class="training-images">
                        <h4>Training images with same description as prediction and reference:</h4>
                        <div class="training-images-container">
                            {''.join(img_tags)}
                        </div>
                    </div>
                    """
        else:
            # When different, show separate sections
            if train_images_reference:
                ref_img_tags = []
                for train_img_path in train_images_reference:
                    train_img_data = image_to_base64(train_img_path)
                    if train_img_data:
                        ref_img_tags.append(
                            f'<img src="{train_img_data}" class="training-image" alt="Training image matching reference">'
                        )

                if ref_img_tags:
                    training_images_html += f"""
                    <div class="training-images">
                        <h4>Training images with same description as reference:</h4>
                        <div class="training-images-container">
                            {''.join(ref_img_tags)}
                        </div>
                    </div>
                    """

            if train_images_predicted:
                pred_img_tags = []
                for train_img_path in train_images_predicted:
                    train_img_data = image_to_base64(train_img_path)
                    if train_img_data:
                        pred_img_tags.append(
                            f'<img src="{train_img_data}" class="training-image" alt="Training image matching prediction">'
                        )

                if pred_img_tags:
                    training_images_html += f"""
                    <div class="training-images">
                        <h4>Training images with same description as prediction:</h4>
                        <div class="training-images-container">
                            {''.join(pred_img_tags)}
                        </div>
                    </div>
                    """

        # Calculate individual error rates
        if program_report:
            ins, dels, subs, ter = calculate_token_error_rate(reference, predicted)
            error_info = f'Individual TER: {ter:.3f} (I:{ins}, D:{dels}, S:{subs})'
            error_class = 'error-info perfect' if ter == 0.0 else 'error-info'
        else:
            ins, dels, subs, cer = calculate_character_error_rate(reference, predicted)
            error_info = f'Individual CER: {cer:.3f} (I:{ins}, D:{dels}, S:{subs})'
            error_class = 'error-info perfect' if cer == 0.0 else 'error-info'

        item_html = f"""
        <div class="item" id="example-{idx + 1}">
            <div class="item-number">Example {idx + 1}</div>

            <div class="image-container">
                {"<img src='" + main_image_data + "' class='main-image' alt='Kamon image'>" if main_image_data else '<p>Image not found: ' + html.escape(image_path) + '</p>'}
            </div>

            {alignment_html}

            <div class="{error_class}">{error_info}</div>

            {training_images_html}
        </div>
        """

        html_content += item_html

    html_content += """
    </div>
</body>
</html>
"""

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f'HTML visualization generated: {output_file}')
    print(f'Total examples: {metrics.total}')


def main():
    parser = argparse.ArgumentParser(description='Generate HTML visualization for Kamon model outputs')
    parser.add_argument('input_jsonl', nargs='?', default=None, help='Input JSONL file with model outputs')
    parser.add_argument('-o', '--output', default=None, help='Output HTML file (default: visualization.html)')
    parser.add_argument('--model', choices=['vgg', 'vit_decoder'], default='vgg')
    parser.add_argument(
        '--use-translation', action='store_true', help='Use the default inference output path for translation mode.'
    )

    args = parser.parse_args()
    config = resolve_report_config(
        ReportConfig(
            input_jsonl=args.input_jsonl, output=args.output, model=args.model, use_translation=args.use_translation
        )
    )

    if not os.path.exists(config.input_jsonl):
        print(f'Error: Input file {config.input_jsonl} does not exist')
        return

    generate_html_page(config.input_jsonl, config.output)


if __name__ == '__main__':
    main()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('visualize', help='Render an HTML report from JSONL.')
    p.add_argument('input_jsonl', nargs='?', default=None, help='Input JSONL file with model outputs')
    p.add_argument('-o', '--output', default=None)
    p.add_argument('--model', choices=['vgg', 'vit_decoder'], default='vgg')
    p.add_argument(
        '--use-translation', action='store_true', help='Use the default inference output path for translation mode.'
    )
    p.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    config = resolve_report_config(
        ReportConfig(
            input_jsonl=args.input_jsonl, output=args.output, model=args.model, use_translation=args.use_translation
        )
    )
    generate_html_page(config.input_jsonl, config.output)
    return 0
