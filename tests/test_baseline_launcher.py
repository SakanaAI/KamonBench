from pathlib import Path

import pytest
from experiments.baselines import commands, main


@pytest.mark.parametrize(
    'model,augmentations,epochs,ngram',
    [('vit', '3', '50', None), ('vgg-masks', '5', '200', '1'), ('vgg-no-masks', '5', '200', '4')],
)
def test_reported_training_protocol_and_selected_checkpoint(model, augmentations, epochs, ngram):
    train, infer, aggregate, factors = commands(model, 'rm', 20260428, Path('run'), 'cuda:0')

    def value(argv, key):
        return argv[argv.index(key) + 1]

    assert value(train, '--metadata-path') == 'metadata/rm.json'
    assert value(train, '--num-train-augmentations') == augmentations
    assert value(train, '--num-epochs') == epochs
    assert value(train, '--early-stop-patience') == '10'
    assert value(train, '--batch-size') == '64'
    assert '--eval-composites-only' in train
    assert value(train, '--seed') == value(infer, '--seed') == '20260428'
    assert value(infer, '--checkpoint-path') == 'run/checkpoints/checkpoint_best_*.pt'
    if ngram:
        assert value(train, '--ngram-length') == ngram
        assert ('--no-masks' in train) == (model == 'vgg-no-masks')
    assert value(infer, '--output-file') == aggregate[4] == factors[4]


@pytest.mark.parametrize('model,runs', [('vit', 12), ('vgg-masks', 2), ('vgg-no-masks', 2)])
def test_dry_run_covers_published_matrix_without_writing(model, runs, tmp_path, capsys):
    output = tmp_path / 'runs'
    assert main(['--model', model, '--output-dir', str(output), '--dry-run']) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == runs * 4
    assert not output.exists()
