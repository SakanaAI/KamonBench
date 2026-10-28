import argparse
import importlib
import sys

# Import only the selected command, so reconstruction does not require torch.
COMMANDS = {
    'data.reconstruct': ('reconstruct', 'verify-data'),
    'data.install_mon_data': ('install-mon-data',),
    'data.convert': ('convert-to-metadata',),
    'data.recombination_split': ('create-recombination-split',),
    'generator.cli': ('generate',),
    'training': ('train',),
    'inference': ('infer',),
    'evaluation.program_counterfactual': ('summarize-program-counterfactuals',),
    'evaluation.predictions': ('summarize-predictions',),
    'evaluation.program_confusion': ('summarize-program-confusions',),
    'evaluation.program_factors': ('evaluate-program-factors',),
    'evaluation.representation_probes': (
        'extract-representation-features',
        'train-representation-probes',
        'compare-representation-probes',
        'compare-representation-probe-support',
    ),
    'reporting.html': ('visualize',),
}


def _build_parser(command: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='kamonbench')
    sub = parser.add_subparsers(dest='command', required=True)
    for module, commands in COMMANDS.items():
        if command in commands:
            importlib.import_module(f'kamonbench.{module}').add_parser(sub)
        else:
            for name in commands:
                sub.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = _build_parser(argv[0] if argv else None)
    args = parser.parse_args(argv)
    try:
        return int(args._fn(args))
    except (OSError, ValueError) as error:
        parser.exit(1, f'kamonbench: {error}\n')
