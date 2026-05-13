import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='kamonbench')
    sub = parser.add_subparsers(dest='command', required=True)

    from kamonbench.data.croissant import add_parser as add_convert_to_croissant
    from kamonbench.data.install_main_data import add_parser as add_install_main_data
    from kamonbench.data.install_mon_data import add_parser as add_install_mon_data
    from kamonbench.data.recombination_split import add_parser as add_create_recombination_split
    from kamonbench.evaluation.modifier_operators import add_parser as add_summarize_modifier_operators
    from kamonbench.evaluation.predictions import add_parser as add_summarize_predictions
    from kamonbench.evaluation.program_confusion import add_parser as add_summarize_program_confusions
    from kamonbench.evaluation.program_container import add_parser as add_evaluate_program_containers
    from kamonbench.evaluation.program_counterfactual import add_parser as add_summarize_program_counterfactuals
    from kamonbench.evaluation.program_factors import add_parser as add_evaluate_program_factors
    from kamonbench.evaluation.program_modifier import add_parser as add_evaluate_program_modifiers
    from kamonbench.evaluation.program_support import add_parser as add_evaluate_program_support
    from kamonbench.evaluation.representation_probes import add_parser as add_representation_probes
    from kamonbench.evaluation.validation_curve import add_parser as add_summarize_validation_curve
    from kamonbench.generator.cli import add_parser as add_generate
    from kamonbench.inference import add_parser as add_infer
    from kamonbench.reporting.html import add_parser as add_visualize
    from kamonbench.training import add_parser as add_train

    add_convert_to_croissant(sub)
    add_install_main_data(sub)
    add_install_mon_data(sub)
    add_create_recombination_split(sub)
    add_generate(sub)
    add_train(sub)
    add_infer(sub)
    add_summarize_program_counterfactuals(sub)
    add_summarize_predictions(sub)
    add_summarize_program_confusions(sub)
    add_evaluate_program_containers(sub)
    add_evaluate_program_factors(sub)
    add_evaluate_program_modifiers(sub)
    add_evaluate_program_support(sub)
    add_representation_probes(sub)
    add_summarize_modifier_operators(sub)
    add_summarize_validation_curve(sub)
    add_visualize(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args._fn(args))
