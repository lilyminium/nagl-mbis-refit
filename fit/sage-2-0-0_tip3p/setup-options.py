import click
import pathlib

from openff.evaluator import unit
from forcebalance.evaluator_io import Evaluator_SMIRNOFF


@click.command()
@click.option(
    "--port",
    type=int,
    default=8000,
    help="The port to run the server on",
)
@click.option(
    "--output-file",
    type=str,
    default="targets/phys-prop/options.json",
    help="The output file for the options",
)
def main(
    port: int = 8000,
    output_file: str = "targets/phys-prop/options.json",
):
    options_file = Evaluator_SMIRNOFF.OptionsFile()
    options_file.connection_options.server_address = "localhost"
    options_file.connection_options.server_port = port

    # barebones options
    options_file.estimation_options.calculation_layers = ["SimulationLayer"]

    options_file.data_set_path = "training-set.json"

    # hardcode weights to be equal
    options_file.weights = {
        "Density": 1.0,
        "EnthalpyOfMixing": 1.0,
    }

    # hardcode denominators from Sage
    options_file.denominators = {
        "Density": 0.05 * unit.gram / unit.milliliter,
        "EnthalpyOfMixing": 1.6 * unit.kilojoule / unit.mole,
    }

    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as file:
        file.write(options_file.to_json())


if __name__ == "__main__":
    main()
