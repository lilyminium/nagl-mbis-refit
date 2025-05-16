"""
This script reads the intermediate stages in `optimize.tmp` for a particular force field fit and saves a dataframe of the results, combined with the reference values.
"""

import pathlib

import click
import tqdm
import pandas as pd

from openff.evaluator.client.client import RequestResult
from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet


@click.command()
@click.option(
    "--input",
    "-i",
    "input_directory",
    default="../fit/sage-2-0-0_tip3p",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Directory containing the force field fit",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    default="training_per_iteration_nagl.csv",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    help="File to write with property values per iteration",
)
def main(
    input_directory: str = "../fit/sage-2-0-0_tip3p",
    output_file: str = "training_per_iteration_nagl.csv",
):
    input_directory = pathlib.Path(input_directory)
    iter_directory = input_directory / "optimize.tmp"
    results_files = sorted(iter_directory.glob("phys-prop/iter*/results.json"))

    reference = PhysicalPropertyDataSet.from_json(
        str(input_directory / "targets/phys-prop/training-set.json")
    )

    data_over_iterations = {}
    for prop in reference.properties:
        data_over_iterations[prop.id] = [prop.value.m]

    iter_cols = []
    for result_file in tqdm.tqdm(results_files):
        result = RequestResult.from_json(result_file)
        dataset = result.estimated_properties
        for prop in dataset.properties:
            data_over_iterations[prop.id].append(prop.value.m)
        iter_cols.append(result_file.parent.name)

    df = pd.DataFrame.from_dict(
        data_over_iterations, orient="index", columns=["Reference"] + iter_cols
    )
    df.to_csv(output_file)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
