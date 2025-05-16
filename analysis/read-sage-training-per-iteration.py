"""
This script reads the intermediate stages in `optimize.tmp` for a particular force field fit and saves a dataframe of the results, combined with the reference values.
"""

import pathlib

import tqdm

import pandas as pd

from openff.evaluator.client.client import RequestResult
from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet

from matplotlib import pyplot as plt

import click


@click.command()
@click.option(
    "--input",
    "-i",
    "input_directory",
    default="../fit/vdw-v1",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Directory containing the force field fit",
)
@click.option(
    "--reference",
    "-r",
    "reference_dataset",
    default="../fit/sage-2-0-0_tip3p/targets/phys-prop/training-set.json",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="File containing the NAGL property set for remapping",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    default="training_per_iteration_sage.csv",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    help="File to write with property values per iteration",
)
def main(
    input_directory: str = "../fit/vdw-v1",
    reference_dataset: str = "../fit/sage-2-0-0_tip3p/targets/phys-prop/training-set.json",
    output_file: str = "training_per_iteration_sage.csv",
):
    input_directory = pathlib.Path(input_directory)
    iter_directory = input_directory / "optimize.tmp"
    results_files = sorted(iter_directory.glob("phys-prop/iter*/results.json"))

    reference = PhysicalPropertyDataSet.from_json(
        str(input_directory / "targets/phys-prop/training-set.json")
    )

    # we just need to map property IDs -- numbering was different.
    # we map Sage's to the NAGL fit.
    reference_sage = PhysicalPropertyDataSet.from_json(reference_dataset)
    reference_sage_properties = {
        (prop.thermodynamic_state, prop.substance, prop.value): prop.id
        for prop in reference_sage.properties
    }
    reference_properties = {
        (prop.thermodynamic_state, prop.substance, prop.value): prop.id
        for prop in reference.properties
    }
    old_to_new_ids = {}
    for key, old in reference_properties.items():
        old_to_new_ids[old] = reference_sage_properties[key]

    data_over_iterations = {}
    for prop in reference_sage.properties:
        data_over_iterations[prop.id] = [prop.value.m]

    failed = []
    iter_cols = []
    for result_file in tqdm.tqdm(results_files):
        result = RequestResult.from_json(result_file)
        dataset = result.estimated_properties
        for prop in tqdm.tqdm(dataset.properties):
            key = old_to_new_ids[prop.id]
            try:
                data_over_iterations[key].append(prop.value.m)
            except KeyError:
                failed.append(key)
        iter_cols.append(result_file.parent.name)

    assert len(failed) == 0

    df = pd.DataFrame.from_dict(
        data_over_iterations, orient="index", columns=["Reference"] + iter_cols
    )
    df.to_csv(output_file)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
