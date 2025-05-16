"""
This script combines the performance of the Sage 2.0 and NAGL-MBIS-refit force fields into single dataframes.
It plots comparison scatter plots with RMSEs.

The output files are:

* combined-training-property-values.csv: the combined reference and simulated values for each FF
* combined-training-properties-full.csv: same as above, but with functional group labels
* images/by_hue: comparisons between NAGL-MBIS and Sage, with functional groups separated by hue
* images/separate: comparisons between NAGL-MBIS and Sage, with functional groups separated
"""
import click
import json
import pathlib
import tqdm

from collections import defaultdict
import pandas as pd

from openff.toolkit import Molecule
from openff.evaluator.utils.checkmol import analyse_functional_groups
from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet

import seaborn as sns
from matplotlib import pyplot as plt


def get_limits(subdf, y, x="Reference"):
    """For a particular subset of data plotted on the `x` and `y` axes, get axis limits."""
    min_, max_ = min(subdf[y].values), max(subdf[y].values)
    min2_, max2_ = min(subdf[x].values), max(subdf[x].values)
    min_ = min([min_, min2_])
    max_ = max([max_, max2_])
    range_ = max_ - min_
    inc = range_ / 20
    return min_ - inc, max_ + inc


def plot_scatter(
    melted,
    x="Reference",
    y="Value",
    hue=None,
    col="Force field",
    row="Property type",
    add_rmse: bool = True,
    filename: str = None,
):
    g = sns.FacetGrid(
        data=melted,
        col=col,
        row=row,
        row_order=["Density", "EnthalpyOfMixing"],
        sharex=False,
        sharey=False,
        height=3,
        aspect=1,
        hue=hue,
        margin_titles=True,
    )
    g.map(sns.scatterplot, x, y, s=3)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    for (row_name, col_name), ax in g.axes_dict.items():
        subdf = melted[(melted[row] == row_name) & (melted[col] == col_name)]

        # set limits and plot line
        limits = get_limits(subdf, y)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.plot(limits, limits, ls="--", color="gray", lw=1)

        if add_rmse:
            if hue:
                subdf = subdf[subdf[hue]]
            obs = subdf[y].values
            ref = subdf[x].values
            rmse = ((obs - ref) ** 2).mean() ** 0.5
            ax.text(0.1, 0.9, f"RMSE={rmse:.2e}", transform=ax.transAxes)

    # reset axis labels
    for ax in g.axes[0]:
        ax.set_xlabel("Experiment (g / mL)")
        ax.set_ylabel("Simulated (g / mL)")
    for ax in g.axes[1]:
        ax.set_xlabel("Experiment (kJ / mol)")
        ax.set_ylabel("Simulated (kJ / mol)")

    plt.tight_layout()

    if hue:
        g.add_legend()

    if filename:
        g.savefig(filename, dpi=300)
        print(f"Saved to {filename}")


@click.command()
def main(
    reference_dataset: str = "../fit/sage-2-0-0_tip3p/targets/phys-prop/training-set.json",
    nagl_file: str = "training_per_iteration_nagl.csv",
    sage_file: str = "training_per_iteration_sage.csv",
    exclude_ids: str = "bad-ids.json",
    images_directory: str = "images",
):
    sage_df = pd.read_csv(sage_file, index_col=0)[["Reference", "iter_0015"]]
    sage_df = sage_df.rename(columns={"iter_0015": "Sage 2.0"})
    nagl_df = pd.read_csv(nagl_file, index_col=0)[["iter_0006"]]
    nagl_df = nagl_df.rename(columns={"iter_0006": "NAGL-MBIS-refit"})

    df = pd.merge(sage_df, nagl_df, left_index=True, right_index=True)
    df["Id"] = df.index

    combined_file = "combined-training-property-values.csv"
    df.to_csv(combined_file)
    print(f"Saved combined dataframe to {combined_file}")

    # associate IDs with actual property types and functional groups
    dataset = PhysicalPropertyDataSet.from_json(reference_dataset)

    property_type_ids = defaultdict(list)
    functional_group_ids = defaultdict(list)

    for physprop in tqdm.tqdm(dataset.properties):
        # property type
        pid = int(physprop.id)
        property_type_ids[type(physprop).__name__].append(pid)

        # label compounds with functional groups
        functional_groups = set()
        for component in physprop.substance.components:
            groups = [x.value for x in analyse_functional_groups(component.smiles)]
            functional_groups |= set(groups)
        for group in functional_groups:
            functional_group_ids[group].append(pid)

    # associate it back to dataframe
    property_type_ids = dict(property_type_ids)
    functional_group_ids = dict(functional_group_ids)

    df["Property type"] = ""
    for property_type, ids in property_type_ids.items():
        df["Property type"].loc[ids] = property_type

    for functional_group, ids in functional_group_ids.items():
        # ignore any super rare groups
        if len(ids) < 2:
            continue
        df[functional_group] = False
        df[functional_group].loc[ids] = True

    # add an extra group for everything
    df["All"] = True

    # tidy it up a bit
    ff_cols = ["Sage 2.0", "NAGL-MBIS-refit"]
    id_cols = ["Reference", "Id", "Property type"]
    other_cols = [x for x in df.columns if x not in id_cols + ff_cols]
    melted = df.melt(
        id_vars=id_cols + other_cols,
        value_vars=ff_cols,
        value_name="Value",
        var_name="Force field",
    )
    long_file = "combined-training-properties-full.csv"
    melted.to_csv(long_file)
    print(f"Saved full dataframe to {long_file}")

    # now plot.
    # if excluding IDs, now is the time
    excluded_properties = []
    if exclude_ids:
        with open(exclude_ids, "r") as f:
            contents = json.load(f)
        excluded_properties.extend(sorted(map(int, contents.keys())))
        print(f"Loaded {len(exclude_ids)} IDs to exclude from plotting")

        n_rows = len(melted)
        melted = melted[~melted["Id"].isin(excluded_properties)]
        print(f"Filtered rows in long DF from {n_rows} to {len(melted)}")

    images_directory = pathlib.Path(images_directory)
    images_directory.mkdir(exist_ok=True, parents=True)

    by_hue_directory = images_directory / "by_hue"
    by_hue_directory.mkdir(exist_ok=True, parents=True)
    separate_directory = images_directory / "separate"
    separate_directory.mkdir(exist_ok=True, parents=True)

    for column in tqdm.tqdm(other_cols):
        plot_scatter(
            melted=melted,
            hue=column,
            filename=by_hue_directory / f"comparison_to_sage_{column}.png",
        )

        plot_scatter(
            melted=melted[melted[column]],
            filename=separate_directory / f"comparison_to_sage_{column}.png",
        )


if __name__ == "__main__":
    main()
