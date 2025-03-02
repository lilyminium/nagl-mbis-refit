#!/usr/bin/env python
# coding: utf-8

from collections import Counter
import pathlib

import tqdm
import click
import numpy as np

from openff.units import unit
from openff.toolkit import ForceField, Molecule
from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet

from naglmbis.models import load_charge_model
from naglmbis.models.base_model import ComputePartialPolarised


@click.command()
@click.option(
    "-i", "--input-force-field",
    default="openff-2.0.0.offxml",
    help="Input force field file",
)
@click.option(
    "-o", "--output-force-field",
    default="forcefield/force-field.offxml",
    help="Output force field file",
)
@click.option(
    "-d", "--dataset-file",
    default="targets/phys-prop/training-set.json",
    help="Dataset file",
)
@click.option(
    "-n", "--n-min-components",
    default=1,
    help="Minimum number of components to fit a parameter",
)
def main(
    input_force_field: str = "openff-2.0.0.offxml",
    output_force_field: str = "forcefield/force-field.offxml",
    dataset_file: str = "targets/phys-prop/training-set.json",
    n_min_components: int = 1,
):
    # load models
    gas_model = load_charge_model("nagl-gas-charge-dipole-esp-wb-default")
    water_model = load_charge_model("nagl-water-charge-dipole-esp-wb-default")
    polarised_model = ComputePartialPolarised(
        model_gas = gas_model,
        model_water = water_model,
        alpha = 0.5 #scaling parameter which can be adjusted
    )
    
    # load data and figure out SMILES
    dataset = PhysicalPropertyDataSet.from_json(dataset_file)

    all_smiles = set(
        component.smiles
        for physical_property in dataset.properties
        for component in physical_property.substance.components
    )
    all_smiles -= {"O"}
    print(
        f"Found {len(all_smiles)} unique components "
        f"in {len(dataset.properties)} properties"
    )

    molecules = [Molecule.from_smiles(smi) for smi in all_smiles]

    # load force field
    forcefield = ForceField(input_force_field)
    forcefield.deregister_parameter_handler("ToolkitAM1BCC")

    all_vdW_labels = Counter()
    library_charge_handler = forcefield.get_parameter_handler("LibraryCharges")

    for mol in tqdm.tqdm(
        sorted(molecules, key=lambda x: x.n_atoms),
        desc="Parameterizing charges",
    ):
        smirks = mol.to_smiles(mapped=True)
        charges = polarised_model.compute_polarised_charges(mol.to_rdkit())
        library_charge_handler.add_parameter(
            {
                "smirks": smirks,
                "charge": charges.detach().numpy().flatten() * unit.elementary_charge
            }
        )

        labels = forcefield.label_molecules(mol.to_topology())[0]["vdW"]
        for parameter in labels.values():
            all_vdW_labels[parameter.smirks] += 1


    # check charges are assigned correctly
    for mol in tqdm.tqdm(molecules, desc="Checking assigned charges"):
        ic = forcefield.create_interchange(mol.to_topology())
        assigned_charges = np.array([
            q.m_as(unit.elementary_charge)
            for i, q in sorted(
                ic["Electrostatics"].charges.items(),
                key=lambda x: x[0].atom_indices
            )
        ])
        predicted_charges = polarised_model.compute_polarised_charges(
            mol.to_rdkit()
        ).detach().numpy().flatten()
        if not np.allclose(assigned_charges, predicted_charges):
            print("Assigned by force field: ", assigned_charges)
            print("Predicted by model: ", predicted_charges)
            raise ValueError("Charges not equal")


    # set vdW parameters to parameterize
    vdw_handler = forcefield.get_parameter_handler("vdW")

    parameters_to_fit = []
    for vdw_smirks, n_components in all_vdW_labels.items():
        if n_components < n_min_components:
            continue
        parameter = vdw_handler[vdw_smirks]
        parameter.add_cosmetic_attribute("parameterize", "epsilon, rmin_half")
        parameters_to_fit.append(vdw_smirks)

    print(f"Parameterizing {len(parameters_to_fit)} vdW parameters")

    pathlib.Path(output_force_field).parent.mkdir(parents=True, exist_ok=True)

    forcefield.to_file(
        output_force_field,
        discard_cosmetic_attributes=False,
    )
    print(f"Force field written to {output_force_field}")


if __name__ == "__main__":
    main()
