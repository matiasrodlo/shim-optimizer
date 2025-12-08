# MRI Shimming Tools

A comprehensive toolkit for magnetic field shimming in MRI, including shim-coil optimization using Biot-Savart law and RF shimming tools.

## About

This repository provides tools and datasets for optimizing magnetic field homogeneity in MRI systems through shim-coil optimization and RF shimming techniques.

### Components

- **`savart-optimizer/`** - 2D shim-coil optimizer that uses the Biot-Savart law to compute magnetic fields from circular shim loops and optimizes loop currents to minimize field variance within a region of interest (ROI). The optimizer works with real B0 field map data from BIDS-formatted datasets.

- **`rf-shimming-7t/`** - Reproducible notebook and tools for RF shimming in the cervical spinal cord at 7T (included as a git submodule from [shimming-toolbox/rf-shimming-7t](https://github.com/shimming-toolbox/rf-shimming-7t)).

- **`dataset/`** - BIDS-formatted MRI dataset containing B0 field maps, T1w, and T2* weighted images acquired for RF shimming research in the cervical spinal cord at 7T.

### Key Features

- **Biot-Savart Field Computation**: Numerically computes Bz field from circular shim loops
- **Real B0 Data Integration**: Loads and uses actual B0 field maps from BIDS datasets
- **Tikhonov Regularization**: Optimizes loop currents to minimize ROI variance with regularization
- **BIDS-Compatible**: Full support for BIDS-formatted neuroimaging datasets
- **Comprehensive Analysis**: Generates field maps, optimization results, and statistical reports

## Quick Start

### Shim-Coil Optimizer

```bash
cd savart-optimizer
pip install -r requirements.txt
./run.sh --subject 01 --acq CP
```

See [`savart-optimizer/README.md`](savart-optimizer/README.md) for detailed usage instructions.

### RF Shimming Tools

The RF shimming tools are available as a submodule. To initialize:

```bash
git submodule update --init --recursive
cd rf-shimming-7t
```

See [`rf-shimming-7t/README.md`](rf-shimming-7t/README.md) for usage instructions.

## Requirements

- Python 3.7+
- NumPy, SciPy, Matplotlib
- Optional: nibabel, scikit-image, pybids (for full BIDS support)

## Citation

If you use this software or dataset, please cite the relevant publications:

- RF Shimming: See [`rf-shimming-7t/paper.md`](rf-shimming-7t/paper.md)
- Dataset: [DOI: 10.18112/openneuro.ds004906](https://openneuro.org/datasets/ds004906)

## License

See individual component licenses:
- `savart-optimizer/` - Check LICENSE file
- `rf-shimming-7t/` - See submodule LICENSE
- `dataset/` - Check dataset_description.json

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Related Projects

- [Shimming Toolbox](https://github.com/shimming-toolbox/shimming-toolbox)
- [Spinal Cord Toolbox](https://spinalcordtoolbox.com)

