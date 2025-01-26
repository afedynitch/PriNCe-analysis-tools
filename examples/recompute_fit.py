from os import path


def setup_fit():
    """Setup function is executed at the start of each job
    The return value is passed to single_run for each index
    """
    from os import path

    filepath = path.join(config["targetdir"], config["project_tag"], "collected.hdf5")
    input_spec = config["input_spec"]
    paramlist = config["paramlist"]

    from analyzer.plotter import ScanPlotter

    scan = ScanPlotter(filepath, input_spec, paramlist)

    scan.print_summary()
    return scan


def single_fit(setup, index):
    """Single run function is executed for each index
    Every job will loop over a subset of all indices and call this function
    The list of outputs is then stored in .out
    """
    scan = setup

    # Note
    m, opt = scan.recompute_fit(
        index, minimizer_args={"fix_deltaE": False}, Emin=6e9, xmax_model="sibyll"
    )
    print("chi2:", opt.get_chi2_spectrum(), opt.get_chi2_Xmax(), opt.get_chi2_VarXmax())
    mindetail = m.parameters, m.args, m.values, m.errors
    return m.fval, mindetail


# The base path assumes that this files is in the folder of the project created by example_create_project.py
from run import config

base = path.abspath(__file__)
# set config for jobs options below
# The project will loop over all index combinations in 'paramlist'
# Each job will receive an equal subset of indices to compute
config.update(
    {
        "fit_tag": "floating_E_Sibyll",
        "fit_only": True,
        "inputpath": base,
        "setup_func": setup_fit,
        "single_run_func": single_fit,
        "njobs": 120,
        "hours per job": 4,
        "max memory GB": 3,
    }
)

# run this script as python example_create_project.py -[options]
# PropagationProject.run_from_terminal() for all options
if __name__ == "__main__":
    # Parse the run arguments
    from analyzer.cluster import PropagationProject

    project = PropagationProject(config)
    project.run_from_terminal()
