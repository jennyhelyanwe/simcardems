from simcardems.postprocess import make_xdmffiles

make_xdmffiles("results_4mm/biv_coarse_run_output/results.h5", names=["u", "Ta", "lambda", "XS", "XW"])