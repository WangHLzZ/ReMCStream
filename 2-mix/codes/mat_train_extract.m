function mat_train_extract(datasetname)

    pyfunction = py.importlib.import_module('train_extract_4matlab');
    py.importlib.reload(pyfunction);
    pyfunction.train_extract(pyargs('input_datasetname', datasetname));

end