function A = readNPY_matlab(filename)
    np = py.importlib.import_module("numpy");
    pyArray = np.load(filename);

    shape = cellfun(@double, cell(pyArray.shape));
    data = double(py.array.array('d', py.numpy.nditer(pyArray)));

    if isempty(shape)
        A = data;                 % scalar
    elseif numel(shape) == 1
        A = reshape(data, shape(1), 1);   % 1D vector
    else
        A = reshape(data, fliplr(shape));
        A = permute(A, numel(shape):-1:1);
    end
end