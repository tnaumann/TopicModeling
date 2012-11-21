matlabpool open local 8

NUM_TOPICS = 50:50:200;
dataStore = [];
dataKey = [];

parfor t = 1:4
    t
    numT = NUM_TOPICS(t);  

    matrix = testFunc(numT);
    matrixPad = ones(10000, 200)*-1;
    matrixPad(:, 1:numT) = matrix;
    dataStore = [matrixPad; dataStore];
  	dataKey = [t; dataKey];
end
