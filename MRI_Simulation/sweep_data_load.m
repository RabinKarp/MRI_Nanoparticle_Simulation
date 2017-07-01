%%
% Script that looks through all files prefixed 'psweep_[TIME].csv'
% within a directory, collects the files generated by each simulation
% with a specific configuration of parameters (MNP radius, concentration,
% and cell permeability), and then sums up the magnetization values at each
% time step for each parameter configuration.
%
% Parameters: dirName - The absolute path of the folder containing the
% simulation outputs and the parameter sweep files - do not add a slash
% at the end of the folder name
%
% Returned: A table with 3 columns for independant variables: MNP radius,
% iron concentration, and cell permeability, and for each row a reference
% to a table consisting of a series of equally spaced time points and
% another table with the net magnetization at each of those time points,
% summed over all runs of the simulation on that specific parameter
% configuration.
%
%%
function summary = sweep_data_load(dirName)
    % Get a list of psweep files in the current directory
    dirData = dir([dirName '/psweep_*.csv']);
    summary = struct;
    init = false;
    
    for i = 1:length(dirData)
        % Read parameter sweep summary file, skip the header row
        currentTable = table2struct(readtable(dirData(i).name));
       
        
        % Loop through every output file referenced by this particular
        % summary file
        
        for j = 1:length(currentTable)
            fName = currentTable(j).Filename;
            result = readtable(fName, 'ReadVariableNames',false);
            
            % Take the first column (time column) and the 
            % last (sum magnetization column, and store them
            % in the current summary table
            
            currentTable(j).tValues = table2array(result(:, 1));
            currentTable(j).netMag = table2array(result(:, width(result)));
        end
        
        % Delete the filename column from the table
        rmfield(currentTable, 'Filename')
        
        % If grand summary table not initialized, initialize
        % it as the current table
        if (~init)
            init = true;
            summary = currentTable;
        % Otherwise, find the corresponding row of the grand table
        % for each row in the current summary table, and add the
        % net magnetization column.
        else
            for j = 1:length(currentTable)
                summary(j).netMag = summary(j).netMag + currentTable(j).netMag;
            end
        end
    end
    return;
end