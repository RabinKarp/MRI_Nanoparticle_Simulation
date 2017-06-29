%%
% Basic script that looks through all files prefixed 'psweep_[TIME].csv'
% within a directory and 
%
% Parameters: dirName - The absolute path of the folder containing the
% simulation outputs and the parameter sweep files - do not add a backslash
% at the end of the folder name
%%
function sweep_data_load(dirName)
    % Get a list of psweep files in the current directory
    dirData = dir([dirName '/psweep_*.csv']);
    
    for i = 1:length(dirData)
        % Read parameter sweep summary file, skip the header row
        summary = table2struct(readtable(dirData(i).name));
        
        % Compute the dimensions of the parameter space
        fields = fieldnames(summary);
        for j = 1:length(fields)
            
        end
        
        % Loop over all files referenced by the summary file

    end
    