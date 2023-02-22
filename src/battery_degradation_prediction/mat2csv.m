function dummy = mat2csv(filepath, output_name)
    S=load(filepath);
    csv_file = fopen(output_name, 'wt');
    cHeader = {'cycle' 'type' 'ambient_temp' 'voltage_measured' 'current_measured' 'temperatrue_measured' 'current_charge' 'voltage_charge' 'datetime' 'capacity'};
    commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
    commaHeader = commaHeader(:)';
    textHeader = cell2mat(commaHeader); %cHeader in text with commas
    %write header to file
    fprintf(csv_file,'%s\n',textHeader);
    
    cycle_cell = struct2cell(S.B0005.cycle);
    num_data = size(cycle_cell, 3);
    num_cycle_charge = 0;
    num_cycle_discharge = 0;
    num_data = 150; % too many data
    for row = 1:num_data
        if mod(row, 10) == 0
            disp("Current Row/Total Row = " + string(row)+"/"+string(num_data))
        end
        row_information = cycle_cell(:,:,row);  % The first element refers to columns, the second one doesn't have any meaning, and the third one refers to row
        ambient_temp = row_information{2};
        if strcmp(row_information{1}, "charge")
            num_cycle_charge = num_cycle_charge + 1;
            write_charging_data(csv_file, num_cycle_charge, ambient_temp, row_information);
        elseif strcmp(row_information{1}, "discharge")
            num_cycle_discharge = num_cycle_discharge + 1;
            write_discharging_data(csv_file, num_cycle_discharge, ambient_temp, row_information);
        end
    end
    fclose(csv_file);
end

