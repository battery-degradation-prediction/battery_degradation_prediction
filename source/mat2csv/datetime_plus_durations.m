function new_datetimes = datetime_plus_durations(current_datetime, durations)
    % Given a series of durations (unit in second), convert them to
    % date_time.
    %
    % Parameters
    % ----------
    % durations : matrix
    %     durations = [t0, t1, t2, ...] unit = [second]
    % current_datetime : datetime 
    %     DD-MM-YYYY HH:MM:SS
    %
    % Returns
    % -------
    % new_datetime : datetime
    %    current date time + duration, DD-MM-YYYY HH:MM:SS
    num_data = length(durations);
    num_char = length('yyyy-mm-dd-HH-MM-SS');
    new_datetimes = datestr(zeros(3,1), 'yyyy-mm-dd-HH-MM-SS');
    for i = 1:num_data
        new_datetimes(i, :) = datestr(current_datetime + seconds(durations(i)), 'yyyy-mm-dd-HH-MM-SS'); 
    end
end