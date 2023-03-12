def make_training_set(dir_path : str) -> pd.DataFrame:
    """The function finds all of the .csv files in the given directory and loads them
    into one Pandas DataFrame
    Parameters
    --------------------------------
    dir_path : str
      A string object that contains the absolute path that the user has copy/pasted from
      their computer
    Returns
    --------------------------------
    training_set : pd.DataFrame
      All csv files as processed data frames with a column including the battery it is associated with
    """

    f_list = os.listdir(dir_path)

    #path string list

    dir_string = []

    #list of file names in directory

    fn_list = []

    #list of file paths

    path_list = []

    #list of battery names to make name column

    name_list = []

    #list of dataframes from each csv file

    df_list = []

    #empty list of processed data frames to be merged

    processed_list = []

    #1st loop makes list of all file paths, makes a list of all battery names DONE
    for k in f_list:
        if k.endswith(".csv"):
            fn_list.append(k)
            k = Path(k)
            name_list.append((k.with_suffix('')))

    #2nd loop makes list of full absolute paths by joining the file name to the dir_path DONE

    for f in fn_list:
        print(f)
        full_path = os.path.join(dir_path, f)
        path_list.append(full_path)

    #creates list of dataframes to be processed, adds Battery name row DONE
    for p in path_list:
        p = str(p)
        df = pd.read_csv(p)
        df_list.append(df)

        for n in name_list:
                df['Battery'] = n

    for i in df_list:
        i["time"] = i["datetime"].apply(convert_datetime_str_to_obj)
#####
        i["elapsed_time"] = i["time"].apply(calc_test_time_from_datetime, args=(i["time"].iloc[0],))
#####
        df_discharge = isolate_discharge_cyc_data(i)
#####
        time_elapsed_list = add_elapsed_time_per_cycle(df_discharge)
#####
        df_discharge.insert(len(df_discharge.columns), "elapsed_time_per_cycle", time_elapsed_list)
        df_discharge.reset_index(drop=True, inplace=True)
#####
        capacity_during_discharge = calc_capacity_during_discharge(df_discharge)

        df_discharge.insert(len(df_discharge.columns), "capacity_during_discharge", capacity_during_discharge)
 ######
        remove_jump_voltage(df_discharge)

        processed_list.append(df_discharge)

    training_set = pd.concat(processed_list)

    return training_set

#Edition 03/11/23
