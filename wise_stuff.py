#!/usr/bin/env python
# coding: utf-8

# Import packages
import os
import pandas as pd
import sqlite3
import pickle
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
# Set option to display more rows in the DataFrame for debugging
pd.set_option('display.max_rows', 1000)

###########################################################################333
# functions
# add oddball regions
def add_region(df, **kwargs):
    # Add the new row to the DataFrame using the provided kwargs
    df.loc[len(df)] = kwargs

    # ICRS                                                                                                                                                                                                                                
    galactic_coord = SkyCoord(l=kwargs['GLong'], b=kwargs['GLat'], frame='galactic', unit=(u.deg, u.deg))
    icrs_coord = galactic_coord.transform_to('icrs')
    df['RA (J2000)'] = icrs_coord.ra.value
    df['Dec (J2000)'] = icrs_coord.dec.value
    df['GName'] = generate_gname(df['GLong'], df['GLat'])

# gname                                                                                                                                                                                                                             
def generate_gname(glong, glat):
    glong_str = glong.apply(lambda x: f"{np.floor(x * 1000) / 1000.:07.3f}")
    glat_str = glat.apply(lambda x: f"{np.floor(x * 1000) / 1000.:+07.3f}")

    # Combine them into the desired format                                                                                                                                                                                            
    return 'G' + glong_str + glat_str

###########################################################################333
# Make db files.  First WISE, then RRL_Surveys, then multiple velocities, then KDARs

# WISE df
version = 3.0
str_version = f"{version:.1f}"
dir_path = './'
csv_file = f"{dir_path}wise_hii_master_V{str_version}.csv"

# Read the CSV file into a pandas DataFrame
WISE_df = pd.read_csv(csv_file)

# Get rid of NaNs at the end of the file
WISE_df = WISE_df.dropna(axis=1, how='all')

# Fix groups
WISE_df.loc[(WISE_df['Catalog'] == 'known') & (WISE_df['Group_Flag'] == 1), 'Catalog'] = 'group'

# Make GName
WISE_df['GName'] = generate_gname(WISE_df['GLong'], WISE_df['GLat'])

# Make RA and Dec
galactic_coord = SkyCoord(l=WISE_df['GLong'], b=WISE_df['GLat'], frame='galactic', unit=(u.deg, u.deg))
icrs_coord = galactic_coord.transform_to('icrs')
WISE_df['RA (J2000)'] = icrs_coord.ra.value
WISE_df['Dec (J2000)'] = icrs_coord.dec.value



# RRL_Surveys df
pkl_directory = 'rrl_surveys/'

# List of columns to keep
rrl_columns = ['Name', 'GName', 'GLong', 'GLat', 'RA (J2000)', 'Dec (J2000)', 'RMS', 'Te', 'e_Te', 'TL', 'e_TL', 'FWHM', 'e_FWHM', 
               'VLSR', 'e_VLSR', 'VLSR_He', 'e_VLSR_He', 'FWHM_He', 'e_FWHM_He', 
               'TL_He', 'e_TL_He', 'VLSR_C', 'e_VLSR_C', 'FWHM_C', 'e_FWHM_C', 'TL_C', 'e_TL_C', 
               'Telescope', 'Resolution', 'Wavelength', 'Frequency', 
               'Author', 'Year', 'KDAR', 'DMethod', 'DAuthor'
              ]

# Load each .pkl file into the dataframe
RRL_Surveys_df = pd.DataFrame()
for filename in os.listdir(pkl_directory):
    if filename.endswith('.pkl') and not filename.endswith('_2.pkl'):
        file_path = os.path.join(pkl_directory, filename)
        
        # Load the .pkl file into a pandas DataFrame
        df = pd.read_pickle(file_path)
        df_filtered = df[rrl_columns]

        # Remove rows where 'VLSR', 'FWHM', or 'TL' are NaN, None, or 0
        #df_filtered = df_filtered[~df_filtered[['VLSR', 'FWHM', 'TL']].isin([None, 0]).any(axis=1) & 
        #                          df_filtered[['VLSR', 'FWHM', 'TL']].notna().all(axis=1)]


        # Remove any columns with all NaN or empty entries before concatenation
        # Removing e_Te column.  Causes error later
        #df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
        RRL_Surveys_df = pd.concat([RRL_Surveys_df, df_filtered], ignore_index=True)

# Deal with multiple RRLs from individual observations
# Sort the DataFrame by 'GLong' and 'GLat' to ensure duplicates are adjacent
RRL_Surveys_df = RRL_Surveys_df.sort_values(by=['GLong', 'GLat', 'TL'], ascending = [True, True, False])

# In cases of the same glong and glat (to three decimal places), store additional lines in new variables
RRL_Surveys_df['VLSR2'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['VLSR'].shift(-1)
RRL_Surveys_df['e_VLSR2'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['e_VLSR'].shift(-1)
RRL_Surveys_df['FWHM2'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['FWHM'].shift(-1)
RRL_Surveys_df['e_FWHM2'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['e_FWHM'].shift(-1)
RRL_Surveys_df['TL2'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['TL'].shift(-1)
RRL_Surveys_df['e_TL2'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['e_TL'].shift(-1)
RRL_Surveys_df['VLSR3'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['VLSR'].shift(-2)
RRL_Surveys_df['e_VLSR3'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['e_VLSR'].shift(-2)
RRL_Surveys_df['FWHM3'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['FWHM'].shift(-2)
RRL_Surveys_df['e_FWHM3'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['e_FWHM'].shift(-2)
RRL_Surveys_df['TL3'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['TL'].shift(-2)
RRL_Surveys_df['e_TL3'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['e_TL'].shift(-2)
RRL_Surveys_df['VLSR4'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['VLSR'].shift(-3)
RRL_Surveys_df['e_VLSR4'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['e_VLSR'].shift(-3)
RRL_Surveys_df['FWHM4'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['FWHM'].shift(-3)
RRL_Surveys_df['e_FWHM4'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['e_FWHM'].shift(-3)
RRL_Surveys_df['TL4'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['TL'].shift(-3)
RRL_Surveys_df['e_TL4'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['e_TL'].shift(-3)

# Remove duplicate entries that arise in the cases of multiple lines
# First, get the duplicate rows based on 'GLong' and 'GLat'
duplicates = RRL_Surveys_df.duplicated(subset=['GLong', 'GLat'], keep='first')

# For rows with matching 'GLong' and 'GLat', store the shortest name
RRL_Surveys_df['Name'] = RRL_Surveys_df.groupby(['GLong', 'GLat'])['Name'].transform(lambda x: x.loc[x.str.len().idxmin()])

# Remove the duplicate rows (keep the first occurrence)
RRL_Surveys_df = RRL_Surveys_df[~duplicates]

# Reset the index after dropping rows
RRL_Surveys_df = RRL_Surveys_df.reset_index(drop=True)
for i in range(len(RRL_Surveys_df)):
    print(i, RRL_Surveys_df['GLong'][i], RRL_Surveys_df['GLat'][i], RRL_Surveys_df['Name'][i], RRL_Surveys_df['TL'][i])

# Now deal with the rare instances of when the same source was observed twice, with the same name, in different surveys
# Sort the dataframe by 'Year' and 'GLong'
RRL_Surveys_df_sorted = RRL_Surveys_df.sort_values(by=['Year', 'GLong'], ascending=[False, True])

# Drop duplicates, keeping the first (highest 'Year') for each 'Name'
RRL_Surveys_df_unique = RRL_Surveys_df_sorted.drop_duplicates(subset='Name', keep='first')

# Sort back into correct order
RRL_Surveys_df = RRL_Surveys_df_unique.sort_values(by=['GLong', 'GLat'])

# Find the rows that were dropped (those in sorted but not in unique)
dropped_rows = RRL_Surveys_df_sorted.loc[~RRL_Surveys_df_sorted.index.isin(RRL_Surveys_df_unique.index)]

# Show the dropped rows
print("Rows that were dropped:")
print(dropped_rows[['Name', 'VLSR', 'Author']])

# These three will be missing Te values.  Reobserved in HRDS.  Should fix
#1724             S128     Balser et al. (2011)
#1784             S168     Balser et al. (2011)
#1880             S257     Balser et al. (2011)

# Filter rows where the 'TL' column is NaN
nan_TL_rows = RRL_Surveys_df[RRL_Surveys_df['TL'].isna()]

# Print the 'Author' column for those rows
print('These have missing line parameters')
print(nan_TL_rows['Author'])

# Other regions
# Update for 'G000.394-00.540'
add_region(RRL_Surveys_df,  
            VLSR=24.0, FWHM=39.0, Telescope='Effelsberg', Resolution=2.5, 
            Wavelength=6, Frequency=5, Lines='H110', 
            GLong=0.394, GLat=-0.540, 
            Name='G000.394-00.540', Author='Downes et al. (1980)')

# Update for 'G000.489-00.668'
add_region(RRL_Surveys_df, 
            VLSR=17.5, FWHM=24.0, Telescope='Effelsberg', Resolution=2.5, 
            Wavelength=6, Frequency=5, Lines='H110', 
            GLong=0.510, GLat=-0.051, Name='G000.510-00.051', Author='Downes et al. (1980)')

# Update for 'G000.572-00.628'
add_region(RRL_Surveys_df, 
            VLSR=20.0, FWHM=15.0, Telescope='Effelsberg', Resolution=2.5, 
            Wavelength=6, Frequency=5, Lines='H110', 
            GLong=0.572, GLat=-0.628, Name='G000.572-00.628', Author='Downes et al. (1980)')

# Update for 'G001.149-00.062'
add_region(RRL_Surveys_df, 
            VLSR=-17.0, FWHM=20.0, Telescope='Effelsberg', Resolution=2.5, 
            Wavelength=6, Frequency=5, Lines='H110', 
            GLong=1.149, GLat=-0.062, Name='G001.149-00.062', Author='Downes et al. (1980)')

# Update for 'G000.361-00.780;S20'
add_region(RRL_Surveys_df, 
            VLSR=20.0, FWHM=23.0, Telescope='Effelsberg', Resolution=2.5, 
            Wavelength=6, Frequency=5, Lines='H110', 
            GLong=0.361, GLat=-0.780, Name='G00.361-00.780;S20', Author='Downes et al. (1980)')

# Update for 'G359.277-00.264'
add_region(RRL_Surveys_df,
            VLSR=-2.4, e_VLSR=0.6, FWHM=20.1, e_FWHM=1.5, Telescope='Effelsberg', 
            Resolution=0.8, Wavelength=2, Frequency=15, Lines='H76', 
            GLong=359.277, GLat=-0.264, Name='G359.277-00.264', Author='Wink, Altenhoff, & Metzger (1982)')

# Update for 'G268.034-00.984'
add_region(RRL_Surveys_df,
            VLSR=1.8, e_VLSR=0.9, FWHM=36.5, e_FWHM=1.2, Telescope='Parkes', 
            Resolution=4.4, Wavelength=6, Frequency=5, Lines='H109', 
            GLong=268.034, GLat=-0.984, Name='G268.034-00.984', Author='Wilson et al. (1970)')

# Update for 'G287.782-00.819'
add_region(RRL_Surveys_df,
            VLSR=-21.3, e_VLSR=0.8, FWHM=38.8, e_FWHM=6.0, Telescope='Parkes', 
            Resolution=4.4, Wavelength=6, Frequency=5, Lines='H109', 
            GLong=287.782, GLat=-0.819, Name='G287.782-00.819', Author='Wilson et al. (1970)')

# Update for 'G333.605-00.095'
add_region(RRL_Surveys_df,
            VLSR=-53.7, e_VLSR=3.1, FWHM=44.7, e_FWHM=6.0, Telescope='Parkes', 
            Resolution=4.4, Wavelength=6, Frequency=5, Lines='H109', 
            GLong=333.605, GLat=-0.095, Name='G333.605-00.095', Author='Wilson et al. (1970)')

# Update for 'WB43'
add_region(RRL_Surveys_df,
            VLSR=-0.8, e_VLSR=0.1, FWHM=15.3, e_FWHM=0.2, Telescope='GBT', 
            Resolution=2.5, Wavelength=6, Frequency=5, Lines='H103-H109', 
            GLong=92.668, GLat=3.069, Name='WB43', Author='Arvidsson, Kerton, & Foster (2009)')

# Update for 'Galactic Center Lobe'
# guessed on errors
add_region(RRL_Surveys_df,
            catalog='known', VLSR=2.4, e_VLSR=0.1, FWHM=20.5, e_FWHM=5, TL=33.2*0.001, e_TL = 5*0.001,
            Telescope='GBT', Resolution=2.65, Wavelength=6, Frequency=5, Lines='H95-H117', 
            GLong=359.555, GLat=-0.040, Name='G359.555-00.040', Author='Anderson et al. (2024)')

# Update for 'NR038;SHRDS078;G267.9-1.1'
add_region(RRL_Surveys_df,
            catalog='known', VLSR=3.0, e_VLSR=3.4, FWHM=35.1, e_FWHM=4.3, 
            Telescope='ATCA', Resolution=None, Wavelength=6, Frequency=5, Lines='H107', 
            GLong=267.942, GLat=-1.061, Name='G267.9-1.1', Author='Misanovic, Cram, & Green (2002)')

# Update for 'ATCA352;SHRDS476;G313.8+0.7'
add_region(RRL_Surveys_df,
            catalog='known', VLSR=-53.4, e_VLSR=5.2, FWHM=26.5, e_FWHM=5.2, 
            Telescope='ATCA', Resolution=None, Wavelength=6, Frequency=5, Lines='H107', 
            GLong=313.788, GLat=0.712, Name='G313.8+0.7', Author='Misanovic, Cram, & Green (2002)')

# Update for 'FA454'
add_region(RRL_Surveys_df,
            catalog='known', VLSR=-217.0, e_VLSR=0.09, FWHM=35.6, e_FWHM=0.22, 
            TL=17.1, e_TL=0.09, Telescope='GBT', Resolution=1.3, Wavelength=3, 
            Frequency=9, Lines='H87-H93', GLong=358.517, GLat=0.036, 
            Name='G358.517+0.036', Author='Anderson et al. (2020)')

# Update for 'FA461'
add_region(RRL_Surveys_df,
            catalog='known', VLSR=-206.9, e_VLSR=0.10, FWHM=33.9, e_FWHM=0.23, 
            TL=20.8, e_TL=0.12, Telescope='GBT', Resolution=1.3, Wavelength=3, 
            Frequency=9, Lines='H87-H93', GLong=358.796, GLat=0.001, 
            Name='G358.796+0.001', Author='Anderson et al. (2020)')

# Update for 'GS113'
add_region(RRL_Surveys_df,
            catalog='known', VLSR=-209.3, e_VLSR=0.22, FWHM=24.3, e_FWHM=0.28, 
            TL=5.8, e_TL=0.04, Telescope='GBT', Resolution=1.3, Wavelength=3, 
            Frequency=9, Lines='H87-H93', GLong=358.844, GLat=0.026, 
            Name='G358.844+0.026', Author='Anderson et al. (2020)')


# Multiple Velocities df
# Directory containing the .pkl files
pkl_directory = 'multiple_velocities/'

# Column names for the final DataFrame
multvel_columns = ["Name", "GName", "GLong", "GLat", "RA (J2000)", "Dec (J2000)", "Author", "Year", "Real_VLSR"]

# Create an empty DataFrame to hold all data
Multiple_Velocities_df = pd.DataFrame()

# Load each .pkl file into the dataframe
for filename in os.listdir(pkl_directory):
    if filename.endswith('.pkl') and not filename.endswith('_2.pkl'):
        file_path = os.path.join(pkl_directory, filename)
        
        # Load the .pkl file into a pandas DataFrame
        df = pd.read_pickle(file_path)
        df_filtered = df[multvel_columns]
        Multiple_Velocities_df = pd.concat([Multiple_Velocities_df, df_filtered], ignore_index=True)


# KDAR df
# Directory containing the .pkl files
kdars_directory = 'kdars/'

# Column names for the final DataFrame
kdar_columns = ["Name", "GName", "GLong", "GLat", "RA (J2000)", "Dec (J2000)", "DAuthor", "DMethod", "Year", "KDAR"]

# Create an empty DataFrame to hold all data
KDARs_df = pd.DataFrame()

# Load each .pkl file into the dataframe
for filename in os.listdir(pkl_directory):
    if filename.endswith('.pkl') and not filename.endswith('_2.pkl'):
        file_path = os.path.join(kdars_directory, filename)
        
        # Load the .pkl file into a pandas DataFrame
        df = pd.read_pickle(file_path)
        df_filtered = df[kdar_columns]
        KDARs_df = pd.concat([KDARs_df, df_filtered], ignore_index=True)


# Fluxes df
fluxes_directory = 'fluxes/'
IR_df = pd.read_csv(fluxes_directory + 'ir_fphot_NEW.csv')
VGPS_df = pd.read_csv(fluxes_directory + 'vgps_fphot_NEW.csv')
MAGPIS_df = pd.read_csv(fluxes_directory + 'magpis_fphot_NEW.csv')
Fluxes_df = pd.merge(pd.merge(IR_df, VGPS_df, on='GName', how='left'), MAGPIS_df, on='GName', how='left')

###########################################################################333
# Matching
# Split the 'Name' column in WISE_df by semicolons to handle multiple names
WISE_df['Name_Split'] = WISE_df['Name'].apply(lambda x: x.split(';') if isinstance(x, str) else [])

# Initialize an empty list to store matched rows with the most recent year
matched_rows = []

# Loop through each row in WISE_df
for index, row in WISE_df.iterrows():
    matched_name = None  # This will store the name with the highest year
    matched_row = None  # This will store the entire row from RRL_Surveys_df_unique that matches
    highest_year = 0
    
    # Loop through each name in the 'Name_Split' for this row
    for name in row['Name_Split']:
        # Check if the name is in RRL_Surveys_df
        matches = RRL_Surveys_df[RRL_Surveys_df['Name'] == name]
        
        # Take most recent year
        for _, match in matches.iterrows():
            if match['Year'] > highest_year:  # If the Year is higher, update the values
                highest_year = match['Year']
                matched_name = name
                matched_row = match
    
    # If there was a match, append the information to matched_rows
    if matched_row is not None:
        matched_row_info = matched_row.to_dict()  # Convert the matched row to a dictionary
        matched_row_info['Matched_Name'] = matched_name  # Store the matched name with highest year
        matched_row_info['WISE_Idx'] = index  # Store the WISE index for later merging
        matched_rows.append(matched_row_info)

# Convert the matched rows into a DataFrame
matched_df = pd.DataFrame(matched_rows)

# Merge it back with the WISE_df to retain all original columns
WISE_Matched_df = pd.merge(WISE_df, matched_df, left_index=True, right_on='WISE_Idx', 
                           how='left', suffixes=['', '_Observed']).reset_index(drop=True)


# Store real velocity
matched_rows = []

# Iterate over each row in the WISE_Matched_df
for _, row in WISE_Matched_df.iterrows():
    # Find matching rows in Multiple_Velocities_df where 'Name' is in 'Name_Split'
    matching_rows = Multiple_Velocities_df[Multiple_Velocities_df['Name'].isin(row['Name_Split'])]

    if not matching_rows.empty:
        # Iterate over the columns of matching_rows
        for col in matching_rows.columns:
            if col != 'Real_VLSR':  # Avoid modifying the 'Real_VLSR' column
                new_col_name = f'{col}_Multiple'
            else:
                new_col_name = f'{col}'
            
            # Assign the value from the matching row to the current row
            row[new_col_name] = matching_rows[col].values[0]  # Take the first match if there's a match
    
    # Append the updated row to the list
    matched_rows.append(row)

# Convert the list of updated rows back to a DataFrame
WISE_Matched_df = pd.DataFrame(matched_rows)


# KDARs
matched_rows = []

# Iterate over each row in the WISE_Matched_df
for _, row in WISE_Matched_df.iterrows():
    # Find matching rows in Multiple_Velocities_df where 'Name' is in 'Name_Split'
    matching_rows = KDARs_df[KDARs_df['Name'].isin(row['Name_Split'])]

    if not matching_rows.empty:
        # Iterate over the columns of matching_rows
        for col in matching_rows.columns:
            if col != 'KDAR':  # Avoid modifying the 'Real_VLSR' column
                new_col_name = f'{col}_KDAR'
            else:
                new_col_name = f'{col}'
            
            # Assign the value from the matching row to the current row
            row[new_col_name] = matching_rows[col].values[0]  # Take the first match if there's a match

    # Append the updated row to the list
    matched_rows.append(row)

# Convert the list of updated rows back to a DataFrame
WISE_Matched_df = pd.DataFrame(matched_rows)


# Fluxes
WISE_Matched_df = pd.merge(WISE_Matched_df, Fluxes_df, on='GName', how='left', suffixes=('', '_Flux'))


# Add Spectra
spectra_directory = 'spectra/'

# Define a dictionary to map subdirectories to authors
subdirectory_to_author = {
    'hrds': 'Anderson et al. (2011)',
    'hrds_diffuse': 'Anderson et al. (2017)',
    'hrds_multvel': 'Anderson et al. (2015b)',
    'hrds_wise': 'Anderson et al. (2015a)'
}

# initialize objects
WISE_Matched_df['x'] = None
WISE_Matched_df['y'] = None

# Loop through all the subdirectories in the root directory
for survey in os.listdir(spectra_directory):
    survey_path = os.path.join(spectra_directory, survey)
    
    # Check if the current item is a subdirectory
    if os.path.isdir(survey_path):
        # Loop through all the text files in the subdirectory
        for filename in os.listdir(survey_path):
            if filename.endswith("_output.txt"):  # Check for .txt files
                name = filename.split('_')[0]
                
                # Check if the name matches any value in the 'Name_Split' column
                for index, row in WISE_Matched_df.iterrows():
                    # Check if 'name' is in the list of names in the 'Name_Split' column for the current row
                    if isinstance(row['Name_Split'], list) and name in row['Name_Split']:
                        # Read the x and y data from the text file
                        file_path = os.path.join(survey_path, filename)
                        data = pd.read_csv(file_path, delimiter=',', skiprows=1, names=['x', 'y'])
                        
                        # Add the x and y values to the matching row
                        WISE_Matched_df.at[index, 'x'] = data['x'].values*u.km/u.s
                        WISE_Matched_df.at[index, 'y'] = data['y'].values/1000 * u.K

                        # Add the author column
                        WISE_Matched_df.loc[index, 'Author_Specrum'] = subdirectory_to_author.get(survey)


###########################################################################333
# Various Fixes and Checks

# Set group catalog
WISE_Matched_df.loc[(WISE_df['Catalog'] == 'observe') & (WISE_Matched_df['Group_Flag'] == 1), 'Catalog'] = 'group'

# Check that all associated sources have a group designation
print(len(WISE_Matched_df[(WISE_Matched_df['Group_Flag'] == 1) & (WISE_Matched_df['Group'] == '')]), ' sources lacking a group name.')

# Add group velocity and distance parameters
WISE_Matched_df['Group_VLSR'] = None
WISE_Matched_df['Group_e_VLSR'] = None
WISE_Matched_df['Group_KDAR'] = None

# Iterate over the rows of the DataFrame and store the group information
print("Sources whose group name doesn't match:")
for i, row in WISE_Matched_df.iterrows():
    group = row['Group']
    if group == group:
        matching_rows = WISE_Matched_df[WISE_Matched_df['Name'] == group]
        
        if not matching_rows.empty:
            # Extract the matching row
            match_row = matching_rows.iloc[0]

            # Assign the values from the matching row
            WISE_Matched_df.at[i, 'Group_VLSR'] = match_row['VLSR']
            WISE_Matched_df.at[i, 'Group_e_VLSR'] = match_row['e_VLSR']
            WISE_Matched_df.at[i, 'Group_KDAR'] = match_row['KDAR']
        else:
            print(i, group)

###########################################################################################################################
# radio continuum detectuions
WISE_Matched_df.loc[(WISE_Matched_df['Catalog'] == 'no_radio') & (WISE_Matched_df['Magpis'] == 1), 'Catalog'] = 'observe'
WISE_Matched_df.loc[(WISE_Matched_df['Catalog'] == 'no_radio') & (WISE_Matched_df['NVSS'] == 1), 'Catalog'] = 'observe'
WISE_Matched_df.loc[(WISE_Matched_df['Catalog'] == 'no_radio') & (WISE_Matched_df['Cornish'] == 1), 'Catalog'] = 'observe'
WISE_Matched_df.loc[(WISE_Matched_df['Catalog'] == 'no_radio') & (WISE_Matched_df['Yusef20cmGC'] == 1), 'Catalog'] = 'observe'
WISE_Matched_df.loc[(WISE_Matched_df['Catalog'] == 'no_radio') & (WISE_Matched_df['LangRadio'] == 1), 'Catalog'] = 'observe'
WISE_Matched_df.loc[(WISE_Matched_df['Catalog'] == 'no_radio') & (WISE_Matched_df['LangPaschen'] == 1), 'Catalog'] = 'observe'
WISE_Matched_df.loc[(WISE_Matched_df['Catalog'] == 'no_radio') & (WISE_Matched_df['THOR'] == 1), 'Catalog'] = 'observe'
WISE_Matched_df.loc[(WISE_Matched_df['Catalog'] == 'no_radio') & (WISE_Matched_df['SMGPS'] == 1), 'Catalog'] = 'observe'
WISE_Matched_df.loc[(WISE_Matched_df['Catalog'] == 'no_radio') & (WISE_Matched_df['MeerKATGC'] == 1), 'Catalog'] = 'observe'

###########################################################################################################################

# Compare old vs new
WISE_V23_df = pd.read_csv('/Users/loren/papers/wise/wise_hii_V2.3_hrds.csv')

# Merging on different column names
dfmerged = pd.merge(WISE_V23_df, WISE_Matched_df, on='GName', how='left')
#print(dfmerged)
dfmerged['VLSR'] = pd.to_numeric(dfmerged['VLSR_x'], errors='coerce')
plt.scatter(dfmerged['VLSR'], dfmerged['VLSR_y'])

# determine where new values are 10km/s from old ones, or there is now a value where there wasn't before, or vise-versa
for i in range(len(dfmerged)):
    if (dfmerged['VLSR_y'][i]>-100) & (dfmerged['VLSR2'][i]!=dfmerged['VLSR2'][i]):
        if (np.abs(dfmerged['VLSR'][i] - dfmerged['VLSR_y'][i]) > 10) or ((dfmerged['VLSR_x'][i]!=dfmerged['VLSR_x'][i]) or (dfmerged['VLSR_y'][i]!=dfmerged['VLSR_y'][i])):
            print(dfmerged['GName'][i], dfmerged['Name_y'][i], dfmerged['VLSR_x'][i], dfmerged['Author_x'][i], dfmerged['VLSR_y'][i], 
                  dfmerged['VLSR2'][i], dfmerged['Author_y'][i], i)

# These were all updated by GLOSTAR, but I'm not sure if they are accurate
#G007.176+00.087 G7.177+0.088;FA099  11.1  Anderson et al. (2011) 0.49 nan Khan et al. (2024) 418
#G028.287-00.365 G28.287-0.364;G028.295-00.377  76.9  Anderson et al. (2015a) 43.85 nan Khan et al. (2024) 1791
#G031.279+00.061 G31.279+0.063;G031.275+00.056  104.7  Lockman (1989) 115.05 nan Khan et al. (2024) 2143
#G031.394-00.258 G31.394-0.259;G031.401-00.259  86.2  Lockman (1989) 74.05 nan Khan et al. (2024) 2152
#G033.419-00.005 G33.419-0.004;G033.418-00.004  76.5  Lockman (1989) 63.49 nan Khan et al. (2024) 2275
#G076.187+00.097 G76.187+0.098;TW044  8.5  Anderson et al. (2015b) -4.81 nan Khan et al. (2024) 3613

# These were all updated in the SHRDS
#G233.753-00.193 G233.761-00.191;G233.760-00.203  35.9000  Wenger et al. (2021) 46.0 nan Caswell & Haynes (1987) 4805
#G289.806-01.242 G289.755-01.152;G289.806-01.242  7.60000  Wenger et al. (2021) 22.0 nan Caswell & Haynes (1987) 5252
#G311.841-00.219 G311.852-00.222;G311.841-00.219  23.9000  Wenger et al. (2021) -55.0 nan Caswell & Haynes (1987) 5886
#G311.866-00.238 G311.852-00.222;G311.866-00.238  24.0000  Wenger et al. (2021) -55.0 nan Caswell & Haynes (1987) 5890
#G321.115-00.546 G321.105-00.549  -67.7000  Wenger et al. (2021) -56.0 nan Caswell & Haynes (1987) 6232
#G323.973+00.057 G322.407+00.221  -58.1000  Wenger et al. (2021) -30.0 nan Caswell & Haynes (1987) 6314
#G324.924-00.569 ATCA469;G324.954-00.584  -79.7000  Wenger et al. (2021) 25.0 nan Caswell & Haynes (1987) 6334
#G328.807-00.078 G328.806-00.083;G328.807-00.078  -36.2000  Wenger et al. (2021) -47.0 nan Caswell & Haynes (1987) 6504
#G333.681-00.441 G333.684-00.457;G333.681-00.441  -0.500000  Wenger et al. (2021) -50.0 nan Caswell & Haynes (1987) 6823
#G350.781-00.027 G350.813-00.019  11.7000  Wenger et al. (2021) 0.3 nan Lockman (1989) 7784

# this I think was a catalog error before
#G342.432-00.233 ATCA811  -7.0  Anderson et al. (2015b) -39.9 nan Anderson et al. (2015a) 7325

# Different parts of same large source
#G080.362+01.212 G080.612+01.465;G079.957+00.866  -12.8  Balser et al. (2011) 11.3 nan Lockman (1989) 3683
#G081.920+00.138 G081.526-00.037;G081.689+00.337  10.5  Lockman (1989) -3.4 nan Lockman (1989) 3708

# Replace NaN with None across the entire DataFrame
WISE_Matched_df = WISE_Matched_df.where(pd.notna(WISE_Matched_df), None)

# Diagnostics
n = len(WISE_Matched_df)
n_known = np.sum(WISE_Matched_df['Catalog'] == 'known')
n_sharpless = np.sum(WISE_Matched_df['Catalog'] == 'sharpless')
n_group = np.sum(WISE_Matched_df['Catalog'] == 'group')
n_observe = np.sum(WISE_Matched_df['Catalog'] == 'observe')
n_no_radio = np.sum(WISE_Matched_df['Catalog'] == 'no_radio')
print(n, n_known+n_sharpless+n_group+n_observe+n_no_radio, n_known, n_sharpless, n_group, n_observe, n_no_radio)


#################################################################################
# stuff not needed yet
# merging based on distance instead of matching on name
def merge_based_on_distance(WISE_df, Multiple_Velocities_df, max_distance=0.04):
    # Extract GLONG and GLAT columns
    glong1 = WISE_df['GLong'].values
    glat1 = WISE_df['GLat'].values
    glong2 = Multiple_Velocities_df['GLong'].values
    glat2 = Multiple_Velocities_df['GLat'].values

    # only take known regions
    good = np.where(WISE_df['VLSR2'] == WISE_df['VLSR2'])[0]
    print(good)
    
    # Create a meshgrid of all combinations between the two dataframes
    glong1_grid, glong2_grid = np.meshgrid(glong1[good], glong2)
    glat1_grid, glat2_grid = np.meshgrid(glat1[good], glat2)
    
    # Calculate the Euclidean distance for all combinations
    distance = np.sqrt((glong1_grid - glong2_grid)**2 + (glat1_grid - glat2_grid)**2)

    WISE_df['Real_VLSR'] = np.zeros(len(WISE_df))
    for i in range(distance.shape[0]):
        j = np.argmin(distance[i,:])
        #WISE_df['Real_VLSR'][j] = Multiple_Velocities_df['Real_VLSR'][i]
        print(i, j, WISE_df['Name'][good[j]], Multiple_Velocities_df['Name'][i], WISE_df['VLSR'][good[j]], Multiple_Velocities_df['Real_VLSR'][i])


# Call the function and get the combined DataFrame
merge_based_on_distance(WISE_df, Multiple_Velocities_df)



# Now, let's create a SQL database from the DataFrame
db_file = '/Users/loren/papers/wise/python/RRL_Surveys.db'
conn = sqlite3.connect(db_file)

# Drop the table if it exists to avoid the duplicate column issue
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS RRL_Surveys")  # Drop the table to avoid conflict
conn.commit()

# Define the schema for the table based on the cleaned dataframe columns
columns = RRL_Surveys_df.columns.tolist()
column_definitions = ', '.join([f'"{col}" TEXT' if RRL_Surveys_df[col].dtype == 'object' else f'"{col}" REAL' for col in columns])

# Create the table with the defined schema
create_table_query = f"""
CREATE TABLE IF NOT EXISTS RRL_Surveys (
    {column_definitions}
);
"""
cursor.execute(create_table_query)
conn.commit()

# Insert the data from the cleaned DataFrame manually
for row in RRL_Surveys_df.itertuples(index=False, name=None):
    placeholders = ', '.join(['?'] * len(row))  # Generate correct number of placeholders
    insert_query = f"""
    INSERT INTO RRL_Surveys ({', '.join([f'"{col}"' for col in columns])})
    VALUES ({placeholders});
    """
    cursor.execute(insert_query, row)

conn.commit()

# Close the database connection
conn.close()

print("Database populated successfully!")




# Create a connection to SQLite database (it will create a new database file if it doesn't exist)
conn = sqlite3.connect(f'{dir_path}python/wise_hii_master_V{str_version}.db')

# Write the DataFrame to the SQLite database as a new table
table_name = 'WISE_HII_Master'
WISE_df.to_sql(table_name, conn, if_exists='replace', index=False)

# Commit and close the connection to the database
conn.commit()
conn.close()

# Save the DataFrame to a pkl file too
with open(f"{dir_path}wise_hii_master_V{str_version}.pkl", 'wb') as f:
    pickle.dump(WISE_df, f)

print(f"Data saved to {dir_path}wise_hii_master_V{str_version}.pkl")
