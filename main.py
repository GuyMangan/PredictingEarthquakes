################
#Imports
################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from conda_index.yaml import determined_load
from joblib import Parallel, delayed

################
#Functions
################

def filter_catalog(catalog,epicentre_longitude,epicentre_latitude, event_date):
    """Filters catalog of EQs By:
    -Magnitude (Greater than Completeness Magnitude)
    -Date      (1y before earthquake being studied)
    -Depth     (less than 70km Deep)
    -Area      (1 degree latitude/longitude circle
                or 2 degree latitude/longitude square
                around epicentre of earthquake being studied)
    """

    depth_filtered = filter_depth(convert_magnitudes(catalog))
    depth_mag_filtered = filter_magnitude(depth_filtered)
    print(depth_filtered)
    depth_mag_area_filtered = filter_area(depth_mag_filtered,epicentre_longitude,epicentre_latitude)
    depth_mag_area_date_filtered = filter_date(depth_mag_area_filtered, event_date)
    return depth_mag_area_date_filtered

def extract_catalog(datafile):
    """Extracts catalog of EQs from CSV"""
    data = pd.read_csv(datafile, encoding='latin-1')
    return data

def convert_magnitudes(catalog):
    """Converts Magnitude of catalog of EQs to M_w (Seismic Moment)"""
    catalog = catalog.copy()

    #conditions for magnitude types
    conditions = [
        catalog['magType'].isin(['mwr', 'mwc', 'mww', 'mwb']),
        (catalog['magType'] == 'ms') & (catalog['mag'] < 6.1),
        (catalog['magType'] == 'ms') & (catalog['mag'] >= 6.1),
        catalog['magType'] == 'mb'
    ]

    #corresponding calculations
    choices = [
        catalog['mag'],
        catalog['mag'] * 0.67 + 2.12,
        catalog['mag'] * 1.06 - 0.38,
        (catalog['mag'] - 1.65) / 0.65
    ]

    # It defaults to keeping the value unconverted for rows that don't match any condition
    catalog['convertedMagnitude'] = np.select(conditions, choices, default=catalog['mag'])

    # Drop all rows where no conversion was possible
    catalog.dropna(subset=['convertedMagnitude'], inplace=True)

    return catalog

def filter_depth(catalog):
    """Filters catalog by Depth of EQs to those less than 70km Deep"""
    filtered_catalog = catalog[catalog['depth'] < 70].copy()
    return filtered_catalog

def filter_magnitude(catalog):
    """Filters catalog by Magnitude of EQs to those greater than the Completeness Magnitude"""
    catalog = catalog.copy()
    j = 0
    magnitude_limits = [3,3.25,3.5,3.75,4,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,
                        5,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0]
    log_Ns = []
    fig, ax = plt.subplots()
    while j<len(magnitude_limits):
        for magnitude in magnitude_limits:
            count = (catalog['convertedMagnitude'] >= magnitude).sum()
            log_Ns.append(np.log10(count))
            ax.scatter(magnitude,np.log10(count))
            if log_Ns[j - 1] - log_Ns[j] > 1:
                filtered_catalog = catalog[(catalog.convertedMagnitude > magnitude)]
                print(f'completeness magnitude is {magnitude}')
                return filtered_catalog
            j += 1
    print('no completeness magnitude found')
    return catalog[(catalog.convertedMagnitude > 3)]

def filter_area(catalog, epicentre_longitude, epicentre_latitude):
    """Filters catalog by Area of EQs to those within a 2 degree by 2 degree square around the Epicentre of the earthquake being studied"""
    # catalog = catalog.copy()
    # lon_condition1 = catalog['longitude'] <= -66
    # lon_condition2 = catalog['longitude'] >= -75
    #
    # lat_condition1 = catalog['latitude'] <= -17
    # lat_condition2 = catalog['latitude'] >= -76
    return catalog
    # return catalog[lon_condition1 & lat_condition1 & lon_condition2 & lat_condition2]

def filter_date(catalog, event_date):
    """filters catalog to those earthquakes which occurred within 1 year before earthquake being studied"""
    catalog = catalog.copy()
    catalog['time'] = pd.to_datetime(catalog['time'], errors='coerce')

    catalog.dropna(subset=['time'], inplace=True)
    event_date = pd.to_datetime(event_date)
    start_date = event_date - pd.DateOffset(months = 18)

    filtered_catalog = catalog[(catalog['time'] >= start_date) & (catalog['time'] <= event_date)]
    return filtered_catalog

def mag_to_energy(M):
    energy = 10 ** (5.24 + 1.44 * M)  #USGS
    #energy = 10**(11.8 + 1.5*M) #Gutenberg 1958
    return energy

def normalise_energy(Qs):
    """Normalised Energy
    p_k = Q_k/Sum(Q_n, from 1 to N)
    N = Number of EQs in dataset
    """
    sum_Qs = np.sum(Qs)
    if sum_Qs == 0:
        # to avoid division by zero, return an array of zeros
        return np.zeros_like(Qs)
    return Qs / sum_Qs

def natural_time_variance(Qs):
    """Natural Time Variance"""
    ps = normalise_energy(Qs)
    N = len(Qs)
    # Create the (i+1)/N term as a numpy array
    k_over_N = (np.arange(1, N + 1) / N)

    term_1 = np.sum(ps * (k_over_N ** 2))
    term_2 = np.sum(ps * k_over_N)

    kappa = term_1 - (term_2 ** 2)
    return kappa

def beta_w(dataframe,W,ks):
    """Beta w = standard deviation of kappa/ variance of kappa"""
    beta_w = np.std(ks)/np.var(ks)
    dataframe.iloc['beta_w',W] = beta_w
    return beta_w

def w_calc(catalog,event_date):
    """Calculate w based on number of EQs in the 6mo before Main EQ"""
    catalog['time'] = pd.to_datetime(catalog['time'], errors='coerce')

    catalog.dropna(subset=['time'], inplace=True)
    event_date = pd.to_datetime(event_date)
    start_date = event_date - pd.DateOffset(months=6)
    print(start_date)
    filtered_catalog = catalog[(catalog['time'] >= start_date) & (catalog['time'] <= event_date)]

    w = len(filtered_catalog)

    return w

def find_eqs(catalog,mag):
    """Find EQs above a threshold magnitude in catalog"""
    eqs = catalog[catalog['convertedMagnitude'] > mag]
    return eqs

################
#Global Variables
################
MAIN_LAT = -36.122
MAIN_LON = -72.898
MAIN_DATE = '2019-07-06T03:20:00.000Z'
n_to_plot = 6

################
#Main
################
#Read Data File In
catalog = extract_catalog('/Users/guy/Desktop/T1 Lab Project/Report Data Files/CALI_2000_2024_M25+.csv')
current_catalog = filter_catalog(catalog,MAIN_LON,MAIN_LAT,MAIN_DATE)
new_current_catalog = current_catalog.copy()
W = w_calc(current_catalog,MAIN_DATE)
print(len(current_catalog))
print(W)
################
#Parallelisation
################

def process_window(j, catalog, W, n_to_plot):
    """
    Processes a single sliding window to calculate beta and k values.
    """
    # Select the slice of the catalog for this window
    w_window = catalog.iloc[j : j + W]

    k_values_for_this_window = []
    # Use None as a placeholder for the optional k value to plot
    k_to_plot = None
    k_index_to_plot = None

    # Calculate k for sub-excerpts of size N (from 6 to W)
    for N in range(6, W + 1):
        for i in range(W - N + 1):
            sub_excerpt_magnitudes = w_window['convertedMagnitude'].iloc[i : i + N]
            if len(sub_excerpt_magnitudes) >= 6:
                energies = mag_to_energy(sub_excerpt_magnitudes)
                k = natural_time_variance(energies)
                k_values_for_this_window.append(k)
                # Check if this is the specific k value to save for plotting
                if N == n_to_plot and i == 1:
                    k_to_plot = k
                    k_index_to_plot = catalog.index[j + W]

    # Calculate beta_w for the current main window
    if k_values_for_this_window:
        mean_k = np.mean(k_values_for_this_window)
        std_k = np.std(k_values_for_this_window)
        beta_w = (std_k / mean_k) if mean_k != 0 else np.nan
    else:
        beta_w = np.nan

    # Get the index corresponding to the end of this window
    beta_index = catalog.index[j + W]

    # Return all the results from this single iteration as a tuple
    return (beta_w, beta_index, k_to_plot, k_index_to_plot)

print("Starting parallel processing.")
results = Parallel(n_jobs=-1, verbose=20)(
    delayed(process_window)(j, new_current_catalog, W, n_to_plot)
    for j in range(len(new_current_catalog) - W)
)
print("Parallel processing finished.")

beta_results = []
beta_indices = []
k_values_for_plot = []
k_indices_for_plot = []

for beta_w, beta_idx, k_plot, k_idx_plot in results:
    beta_results.append(beta_w)
    beta_indices.append(beta_idx)
    # Only append the k value if it was found in that iteration
    if k_plot is not None:
        k_values_for_plot.append(k_plot)
        k_indices_for_plot.append(k_idx_plot)

beta_series = pd.Series(beta_results, index=beta_indices)
new_current_catalog['beta'] = beta_series

beta_min = np.where(new_current_catalog['beta'] == new_current_catalog['beta'].min())[-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(new_current_catalog)), new_current_catalog['beta'], label=f'β fluctuation (W={W})', marker='o', linestyle='-', markersize=4)
ax.axvline(x=(len(new_current_catalog)-1), color='r', linestyle='--', label=f'{new_current_catalog["mag"].max()} Mag EQ {new_current_catalog["time"].iloc[np.where(new_current_catalog["mag"] == new_current_catalog["mag"].max())]}')
ax.axvline(x=beta_min[0], color='r', linestyle='--', label=f'Potential SES Event at {new_current_catalog["time"].iloc[beta_min]}')

ax.set_title(f'Fluctuation (β) of Natural Time Variance over Time (Window Size W={W})')
ax.set_xlabel('Event Index or Time')
ax.set_ylabel('β Value (std(k) / mean(k))')
ax.legend()
ax.grid(True)

SES_to_Main = new_current_catalog[beta_min[0]:]
print(len(SES_to_Main))
fig,ax3 = plt.subplots(figsize=(10, 6))
ax3.set_title(f'Natural Time Variance (k) over Natural Time')
ax3.set_xlabel('Natural Time')
ax3.set_ylabel('k value')
ax3.set_xticks(np.arange(0, len(SES_to_Main)+1, 10))
#ax3.grid(True)
for i in range(6,len(SES_to_Main)+1):
    energies = mag_to_energy(SES_to_Main['convertedMagnitude'][0:i])
    k = natural_time_variance(energies)
    ax3.scatter(i,k,color = 'black')
plt.show()
