import os
import pandas as pd
import time
from datetime import datetime

eq = "Kumamoto_6"
# date = "2016-04-15T16:25:06"
date = "2019-07-06T03:19:53"

base_path = f'/cluster/home/fcolosimo/Data/{eq}'
out_path = f'/cluster/scratch/fcolosimo/Data/{eq}'

# Load the centroids data
centroids_df = pd.read_csv(f'{base_path}/fault_csv/grouped_centroids_data_with_magnitude.csv')

# Create a directory with the current UTC time and _synthetics
current_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
synthetics_dir = os.path.join(out_path, f'{current_time}_synthetics')
os.makedirs(synthetics_dir, exist_ok=True)



# Create additional directories inside synthetics_dir
os.makedirs(os.path.join(synthetics_dir, 'mseed_raw'), exist_ok=True)
os.makedirs(os.path.join(synthetics_dir, 'processed_mseeds/realizations'), exist_ok=True)
os.makedirs(os.path.join(synthetics_dir, 'processed_mseeds/summed'), exist_ok=True)


# Save the centroids data to the synthetics directory
centroids_df.to_csv(os.path.join(synthetics_dir, 'grouped_centroids_data_with_magnitude.csv'), index=False)

# Iterate over each centroid
for centroid in centroids_df.to_dict(orient='records'):
    centroid_lat = centroid['centroid_lat']
    centroid_lon = centroid['centroid_lon']
    centroid_depth = centroid['centroid_depth']
    trup = centroid['trup']
    magnitude = centroid['magnitude']
    csv_file = f"{base_path}/generative/generative_tables/generative_centroid_{centroid_lat}_{centroid_lon}_{centroid_depth}_{magnitude}_{trup}.csv"
    output_hdf5 = f"{synthetics_dir}/centroid_{centroid_lat}_{centroid_lon}_{trup}_{magnitude}.hdf5"
    
    # Estimate time remaining
    start_time = time.time()

    # Command to produce synthetics
    command1 = f"generate-waveforms --outfile {output_hdf5} --edm_checkpoint Codes/tqdne/weights/edm.ckpt  --autoencoder_checkpoint Codes/tqdne/weights/autoencoder.ckpt --csv {base_path}/generative/generative_tables/generative_{centroid_lat}_{centroid_lon}_{centroid_depth}_{magnitude}_{trup}.csv "
    os.system(command1)
    
    # Command to write metadata
    command2 = f"python Codes/tqdne/scripts/write_to_seisbench.py {base_path}/generative/centroid_{centroid_lat}_{centroid_lon}_{centroid_depth}_{magnitude}_{trup}.csv {output_hdf5} {synthetics_dir}"
    os.system(command2)

    current_index = centroids_df.index[centroids_df['centroid_lat'] == centroid_lat].tolist()[0]
    total = len(centroids_df)
    print(f'Processing {current_index + 1} out of {total} centroids')
    
    
    elapsed_time = time.time() - start_time
    eta = elapsed_time * (total - (current_index + 1))
    if eta > 3600:
        eta_hours = eta / 3600
        print(f'Estimated time remaining: {eta_hours:.2f} hours')
    elif eta > 60:
        eta_minutes = eta / 60
        print(f'Estimated time remaining: {eta_minutes:.2f} minutes')
    else:
        print(f'Estimated time remaining: {eta:.2f} seconds')

command3 = f"python Codes/Finite_rupture/processing/01_writing_to_mseed.py {synthetics_dir} {eq} {date} "
os.system(command3)

command4 = f"python Codes/Finite_rupture/processing/02_adapting_frequency.py {synthetics_dir} {eq} {date}"
os.system(command4)

command5 = f"python Codes/Finite_rupture/processing/04_group_realisations.py {synthetics_dir} {eq} {date}"
os.system(command5)

command6 = f"python Codes/Finite_rupture/processing/05_frequency.py {synthetics_dir} {eq} {date}"
os.system(command6)
    


