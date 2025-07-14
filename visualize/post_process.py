import pandas as pd
import numpy as np
import torch
import os, sys
from util import plot_pulse_param

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


angle_list = ["0.25", "0.33", "0.50", "0.67", "0.75", "1.00"]

pulse_dir = "figures/phase_control_1600_pulse_finetuned_iter2/pulse_param_csv/"

save_path = "weights/1600_pulse_length_post_process_iter3"


# List of input pulse CSV files
input_files = [
    os.path.join(pulse_dir, f"$R_X$({angle}$\pi$)_pulse.csv")
    for angle in angle_list
]

# Processing parameters
slope_threshold = 1 # rad/unit

# Process each file
for infile in input_files:
    # Read CSV without headers
    df = pd.read_csv(infile, names=['phi', 'tau'])
    
    # Identify spikes based on slope
    phi = df['phi'].values
    tau = df['tau'].values
    slopes = (phi[1:] - phi[:-1]) / tau[1:]
    spike_idx = np.where(np.abs(slopes) > slope_threshold)[0][10:] + 1
    
    # Clean phi by interpolation
    phi_series = pd.Series(phi).copy()
    phi_series.iloc[spike_idx] = np.nan
    phi_clean = phi_series.interpolate().fillna(method='bfill').fillna(method='ffill')
    
    # Prepare cleaned DataFrame with renamed columns
    cleaned_df = pd.DataFrame({
        '0': phi_clean,
        '1': tau
    })
    
    # Drop the first row
    cleaned_df = cleaned_df.iloc[1:].reset_index(drop=True)
    
    # Construct output filename
    angle = infile.split('$R_X$(')[1].split('$\\pi$')[0]
    outfile = os.path.join(save_path, f'cleaned_RX_{angle}pi.csv')
    os.makedirs(save_path, exist_ok=True)
    
    # Save cleaned CSV
    cleaned_df.to_csv(outfile, index=False)
    
    plot_pulse_param(
        save_path, 
        f"post_processed_param_{angle}pi", 
        ["Phase (units of pi)"], 
        cleaned_df
    )


tensors = []
angle_list = ["0.25", "0.33", "0.50", "0.67", "0.75", "1.00"]

for angle in angle_list:
    # csv_path = f"figures/phase_control_{tau_max}_tau_max/pulse_param_csv/$R_X$({angle}$\\pi$)_pulse.csv"
    csv_path = os.path.join(save_path, f"cleaned_RX_{angle}pi.csv")
    df = pd.read_csv(csv_path)
    arr = df.values.astype('float32')        # shape (400, 2)
    tensor = torch.from_numpy(arr)          # Tensor of shape (400, 2)
    tensors.append(tensor)

# 2. Stack the list of tensors along a new dimension to get shape (6, 400, 2)
combined = torch.stack(tensors, dim=0)      # shape (6, 400, 2)

# 3. (Optional) save to disk
torch.save(combined, os.path.join(save_path, 'combined_pulses.pt'))
combined.shape