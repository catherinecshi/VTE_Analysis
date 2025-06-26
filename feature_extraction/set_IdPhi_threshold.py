import os
import pandas as pd

from config.paths import paths
from visualization import generic_plots

# for plotting
vte_zIdPhi_values = {} # {choice: [zIdPhi values]}
norm_zIdPhi_values = {}
vte_IdPhi_values = {}
norm_IdPhi_values = {}

vte_data_path = paths.manual_vte
for vte_filename in os.listdir(vte_data_path):
    parts = vte_filename.split("_")
    rat = parts[0]
    
    rat_vte_file_path = os.path.join(vte_data_path, vte_filename)
    try:
        rat_vte_file = pd.read_csv(rat_vte_file_path)
    except UnicodeDecodeError:
        print(rat_vte_file_path)
        continue
    
    for index, row in rat_vte_file.iterrows():
        zIdPhi = float(row["zIdPhi"])
        IdPhi = float(row["IdPhi"])
        choice = row["Choice"]
        is_VTE = bool(row["VTE"])
        
        # save for later plots
        if is_VTE:
            if choice in vte_zIdPhi_values:
                vte_zIdPhi_values[choice].append(zIdPhi)
                vte_IdPhi_values[choice].append(IdPhi)
            else:
                vte_zIdPhi_values[choice] = [zIdPhi]
                vte_IdPhi_values[choice] = [IdPhi]
        else:
            if choice in norm_zIdPhi_values:
                norm_zIdPhi_values[choice].append(zIdPhi)
                norm_IdPhi_values[choice].append(IdPhi)
            else:
                norm_zIdPhi_values[choice] = [zIdPhi]
                norm_IdPhi_values[choice] = [IdPhi]

vte_zIdPhis = [zIdPhi for values in vte_zIdPhi_values.values() for zIdPhi in values]
norm_zIdPhis = [zIdPhi for values in norm_zIdPhi_values.values() for zIdPhi in values]
vte_IdPhis = [IdPhi for values in vte_IdPhi_values.values() for IdPhi in values]
norm_IdPhis = [IdPhi for values in norm_IdPhi_values.values() for IdPhi in values]

zIdPhi_dict = {"VTE Trajectories": vte_zIdPhis, "Non-VTE Trajectories": norm_zIdPhis}
IdPhi_dict = {"VTE Trajectories": vte_IdPhis, "Non-VTE Trajectories": norm_IdPhis}

generic_plots.create_multiple_frequency_histograms(zIdPhi_dict, title="zIdPhi values between VTE vs Non-VTE Trajectories",
                                              xlabel="zIdPhi values", ylabel="Density")
generic_plots.create_multiple_frequency_histograms(IdPhi_dict, title="IdPhi values between VTE vs Non-VTE Trajectories",
                                              xlabel="IdPhi values", ylabel="Density")