"""
Centralizes all directory and file path construction
"""

from pathlib import Path

BASE_PATH = Path("/Users/catpillow/Documents/VTE_Analysis")

class RemotePaths:
    """path management when getting data from citadel"""

    def __init__(self):
        self.remote_name = "VTE"
        self.remote_path = "data/Projects/bp_inference/Transitive"
        self.module = "inferenceTraining"
        self.excluded_folders = ['.dat', '.mda', 'mountain', '.json', '.mat', '.txt', 'postSleep', 'preSleep', 'Logs', '.timestampoffset',
                                '.DIO', 'log', 'msort', 'LFP', 'spikes', 'reTrain', '.h264', 'geometry', 'HWSync', 'midSleep']
        self.included_patterns = ["*.stateScriptLog", "*.videoTimeStamps"]

class ProjectPaths:
    """Centralized path management for the entire project."""
    
    def __init__(self, base_dir=BASE_PATH):
        self.base = Path(base_dir)
        
        # raw data directories
        self.data = self.base / "data"
        self.vte_data = self.data / "VTE_Data"
        self.timestamps = self.data / "timestamps"
        self.statescripts = self.data / "statescripts"
        
        # processed data directories
        self.processed = self.base / "processed_data"
        self.performance = self.processed / "performance"
        self.dlc_data = self.processed / "dlc_data"
        self.cleaned_dlc = self.processed / "cleaned_dlc"
        self.vte_values = self.processed / "VTE_values"
        self.manual_vte = self.processed / "manual_VTE"
        self.hull_data = self.processed / "hull_data"
        self.vertice_data = self.processed / "VTE_data"
        
        # model data directories
        self.preprocessed_data_model = self.processed / "data_for_model"
        self.betasort_data = self.processed / "new_model_data"
        self.model_comparison = self.processed / "model_comparison"
        self.optimization_data = self.processed / "model_comparison_optimized"
        self.response_time = self.processed / "response_time"
        
        # Output directories
        self.results = self.base / "results"
        self.figures = self.base / "figures"
        self.logs = self.base / "doc"
    
    def get_rat_dlc_path(self, rat_id):
        """Get the DLC data directory for a specific rat."""
        return self.cleaned_dlc / rat_id
    
    def get_rat_vte_path(self, rat_id):
        """Get the VTE values directory for a specific rat."""
        return self.vte_values / rat_id
    
    def get_coordinates_file(self, rat_id, day):
        """Get path to coordinates file for specific rat and day."""
        return self.cleaned_dlc / rat_id / f"{day}_coordinates.csv"

# Create a global instance that can be imported everywhere
paths = ProjectPaths()
remote = RemotePaths()
