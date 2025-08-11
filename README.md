VTE AND CORRELATION WITH UNCERTAINTY

This is the code for Cat's senior thesis in the Jadhav Lab in Brandeis University - The influence of uncertainty on rodent deliberative behavior

main functions of scripts
- process DeepLabCut (DLC) data into coordinate values
- convert StateScript data into trial information
- cut out centre zones from coordinates
- cut out relevant trajectories from centre zone
- calculate zIdPhi values to quantify VTEs
- derive latent variables like uncertainty based on modeling work
- pre-process data for use in HDDM
- pre-process data for use with RL models
- simulate models with reward outcomes from subject behavior fed in
    - betasort model (Transitive Inference-based)
    - value transfer (Transitive Inference-based)
    - TD learning (RL)
    - Q-learning (RL)
    - RW with generalization built in (conditioning-based)
    - Bayesian (seems generally good)
    - Neural network (fun)
- generate figures used for communication (publications, journal club, etc.)

main workflow
1. get data from citadel with load_data/retrieve_data.py
    - this gets most of the raw DLC data, statescript logs and timestamps files
    - formats in data/VTE_Data/{rat}/{module}/{day} for unimplanted rats and data/VTE_Data/{rat}/{module}/{day}/{track} for implanted rats
    - need to change remote.module (in config/paths) if you want to switch between training/testing data
    - note: maybe ~5 things with weird names i dragged and dropped manually because it wouldn't be worth the time to automate
2. organize formats and concatenate duplicates with preprocessing/initial_processing.py
    - changes statescript logs into txt files
    - changes timestamps files into npy files
3. process dlc to remove unlikely data
    - preprocessing/dlc_processing or dlc_processing_test
    - specifics of what is removed in docstring
    - creates coordinate files with x & y coordinates & time (from timestamps) in processed_data/cleaned_dlc
    - creates csv file of filtered data, noting when it happened and what the reason for filtering is
4. draw out center zone convex hulls in feature_extraction/create_zones.py
    - hull data saved as npy files in processed_data/hull_data
5. create idphi values with feature_extraction/create_IdPhi_values.py
    - or test for testing data
    - saves .jpg of each trajectory & trajectory files in processed_data/VTE_values/{rat}/{day}
    - trajectory files include ID ({rat}_{day}_{trajectory number}), x & y values of trajectory, correct, choice, trial type, idphi, and length (time)
6. create zidphi values by zscoring within each rat and choice with feature_extraction/create_zIdPhi_values.py
    - or test for testing data
    - creates zidphi files, with ID, correct, choice, trial type, idphi, length, zidphi, and zlength
    - this is the csv file used in 75% of visualization scripts

model workflow (for internal models, not including hddm)
1. with the zidphi values files, use preprocessing/model_prehandling.py to create model files in processed_data/data_for_model
    - it just creates easier vectors to work with
    - and also creates different metric for zlength, which are now all ranging from 0 to 1
2. use analysis/betasort_overall_pipeline.py to create the main plots comparing betasort model performance with rats
    - change model_type to switch between betasort, betasort_OG, and betasort_test
    - example rats are automatically excluded when using the pipeline - to look at example data, use analyze_example_rats()
    - plots are found in processed_data/new_model_data, in the folder corresponding to the suffix you have assigned
3. use analysis/betasort_day_pipeline.py to create more indepth plots looking at internal values in the model
    - creates plots for the beta distributions of each day, the U & L boundaries for each day, the positions and uncertainy across days
    - plots found in processed_data/new_model_data/{rat}/{day}
4. use analysis/compare_models.py to create plots comparing different models
    - betasort, Rescorla-Wagner, 2 layer feedforward neural network, bayesian, value transfer, td-lambda

Thesis draft available upon request

Relevant:
Redish A. D. (2016). Vicarious trial and error. Nature reviews. Neuroscience, 17(3), 147–159. https://doi.org/10.1038/nrn.2015.30 
Tolman, E. C. (1948). Cognitive maps in rats and men. Psychological Review, 55(4), 189–208. https://doi.org/10.1037/h0061626