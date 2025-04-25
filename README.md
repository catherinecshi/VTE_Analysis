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

Thesis draft available upon request

Relevant:
Redish A. D. (2016). Vicarious trial and error. Nature reviews. Neuroscience, 17(3), 147–159. https://doi.org/10.1038/nrn.2015.30 
Tolman, E. C. (1948). Cognitive maps in rats and men. Psychological Review, 55(4), 189–208. https://doi.org/10.1037/h0061626