
Files created by me:
    - gradient_calculation.py: Generates more unit while calculating gradients of the DLM input units
    - unit_heatmap.py: Plots heatmap with DLM saliency maps
    - transcript_segmentation_alignmnet.py: Plots transcripts aligned with speaker segmentation from the VAD system
    - channel_weighing.py: Plots saliency score per DLM input channel
    - hubert_saliency_map.py: Saliency map for HuBERT encoder

Supporting files created by me:
    - audio_utils/*
    - transcript_utils/*
    - saliency_mappping/*

Fairseq files edited by me:
    - fairseq/models/speech_dlm/sequence_generator/multichannel_sequence_generator.py: Added gradient calculation during each generation cycle
    fairseq/models/speech_dlm/modules/speech_dlm_decoder.py: Added methods for extracting input gradients
    - fairseq/tasks/speech_dlm_task.py: Changed gradient calculation settings
    - examples/textless_nlp/dgslm/dgslm_utils.py: Returns units as torch.Tensor instead of np.array and also returns distances
