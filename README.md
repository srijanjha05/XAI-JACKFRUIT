# xai4dementia-framework
An Unsupervised Explainable AI Framework for Dementia Detection with Context Enrichment

![Python](https://img.shields.io/badge/Python-v3.11-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.15.1-orange)
![iNNvestigate](https://img.shields.io/badge/iNNvestigate-v2.1.2-blue)


Explainable Artificial Intelligence (XAI) methods enhance the diagnostic efficiency of clinical decision support systems by making the predictions of a convolutional neural network’s (CNN) on brain imaging more transparent and trustworthy. However, their clinical adoption is hindered due to the limited validation of the explanation quality. Our study introduces a framework that evaluates XAI methods by integrating neuroanatomical morphological features - gray matter volumetry and average cortical thickness signals, with CNN-generated relevance maps for disease classification.

Further details could be found in our publication:
Singh, Devesh, et al. "An Unsupervised XAI Framework for Dementia Detection with Context Enrichment."  Sci Rep 15, 39554 (2025). 
[https://doi.org/10.1038/s41598-025-26227-2](https://doi.org/10.1038/s41598-025-26227-2)

## Pipeline Overview
The workflow of our study is schematically presented in Figure below. Our framework provides several ways to generate post-hoc explanations for a CNN model trained to detect dementia diseases, including: i) global-level explanations, such as membership in the stable versus converter subgroups, and ii) local-level explanations for each individual prediction, such as ii-a) example-based explanations of cognitive trajectories or ii-b) textual explanation by pathology summarization.

<p align="center">
  <img src="/images/1.png" style="width:100%; max-width:100%;">
</p>
<![Pipeline Flow](/images/1.png)>




## Citation

```bibtex
@article {Singh2025.05.28.25327435,
	author = {Singh, Devesh and Brima, Yusuf and Levin, Fedor and Becker, Martin and Hiller, Bjarne and Hermann, Andreas and Villar-Munoz, Irene and Beichert, Lukas and Bernhardt, Alexander and Buerger, Katharina and Butryn, Michaela and Dechent, Peter and Duezel, Emrah and Ewers, Michael and Fliessbach, Klaus and D. Freiesleben, Silka and Glanz, Wenzel and Hetzer, Stefan and Janowitz, Daniel and Goerss, Doreen and Kilimann, Ingo and Kimmich, Okka and Laske, Christoph and Levin, Johannes and Lohse, Andrea and Luesebrink, Falk and Munk, Matthias and Perneczky, Robert and Peters, Oliver and Preis, Lukas and Priller, Josef and Prudlo, Johannes and Prychynenko, Diana and Rauchmann, Boris-Stephan and Rostamzadeh, Ayda and Roy-Kluth, Nina and Scheffler, Klaus and Schneider, Anja and Droste zu Senden, Louise and H. Schott, Bjoern and Spottke, Annika and Synofzik, Matthis and Wiltfang, Jens and Jessen, Frank and Weber, Marc-Andre and Teipel, Stefan J. and Dyrba, Martin},
	title = {An Unsupervised XAI Framework for Dementia Detection with Context Enrichment},
	elocation-id = {2025.05.28.25327435},
	year = {2025},
	doi = {10.1101/2025.05.28.25327435},
	publisher = {Cold Spring Harbor Laboratory Press},
	abstract = {Introduction: Explainable Artificial Intelligence (XAI) methods enhance the diagnostic efficiency of clinical decision support systems by making the predictions of a convolutional neural network{\textquoteright}s (CNN) on brain imaging more transparent and trustworthy. However, their clinical adoption is limited due to limited validation of the explanation quality. Our study introduces a framework that evaluates XAI methods by integrating neuroanatomical morphological features with CNN-generated relevance maps for disease classification. Methods: We trained a CNN using brain MRI scans from six cohorts: ADNI, AIBL, DELCODE, DESCRIBE, EDSD, and NIFD (N=3253), including participants that were cognitively normal, with amnestic mild cognitive impairment, dementia due to Alzheimer{\textquoteright}s disease and frontotemporal dementia. Clustering analysis benchmarked different explanation space configurations by using morphological features as proxy-ground truth. We implemented three post-hoc explanations methods: i) by simplifying model decisions, ii) explanation-by-example, and iii) textual explanations. A qualitative evaluation by clinicians (N=6) was performed to assess their clinical validity. Results: Clustering performance improved in morphology enriched explanation spaces, improving both homogeneity and completeness of the clusters. Post hoc explanations by model simplification largely delineated converters and stable participants, while explanation-by-example presented possible cognition trajectories. Textual explanations gave rule-based summarization of pathological findings. Clinicians{\textquoteright} qualitative evaluation highlighted challenges and opportunities of XAI for different clinical applications. Conclusion: Our study refines XAI explanation spaces and applies various approaches for generating explanations. Within the context of AI-based decision support system in dementia research we found the explanations methods to be promising towards enhancing diagnostic efficiency, backed up by the clinical assessments.},
	URL = {https://www.medrxiv.org/content/early/2025/06/04/2025.05.28.25327435},
	eprint = {https://www.medrxiv.org/content/early/2025/06/04/2025.05.28.25327435.full.pdf},
	journal = {medRxiv}
}
```
