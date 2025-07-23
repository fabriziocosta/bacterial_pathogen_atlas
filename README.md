<p align="center">
  <img src="logo.png" alt="Project logo">
</p>


# Large Language Model-assisted text mining reveals bacterial pathogen diversity  

Compiling and characterising the diversity of bacterial pathogens of humans is a critical challenge to tackle infection risk, especially in the context of global antimicrobial resistance, climate change, and changing demographics. Here, we present a scalable, automated pipeline that harnesses large language models (LLMs) to systematically mine the biomedical literature for information of human pathogenicity across the bacterial domain. By interrogating tens of thousands of PubMed abstracts we identify 1,222 species with at least one abstract documenting human infection, of which 783 species are supported by ≥3 abstracts and are regarded as ‘confirmed pathogens’. We extract, summarise, visualise and interpret data on infection contexts using both expert-curated LLM prompts and unsupervised text vectorisation. We show that these methods enable fine-grained trait mapping across taxa, including quantifying the degree of specialism or generalism in body site specificity for different taxa and the classification of pathogen species into 75 ‘pathogen types’. An objective measure of the rate at which species are reported in the literature coupled to species clustering offers insights into the drivers of pathogen emergence. Our LLM-driven strategy generates an open, updatable, evidence-based catalogue of bacterial human pathogens and their ecological and clinical traits, providing a foundation for public health surveillance, diagnostics, and predictive modelling. This work demonstrates the potential of AI-assisted literature synthesis to transform our understanding of microbial diversity, including its impact on human health.


[Download data as a ZIP](https://fabriziocosta.pythonanywhere.com/download-zip) (804MB)

Unizip to obtain the files:
- counts.csv
- data.csv
- emergence.csv
- YES_NO_questions.xlsx

in the directory 'BacterialPathogensData'.

The IPython notebooks image_generation_clusters.ipynb, and image_generation_scatter.ipynb in this repository can then be used to generate the figures in the manuscript. 
