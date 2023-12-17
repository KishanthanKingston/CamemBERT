# CamemBERT
implémentation du CamemBERT
---
### Emir TAS
### Shivamshan SIVANESAN
### Kishanthan KINGSTON
---
## Advanced Machine Learning project
### MSc. Intelligent systems engineering

Our project aims to re-implement the CamemeBERT model as described in the article [1]. To achieve this goal, we developed SimpleRoBERTa, a simplified version of the CamemeBERT architecture that adheres to the structure outlined in the referenced article [1].

The MLM RoBERTa class is specifically designed for training a Masked Language Model (MLM) using the previously created SimpleRoBERTa architecture. The preprocessing class is implemented in the "utils" folder, while the positional encoding class is implemented in the "transformer" folder. Additionally, we pretrained a version of RoBERTa using our French database, and the pretrained model can be found in the "pretrained_roberta_MLM.ipynb" notebook.

We utilize the EvaluateRoBERTa class to evaluate our MLM_RoBERTa, and the "MLM_RoBERTa_POS_Tag.py" is used to fine-tune our model with POS-tag data. The "PosTagging.py" is used to evaluate the pretrained RoBERTa.

All the results concerning our MLM_RoBERTa are consolidated in the "test.ipynb" file, and the POS-tagging evaluation, performed with the pretrained version of RoBERTa trained on our French database, is presented in "Pos-tagging_Results.ipynb."

All the necessary libraries are listed in the "requirements.txt" file. Ensure to install these dependencies by running the appropriate command, typically "pip install -r requirements.txt," to ensure that the development environment is properly configured with the specific versions of the required libraries.

References:

[1] Martin, Louis, et al. « CamemBERT: A Tasty French Language Model ». Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Association for Computational Linguistics, 2020, p. 7203‑19. DOI.org (Crossref), https://doi.org/10.18653/v1/2020.acl-main.645.

[2] Y. Liu et al., « RoBERTa: A Robustly Optimized BERT Pretraining Approach ». arXiv, 26 juillet 2019. Consulté le: 22 novembre 2023. [En ligne]. Disponible sur: http://arxiv.org/abs/1907.11692.

[3] A. Vaswani et al., « Attention Is All You Need ». arXiv, 1 août 2023. Consulté le: 22 novembre 2023. [En ligne]. Disponible sur: http://arxiv.org/abs/1706.03762.

[4] https://github.com/shreydan/masked-language-modeling/blob/main/model.py

[5] https://huggingface.co/blog/how-to-train

[6] https://github.com/theartificialguy/NLP-with-Deep-Learning/blob/master/BERT/Fine%20Tune%20BERT/fine_tuning_bert_with_MLM.ipynb

[7] https://github.com/mdabashar/Deep-Learning-Algorithms/blob/master/Retraining%20RoBERTa%20for%20MLM.ipynb
