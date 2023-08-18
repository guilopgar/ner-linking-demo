# NER
Module containing utils to perform Named Entity Recognition (NER) using Transformers. The implemented functionality allows the user to load and pre-process the input text data, and post-process the predictions made by the model.

Next, we describe the **hierarchy of classes** implemented following the **template method design pattern**:

## `pre_process.py`
- Tokenizer
  - FastTextTokenizer
  - TransformersTokenizer
- Labeler (used by Annotator)
  - LabelerIOB
    - LabelerNER
    - LabelerIOBNorm
  - LabelerNorm
- Annotator
  - AnnotatorContinuous
  - AnnotatorDiscontinuous
- SubLabelConverter
  - AllSubLabel
  - FirstSubLabel

## `post_process.py`
- WordPredsConverter
  - MaxWordPreds
  - ProdWordPreds
  - SumWordPreds
  - MeanWordPreds
  - FirstWordPreds
  - CRFAllWordPreds
- MentionPredsConverter
  - MaxMentionPreds
  - ProdMentionPreds
  - SumMentionPreds
  - MeanMentionPreds
  - FirstMentionPreds
- LabExtractor (used by AnnExtractor)
  - LabExtractorIOB
    - LabExtractorNorm
      - LabExtractorNER
      - LabExtractorIOBNorm
- AnnExtractor
  - AnnExtractorContinuous
  - AnnExtractorDiscontinuous
- PredsFragTok
  - NeuralPredsFragTok
  - CRFPredsFragTok
