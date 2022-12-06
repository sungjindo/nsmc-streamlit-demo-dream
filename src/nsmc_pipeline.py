from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline


def get_nsmc_pipeline():
    model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-v3-small-nsmc")
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-v3-small-nsmc")
    
    pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0,
)

    return pipe