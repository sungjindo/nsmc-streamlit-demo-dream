from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline


def get_nsmc_pipeline(
    model_name_or_path="monologg/koelectra-v3-small-nsmc",
    device= 0,
    
):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=device,
)

    return pipe