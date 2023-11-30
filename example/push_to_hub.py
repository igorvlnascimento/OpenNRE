from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "ckpt\ddi\gpt2\dare_gpt2_ddi_byrelation_finetuning_with_rl"

llm_model = AutoModelForCausalLM.from_pretrained(model_name)
llm_tokenizer = AutoTokenizer.from_pretrained(model_name)

llm_model.push_to_hub("igorvln/dare_gpt2_ddi_byrelation_finetuning_with_rl")
llm_tokenizer.push_to_hub("igorvln/dare_gpt2_ddi_byrelation_finetuning_with_rl")