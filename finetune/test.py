from transformers import VisionEncoderDecoderConfig
from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig
from datasets import load_dataset
import torch

max_length = 128
image_size = [1280, 960]
config = VisionEncoderDecoderConfig.from_pretrained("facebook/nougat-base")
config.encoder.image_size = image_size
config.decoder.max_length = max_length
processor = DonutProcessor.from_pretrained("facebook/nougat-base")
#processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("lightning_logs/version_0/checkpoints/final.ckpt", config=config)
#model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base", config=config)
#model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
dataset = load_dataset("Alchemy5/autodiagram", split="validation")
#dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset["images"][0]
#image = dataset[0]["image"]
pixel_values = processor(image, return_tensors="pt").pixel_values
print(pixel_values.shape)
task_prompt = "{user_input}"
question = "Convert this image into tex."
#question = "When is the coffee break?"
prompt = task_prompt.replace("{user_input}", question)

decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

outputs = model.generate(pixel_values.to(device),
                               decoder_input_ids=decoder_input_ids.to(device),
                               max_length=model.decoder.config.max_position_embeddings,
                               early_stopping=True,
                               pad_token_id=processor.tokenizer.pad_token_id,
                               eos_token_id=processor.tokenizer.eos_token_id,
                               use_cache=True,
                               num_beams=1,
                               bad_words_ids=[[processor.tokenizer.unk_token_id]],
                               return_dict_in_generate=True,
                               output_scores=True)
seq = processor.batch_decode(outputs.sequences)[0]
print(seq)
import pdb;pdb.set_trace()