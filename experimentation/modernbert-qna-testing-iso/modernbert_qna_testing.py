from transformers.models.modernbert.modular_modernbert import ModernBertForQuestionAnswering
from transformers import AutoTokenizer, pipeline
import logging
import transformers

# IMPORTANT: There is an error in recent versions of transformers that links the model to the gemma config
# After testing a few different transformers versions, 4.57.1 appears to be stable and supporting the model
# PR merged orignally (March 25th, 2025) that implemented the question answering config for modernbert:
# https://github.com/huggingface/transformers/pull/35566

# Load the model and tokenizer
model_id = "rankyx/ModernBERT-QnA-base-squad"
model = ModernBertForQuestionAnswering.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize the question-answering pipeline
question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Example input
question = "Which Frenchman was the founder of the modern games?"
context = "Pierre de Coubertin was the French educator and historian who founded the modern Olympic Games. Born into an aristocratic family in Paris in 1863, he became inspired by the British approach to physical education during visits to England and developed a vision of reviving the ancient Greek athletic tradition as a way to promote international peace and character development. His persistence led to the establishment of the International Olympic Committee in 1894 and the first modern Olympics in Athens in 1896."

# Get the answer
result = question_answerer(question=question, context=context)
print(result)
