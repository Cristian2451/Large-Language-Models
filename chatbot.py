import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from user_interface import ChatBot

tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
model = BertForQuestionAnswering.from_pretrained("bert-base-cased")
print(model.load_state_dict(torch.load("models/trained_model_state_dict.pt", map_location=torch.device('cpu'))))
context = """
Leonardo da Vinci (15 April 1452 – 2 May 1519) was an Italian polymath of the High Renaissance who was active 
as a painter, draughtsman, engineer, scientist, theorist, sculptor, and architect.
While his fame initially rested on his achievements as a painter, he has also become known for his notebooks, 
in which he made drawings and notes on a variety of subjects, including anatomy, astronomy, botany, cartography, 
painting, and paleontology. Leonardo epitomized the Renaissance humanist ideal, and his collective works comprise 
a contribution to later generations of artists matched only by that of his younger contemporary Michelangelo.
""".replace("\n", " ")

bot = ChatBot(context, tokenizer, model, max_len=300)
questions = [
    "Who was Leonardo da Vinci?",
    "When was Leonardo da Vinci born?",
    "Who contributed to Leonardo da Vinci's work?",
]

answers = bot.answer(questions, disable_progress_bar=False)
for answer in answers:
        print("Answer is: ", answer)
