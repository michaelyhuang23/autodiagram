from gpt_bot import GPTBot
import pandas as pd

ideaBotInstruction = "You are an AI assistant that helps researchers train another GPT model by generating good prompts. You are to help researchers generate creative prompts for various kinds of TikZ diagrams that students typically see in math and physics classes. Your prompts will be used by another GPT model to generate TikZ codes. Generate plain text. Do not enclose the prompt you generate in quotation marks. The prompts you generate should be sufficiently simple that a GPT model can easily generate the accurating TikZ code corresponding to it. And you should focus on generating geometric diagrams or physics diagrams like free body diagrams."
ideaBotPrompt = "Generate one prompt for producing a creative TikZ diagram that depicts a free body diagram one might see in a physics textbook."

ideaBot = GPTBot(ideaBotInstruction, auto_regressive=False)
print(ideaBot.gen_completion(ideaBotPrompt))

tikzBotInstruction = "You are an AI assistant that generates TikZ codes. You will be given a prompt for a diagram, and you will generate a compilable TikZ code that produces the diagram as described. Your TikZ code should represent the text describing the diagram as accurately as possible. It should follow standard formatting. And it should produce a diagram that is easy to read for the human eye (e.g. no overlapping elements). Generate a chain of thoughts describing the diagram before outputing the latex code."
tikzBot = GPTBot(tikzBotInstruction, auto_regressive=False)

df = pd.DataFrame(columns=['idea', 'response'])
ideas = []
tikzcodes = []
for i in range(10):
    idea = ideaBot.gen_completion(ideaBotPrompt)
    completion = tikzBot.gen_completion(idea)
    ideas.append(idea)
    tikzcodes.append(completion)

df['idea'] = ideas
df['response'] = tikzcodes

df.to_csv('data/tikz_gen_gpt4.csv', index=False)



