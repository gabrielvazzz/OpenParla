hey, can you do a code review on this codebase and create a REVIEW.md file with the findings you have? the main things i'd like you to search and understand here are:

- threats to validity: are there any things that make this training not work? what worries are here?
- bias: we know that this type of training and data processing can introduce biases, and we don't negate that. but i wanted to understand how to explain how we based ourselves in real data and tried to make justice as much as we could
- correctness: are there any correctness flaws in the code? in the way we treat vectors, code, or anything like that? 

i just want a final judgement on the final state of the code and results that we had. fyi, the results we had here were similar to the naive approach of simply inputting these data into a GPT 4o model withou any LLMs.

feel free to use sub-agents for this so we have a good and broad final review. the final file shouldn't be too verbose, it should have direct points and explain things clearly, beginning with a brief introduction of what this codebase wants to make. focus on the content of the `lorenna-treino` branch.
