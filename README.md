# reflex_attention
## Обучение модифицированной архитектуры декодера для генерации текста

Based on nanoGPT, dataset - openwebtext

## Motivation
- due to a calculation error with model fleets, it is difficult to fit information about a large context in one hidden ([FACTS](http://arxiv.org/pdf/2406.04267 ))
- Decoder layers are trained to predict the following tokens, so the model forgets information from the context when doing only self-attention

## Idea of Reflex attention

