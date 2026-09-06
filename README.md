## Financial_QA_10k pipeline, OpenAI vs Gemini AI model comparsion ##

This assement project comapres open_ai and gemini_ai on 50 financial _qa_10K dataset.it include abstention logic so models can decline to answer when the given context lacks sufficient information.

# setup instructions  #

create a virtual envirment activate it then run pip install dash r requirments dot text to install dependencies
add openai and gemini api keys into a dot env file inthe project root.

# how to run #

run src/__models__.py first this generates results.csv with predictions from both models
then run python src/evaluate.py to print the rouge1, rouge2, rougeL scores comparing both models aginst the refernces answers

# abstention logic #

both models are prompted to respond with exactly cannot determine from context when the given context does not contain enough information to answer the question the pipline detects  this pharse in each response and records it as an abstained answer for that models.

# space result #
OpenAI average rouge1 score : 0.44134034844680355
gemini average rouge1 score : 0.5122530782858322
OpenAI average rouge2 score : 0.366043656942121
gemini average rouge2 score : 0.4071953046639357
OpenAI average rougeL score : 0.41942919128732414
gemini average rougeL score : 0.48551303244993177

overall gemini 3.6 flash outperformed the Openai model 4o mini across all three rouge metrice in this sample suggesting its responses aligened more closely with the refernce answers under these baseline settings through real world suitabilty would also depend on cost latency,and consistency ,not rouge scores alone

note :both models were called using their default temperture and token settings, with no custom tunning applied, so these results reflects baseline preformce without optimized configurations.

