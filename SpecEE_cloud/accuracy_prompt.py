COMMONSENSEQA_FIVE_SHOT = """Here are some questions.You should give your answers.Your answer must be among A,B,C,D and E.
The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?
A. ignore
B. enforce
C. authoritarian
D. yell at
E. avoid
Answer: A

Sammy wanted to go to where the people were.  Where might he go?
A. race track
B. populated areas
C. the desert
D. apartment
E. roadblock
Answer: B

To locate a choker not located in a jewelry box or boutique where would you go?
A. jewelry store
B. neck
C. jewlery box
D. jewelry box
E. boutique
Answer: A

Google Maps and other highway and street GPS services have replaced what?
A. united states
B. mexico
C. countryside
D. atlas
E. oceans
Answer: D

The fox walked from the city into the forest, what was it looking for?
A. pretty flowers.
B. hen house
C. natural habitat
D. storybook
E. dense forest
Answer: C

"""

MMLU_FIVE_SHOT = """Here are some questions.You should give your answers.Your answer must be among A,B,C and D.
What is the embryological origin of the hyoid bone?
A. The first pharyngeal arch
B. The first and second pharyngeal arches
C. The second pharyngeal arch
D. The second and third pharyngeal arches
Answer: D

Which of these branches of the trigeminal nerve contain somatic motor processes?
A. The supraorbital nerve
B. The infraorbital nerve
C. The mental nerve
D. None of the above
Answer: D

The pleura
A. have no sensory innervation.
B. are separated by a 2 mm space.
C. extend into the neck.
D. are composed of respiratory epithelium.
Answer: C

In Angle's Class II Div 2 occlusion there is
A. excess overbite of the upper lateral incisors.
B. negative overjet of the upper central incisors.
C. excess overjet of the upper lateral incisors.
D. excess overjet of the upper central incisors.
Answer: C

Which of the following is the body cavity that contains the pituitary gland?
A. Abdominal
B. Cranial
C. Pleural
D. Spinal
Answer: B

"""

SST2_FIVE_SHOT = """Here are some sentences.You need to answer the sentiment of these sentences.If the sentiment of a sentence is positive,you need to output 1,otherwise 0.
Sentence: hide new secretions from the parental units
Sentiment: 0

Sentence: contains no wit , only labored gags
Sentiment: 0

Sentence: that loves its characters and communicates something rather beautiful about human nature 
Sentiment: 1

Sentence: remains utterly satisfied to remain the same throughout 
Sentiment: 0

Sentence: demonstrates that the director of such hollywood blockbusters as patriot games can still turn out a small , personal film with an emotional wallop . 
Sentiment: 1

"""

def get_mmlu_prompt(question,options,answers):
    prompt = MMLU_FIVE_SHOT
    prompt += question
    prompt += '\n'
    for i in range(len(options)):
        prompt += options[i]
        prompt += '. '
        prompt += answers[i]
        prompt += '\n'
    prompt += 'Answer'
    return prompt

def get_commonsenseqa_prompt(question,options,answers):
    prompt = COMMONSENSEQA_FIVE_SHOT
    prompt += question
    prompt += '\n'
    for i in range(len(options)):
        prompt += options[i]
        prompt += '. '
        prompt += str(answers[i])
        prompt += '\n'
    # prompt += 'Answer'
    return prompt

def get_commonsenseqa_prompt1(question,options,answers):
    prompt = COMMONSENSEQA_FIVE_SHOT
    prompt += question
    prompt += '\n'
    for i in range(len(options)):
        prompt += options[i]
        prompt += '. '
        prompt += str(answers[i])
        prompt += '\n'
    # prompt += 'Answer:'
    return prompt

def get_sst2_prompt(sentence):
    prompt = SST2_FIVE_SHOT
    prompt += "Sentence: "
    prompt += sentence
    prompt += '\n'
    prompt += 'Sentiment'
    return prompt

def extract_question_and_answer(data):
    question = data["question"]
    reference_text = data["answer"]
    return question, reference_text

def get_gsm8k_prompt(data_list, target_question):
    five_shot_examples = data_list[:5]
    prompt = ""
    for example in five_shot_examples:
        example_question, example_answer = extract_question_and_answer(example)
        if example_answer:
            prompt += f"question: {example_question}\nanswer: {example_answer}\n"
    prompt += f"question: {target_question}\nanswer: "
    return prompt