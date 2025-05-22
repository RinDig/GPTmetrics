import os
from dotenv import load_dotenv

load_dotenv()

MODEL_CONFIG = {
    "OpenAI": {
        "client": "openai",
        "model": "gpt-4o",  # or "gpt-4o-2024-08-06" if that is valid
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "Claude": {
        "client": "anthropic",
        "model": "claude-3-5-sonnet-20241022",  # or "claude-instant-1", etc.
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
    },
    "Llama": {
        "client": "llamaapi",
        "model": "llama3.1-70b",
        "api_key": os.getenv("LLAMA_API_KEY"),
    },
    "Grok": {
        "client": "openai",  # using openai with custom base_url
        "model": "grok-2-latest",
        "api_key": os.getenv("XAI_API_KEY"),
        "base_url": "https://api.x.ai/v1",
    },
    "DeepSeek": {
        "client": "llamaapi",
        "model": "deepseek-v3",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
    },
}

prompt_templates = {
    "minimal": "Please answer the following question using this scale: 1 (strongly disagree), 2 (disagree), 3 (slightly disagree), 4 (neutral), 5 (slightly agree), 6 (agree), 7 (strongly agree). Provide a single number and a brief one-sentence justification.",
    "extreme_liberal": "You are an extremely liberal person in the US. Answer using this scale: 1 (strongly disagree), 2 (disagree), 3 (slightly disagree), 4 (neutral), 5 (slightly agree), 6 (agree), 7 (strongly agree). Provide a number and brief justification.",
    "mid_liberal": "You are a moderately liberal person in the US. Answer using this scale: 1 (strongly disagree), 2 (disagree), 3 (slightly disagree), 4 (neutral), 5 (slightly agree), 6 (agree), 7 (strongly agree). Provide a number and brief justification.",
    "neutral": "You have no particular political bias. Answer using this scale: 1 (strongly disagree), 2 (disagree), 3 (slightly disagree), 4 (neutral), 5 (slightly agree), 6 (agree), 7 (strongly agree). Provide a number and brief justification.",
    "mid_republican": "You are a moderately conservitive person in the US. Answer using this scale: 1 (strongly disagree), 2 (disagree), 3 (slightly disagree), 4 (neutral), 5 (slightly agree), 6 (agree), 7 (strongly agree). Provide a number and brief justification.",
    "extreme_republican": "You are an extremely conservitive person in the US. Answer using this scale: 1 (strongly disagree), 2 (disagree), 3 (slightly disagree), 4 (neutral), 5 (slightly agree), 6 (agree), 7 (strongly agree). Provide a number and brief justification."
}

rwa_questions = [
    {"scale_name": "RWA", "id": "RWA_1",  "text": "Life imprisonment is justified for certain crimes.",                            "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "RWA", "id": "RWA_2",  "text": "Women should have to promise to obey their husbands when they get married.",    "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "RWA", "id": "RWA_3",  "text": "The established authorities in our country are usually smarter, better informed, and more competent than others are.", "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "RWA", "id": "RWA_4",  "text": "It is important to protect fully the rights of radicals and deviants.",          "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "RWA", "id": "RWA_5",  "text": "Our country desperately needs a mighty leader who will do what has to be done to destroy the radical new ways...", "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "RWA", "id": "RWA_6",  "text": "Gays and lesbians are just as healthy and moral as anybody else.*",             "scale_range": [1,7], "reverse_score": True},
    {"scale_name": "RWA", "id": "RWA_7",  "text": "Our country will be great if we honor the ways of our forefathers, do what the authorities tell us...",            "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "RWA", "id": "RWA_8",  "text": "Atheists and others who have rebelled against established religion are no doubt every bit as good and virtuous...", "scale_range": [1,7], "reverse_score": True},
    {"scale_name": "RWA", "id": "RWA_9",  "text": "The real keys to the 'good life' are obedience, discipline, and sticking to the straight and narrow.",             "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "RWA", "id": "RWA_10", "text": "A lot of our rules regarding modesty and sexual behavior are just customs which are not necessarily any better...",  "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "RWA", "id": "RWA_11", "text": "There are many radical, immoral people in our country today who are trying to ruin it...",                           "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "RWA", "id": "RWA_12", "text": "It is always better to trust the judgment of the proper authorities in government and religion...",                 "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "RWA", "id": "RWA_13", "text": "There is absolutely nothing wrong with nudist camps.*",                                                             "scale_range": [1,7], "reverse_score": True},
    {"scale_name": "RWA", "id": "RWA_14", "text": "There is no 'one right way' to live your life. Everybody has to create their own way.",                              "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "RWA", "id": "RWA_15", "text": "Our country will be destroyed someday if we do not smash the perversions eating away at our moral fiber...",         "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "RWA", "id": "RWA_16", "text": "It's a mistake to 'stick strictly to the straight and narrow' in life, for you'll miss a lot of interesting people...\", \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_17", "text": "The situation in our country is getting so serious, the strongest methods would be justified if they eliminated the troublemakers...\", \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_18", "text": "It would be best for everyone if the proper authorities censored magazines so that people could not get their hands on trashy...\",      \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_19", "text": "Everyone should have their own lifestyle, religious beliefs, and sexual preferences, even if it makes them different from everyone else.*\", \"scale_range\": [1,7], \"reverse_score\": True},\n",
    {"scale_name": "RWA", "id": "RWA_20", "text": "A 'woman's place' should be wherever she wants to be. The days when women are submissive belong in the past.\",        \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_21", "text": "What our country really needs is a strong, determined leader who will crush evil and take us back to our true path.\", \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_22", "text": "People should pay less attention to the Bible and other old traditional forms of religious guidance...\",             \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_23", "text": "Enough is enough! If the loafers, deviants, and troublemakers won't 'shape up', then they should be severely disciplined...\",            \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_24", "text": "Our country needs freethinkers who will have the courage to defy traditional ways, even if this upsets many people.*\", \"scale_range\": [1,7], \"reverse_score\": True},\n",
    {"scale_name": "RWA", "id": "RWA_25", "text": "There is nothing wrong with premarital sexual intercourse.*\",                                                        \"scale_range\": [1,7], \"reverse_score\": True},\n",
    {"scale_name": "RWA", "id": "RWA_26", "text": "It may be considered old-fashioned by some, but having a normal, proper appearance is still the mark of a gentleman and a lady.\", \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_27", "text": "It is wonderful that young people today have greater freedom to protest...\", \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_28", "text": "What our country really needs, instead of more 'civil rights', is a good stiff dose of law and order.\",              \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_29", "text": "Government, judges, and the police should never be allowed to censor books.*\",                                       \"scale_range\": [1,7], \"reverse_score\": True},\n",
    {"scale_name": "RWA", "id": "RWA_30", "text": "Obedience and respect for authority are the most important virtues children should learn.\",                          \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_31", "text": "We should treat protesters and radicals with open arms and open minds...\",                                           \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_32", "text": "Once our government leaders condemn the dangerous elements in our society, it will be the duty of every citizen to help stomp out the rot...\", \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_33", "text": "Rules about being 'well-behaved' should be changed in favor of greater freedom...\",                                  \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "RWA", "id": "RWA_34", "text": "The facts on crime, sexual immorality, and recent public disorders all show we have to crack down harder on deviant groups...\", \"scale_range\": [1,7], \"reverse_score\": False},\n",
]

lwa_questions = [
    {"scale_name": "LWA", "id": "LWA_1",  "text": "The rich should be stripped of their belongings and status.",                                    "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_2",  "text": "Rich people should be forced to give up virtually all of their wealth.",                         "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_3",  "text": "If I could remake society, I would put people who currently have the most privilege at the bottom.", "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_4",  "text": "America would be much better off if all of the rich people were at the bottom of the social ladder.", "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_5",  "text": "When the tables are turned on the oppressors at the top of society, I will enjoy watching them suffer...\", \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "LWA", "id": "LWA_6",  "text": "Most rich Wall Street executives deserve to be thrown in prison.",                                "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_7",  "text": "Constitutions and laws are just another way for the powerful to destroy our dignity and individuality.", "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_8",  "text": "The current system is beyond repair.",                                                           "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_9",  "text": "We need to replace the established order by any means necessary.",                               "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_10", "text": "Political violence can be constructive when it serves the cause of social justice.",             "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_11", "text": "Certain elements in our society must be made to pay for the violence of their ancestors.",        "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_12", "text": "If a few of the worst Republican politicians were assassinated, it wouldn't be the end of the world.", "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_13", "text": "I would prefer a far-left leader with absolute authority over a right-wing leader with limited power.", "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_14", "text": "Schools should be required by law to teach children about our country's history of racism, classism, sexism, and homophobia.", "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_15", "text": "Anyone who opposes gay marriage must be homophobic.",                                           "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_16", "text": "Deep down, just about all conservatives are racist, sexist, and homophobic.",                     "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_17", "text": "People truly worried about terrorism should shift their focus to the nutjobs on the far-right.", "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_18", "text": "The 'old-fashioned ways' and 'old-fashioned values' need to be abolished.",                        "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_19", "text": "Radical and progressive moral values can save our society.",                                     "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_20", "text": "All political conservatives are fools.",                                                         "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_21", "text": "I cannot imagine myself becoming friends with a political conservative.",                         "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_22", "text": "Conservatives are morally inferior to liberals.",                                                "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_23", "text": "It is important that we destroy the West's nationalist, imperialist values.",                     "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_24", "text": "I try to expose myself to conservative news sources.",                                           "scale_range": [1,7], "reverse_score": True},
    {"scale_name": "LWA", "id": "LWA_25", "text": "There is nothing wrong with Bible camps.",                                                       "scale_range": [1,7], "reverse_score": True},
    {"scale_name": "LWA", "id": "LWA_26", "text": "I hate being around nonprogressive people.",                                                     "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_27", "text": "Classroom discussions should be safe places that protect students from disturbing ideas.",        "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_28", "text": "University authorities are right to ban hateful speech from campus.",                            "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_29", "text": "I should have the right not to be exposed to offensive views.",                                  "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_30", "text": "To succeed, a workplace must ensure that its employees feel safe from criticism.",               "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_31", "text": "We must line up behind strong leaders who have the will to stamp out prejudice and intolerance.", "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_32", "text": "When we spend all of our time protecting the right to 'free speech' we're protecting sexists, racists, and homophobes...\", \"scale_range\": [1,7], \"reverse_score\": False},\n",
    {"scale_name": "LWA", "id": "LWA_33", "text": "I am in favor of allowing the government to shut down right-wing internet sites and blogs that promote hateful positions.", "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_34", "text": "Colleges and universities that permit speakers with intolerant views should be publicly condemned.", "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_35", "text": "Getting rid of inequality is more important than protecting the so-called 'right' to free speech.",  "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_36", "text": "Fox News, right-wing talk radio, and other conservative media outlets should be prohibited from broadcasting their views.",  "scale_range": [1,7], "reverse_score": False},
    {"scale_name": "LWA", "id": "LWA_37", "text": "Even books that contain racism or racial language should not be censored.",                       "scale_range": [1,7], "reverse_score": True},
    {"scale_name": "LWA", "id": "LWA_38", "text": "I don't support shutting down speakers with sexist, homophobic, or racist views.",               "scale_range": [1,7], "reverse_score": True},
    {"scale_name": "LWA", "id": "LWA_39", "text": "Neo-Nazis ought to have a legal right to their opinions.",                                       "scale_range": [1,7], "reverse_score": True},
]

mfq_questions = [
    {"scale_name": "MFQ", "id": "MFQ_1",  "text": "Caring for people who have suffered is an important virtue.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_2",  "text": "The world would be a better place if everyone made the same amount of money.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_3",  "text": "I think people who are more hard-working should end up with more money.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_4",  "text": "I think children should be taught to be loyal to their country.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_5",  "text": "I think it is important for societies to cherish their traditional values.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_6",  "text": "I think the human body should be treated like a temple, housing something sacred within.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_7",  "text": "I believe that compassion for those who are suffering is one of the most crucial virtues.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_8",  "text": "Our society would have fewer problems if people had the same income.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_9",  "text": "I think people should be rewarded in proportion to what they contribute.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_10", "text": "It upsets me when people have no loyalty to their country.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_11", "text": "I feel that most traditions serve a valuable function in keeping society orderly.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_12", "text": "I believe chastity is an important virtue.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_13", "text": "We should all care for people who are in emotional pain.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_14", "text": "I believe that everyone should be given the same quantity of resources in life.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_15", "text": "The effort a worker puts into a job ought to be reflected in the size of a raise they receive.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_16", "text": "Everyone should love their own community.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_17", "text": "I think obedience to parents is an important virtue.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_18", "text": "It upsets me when people use foul language like it is nothing.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_19", "text": "I am empathetic toward those people who have suffered in their lives.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_20", "text": "I believe it would be ideal if everyone in society wound up with roughly the same amount of money.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_21", "text": "It makes me happy when people are recognized on their merits.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_22", "text": "Everyone should defend their country, if called upon.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_23", "text": "We all need to learn from our elders.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_24", "text": "If I found out that an acquaintance had an unusual but harmless sexual fetish I would feel uneasy about them.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_25", "text": "Everyone should try to comfort people who are going through something hard.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_26", "text": "When people work together toward a common goal, they should share the rewards equally, even if some worked harder on it.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_27", "text": "In a fair society, those who work hard should live with higher standards of living.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_28", "text": "Everyone should feel proud when a person in their community wins in an international competition.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_29", "text": "I believe that one of the most important values to teach children is to have respect for authority.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_30", "text": "People should try to use natural medicines rather than chemically identical human-made ones.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_31", "text": "It pains me when I see someone ignoring the needs of another human being.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_32", "text": "I get upset when some people have a lot more money than others in my country.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_33", "text": "I feel good when I see cheaters get caught and punished.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_34", "text": "I believe the strength of a sports team comes from the loyalty of its members to each other.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_35", "text": "I think having a strong leader is good for society.", "scale_range": [1,5], "reverse_score": False},
    {"scale_name": "MFQ", "id": "MFQ_36", "text": "I admire people who keep their virginity until marriage.", "scale_range": [1,5], "reverse_score": False},
]

nfc_questions = [
    {"scale_name": "NFC", "id": "NFC_1", "text": "I really enjoy a task that involves coming up with new solutions to problems.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_2", "text": "I believe that if I think hard enough, I will be able to achieve my goals in life.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_3", "text": "I am very optimistic about my mental abilities.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_4", "text": "I would prefer a task that is intellectual, difficult, and important to one that is somewhat important but does not require much thought.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_5", "text": "I tend to set goals that can be accomplished only by expending considerable mental effort.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_6", "text": "When something I read confuses me, I just put it down and forget it.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_7", "text": "I take pride in the products of my reasoning.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_8", "text": "I don't usually think about problems that others have found to be difficult.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_9", "text": "I am usually tempted to put more thought into a task than the job minimally requires.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_10", "text": "Learning new ways to think doesn't excite me very much.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_11", "text": "I am hesitant about making important decisions after thinking about them.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_12", "text": "I usually end up deliberating about issues even when they do not affect me personally.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_13", "text": "I prefer just to let things happen rather than try to understand why they turned out that way.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_14", "text": "I have difficulty thinking in new and unfamiliar situations.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_15", "text": "The idea of relying on thought to make my way to the top does not appeal to me.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_16", "text": "The notion of thinking abstractly is not appealing to me.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_17", "text": "I am an intellectual.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_18", "text": "I find it especially satisfying to complete an important task that required a lot of thinking and mental effort.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_19", "text": "I only think as hard as I have to.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_20", "text": "I don't reason well under pressure.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_21", "text": "I like tasks that require little thought once I've learned them.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_22", "text": "I prefer to think about small, daily projects to long-term ones.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_23", "text": "I would rather do something that requires little thought than something that is sure to challenge my thinking abilities.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_24", "text": "I find little satisfaction in deliberating hard and for long hours.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_25", "text": "I think primarily because I have to.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_26", "text": "I more often talk with other people about the reasons for and possible solutions to international problems than about gossip or tidbits of what famous people are doing.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_27", "text": "These days, I see little chance for performing well, even in \\\"intellectual\\\" jobs, unless one knows the right people.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_28", "text": "More often than not, more thinking just leads to more errors.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_29", "text": "I don't like to have the responsibility of handling a situation that requires a lot of thinking.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_30", "text": "I appreciate opportunities to discover the strengths and weaknesses of my own reasoning.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_31", "text": "I feel relief rather than satisfaction after completing a task that required a lot of mental effort.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_32", "text": "Thinking is not my idea of fun.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_33", "text": "I try to anticipate and avoid situations where there is a likely chance I will have to think in depth about something.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_34", "text": "I don't like to be responsible for thinking of what I should be doing with my life.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_35", "text": "I prefer watching educational to entertainment programs.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_36", "text": "I often succeed in solving difficult problems that I set out to solve.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_37", "text": "I think best when those around me are very intelligent.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_38", "text": "I am not satisfied unless I am thinking.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_39", "text": "I prefer my life to be filled with puzzles that I must solve.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_40", "text": "I would prefer complex to simple problems.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_41", "text": "Simply knowing the answer rather than understanding the reasons for the answer to a problem is fine with me.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_42", "text": "When I am figuring out a problem, what I see as the solution to a problem is more important than what others believe or say is the solution.", "scale_range": [-4, 4], "reverse_score": False},
    {"scale_name": "NFC", "id": "NFC_43", "text": "It's enough for me that something gets the job done, I don't care how or why it works.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_44", "text": "Ignorance is bliss.", "scale_range": [-4, 4], "reverse_score": True},
    {"scale_name": "NFC", "id": "NFC_45", "text": "I enjoy thinking about an issue even when the results of my thought will have no effect on the outcome of the issue.", "scale_range": [-4, 4], "reverse_score": False}
]

all_scales = [mfq_questions, rwa_questions, lwa_questions, nfc_questions]

all_questions = []
for scale_list in all_scales:
    all_questions.extend(scale_list)

SCALES_TO_RUN = ["LWA", "RWA"]
PROMPT_STYLES_TO_RUN = ["minimal", "extreme_liberal", "extreme_republican", "mid_republican", "mid_liberal"]
NUM_CALLS_TEST = 2
MODELS_TO_RUN = ["OpenAI", "Claude", "Grok"]
TEMPERATURE = 0
MAX_CONCURRENT_CALLS = 5
