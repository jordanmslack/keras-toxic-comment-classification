
Project 
### Background
The internet has given people anonymity in many cases and that has created ample opportunities for things to be said that many wouldn't normally be said face to face. Cyberbullying and other online forms of verbal abuse are very prevalent and can be extremely harmful. 
 
I think that machine learning can and should be used for good, and this is an attempt to contibute to the extensive efforts being put forward to change the world.
 
### Problem
"The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions..." To help with this the Conversation AI team has "built a range of publicly available models served through the Perspective API, including toxicity. But the current models still make errors, and they don?t allow users to select which types of toxicity they?re interested in finding (e.g. some platforms may be fine with profanity, but not with other types of toxic content)." - Kaggle, Overview
 
The basis has been laid for starting to solve the problem filtering abusive content online, but there is much to be done to enhance this process and provide greater protection for platforms and users. 
 
 ### Data
The data set includes ?a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are: toxic, severe_toxic, obscene, threat,  insult, identity_hate." - Kaggle Data

### Solution
The desired solution will be a "multi-headed model that?s capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate". - Kaggle, Overview 

I've opted to build a Bidirectional LTSM network with Keras and a Tensorflow backend. 
 
