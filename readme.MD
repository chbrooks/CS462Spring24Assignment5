### Assignment 5: HMMs and Decisions

#### Due: Wed April 17, 11:59pm. 

#### 100 points.

For questions 1 and 3, please include a PDF with your answers.

You do not need a submission.py for question 2, but your code does need to work 
according to the specified command line arguments.

**Question 1: Decisions** 

Our Mars rover has been out collecting samples, and it 
needs to return to the charging station as quickly as possible. 

It knows that over rocky terrain it can go 2 km/h. Over sandy terrain it can go 3 km/h,
and over smooth terrain it can go 5 km/h. 

There are three routes it might choose from. Unfortunately, our terrain data for the three routes is incomplete,
so we only have estimates.

Route 1 is 2 km long. There is a 20% chance it is sandy, 30% chance it is smooth, and a 50% chance it is rocky.

Route 2 is 1.8 km long. There is a 40% chance it is sandy, a 20% chance it is smooth, and a 40 % chance it is rocky. 

Route 3 is 3.1 km long. There is a 50% chance it is sandy, a 40% chance it is smooth, and a 10% chance it is rocky.

**(10 points)** Which route should we pick? Show your work.


We have now found out some additional information. 

Route 1 contains a crater. If the wall of the crater is intact, we can go through it. If the wall has been damaged, we will need to go around, which will add 45 minutes to our journey. There is a 30% chance that the wall is damaged.

Route 2 contains a bridge. If that bridge is damaged, we will need to repair it, which will add 1 hour to our time. There is a 60% chance that the bridge is out.

**(10 points)** Now which route should we pick? Show your work.


**(10 points)** Now suppose that we can use a satellite to find out whether the terrain in route 3 is smooth. 
Is this helpful? What is the value of this information? Expressed differently, how long are we
willing to wait for this information from the satellite?

**(5 points)** Now put this problem into your favorite LLM. Include your prompt and the result. 
Is it able to solve it correctly? If not, where does it make mistakes?

**Question 2: Hidden Markov Models** 

(Note: this is derived from an assignment in AAAI's Model Assignments workshop)

In this assignment you'll be implementing two algorithms associated with Hidden Markov Models.

You'll be building off of the code presented in HMM.py. There's also some included data to use.

The first set of files are called cat.trans and cat.emit They encode the HMM used in the cat example in the lecture slides, and are useful when you're debugging your HMM.

The second set of files are called partofspeech.browntags.trained.emit and partofspeech.browntags.trained.trans; they encode an
HMM trained to recognize parts of speech.  This problem is larger, with 12 states and lots of possible emissions.

The .trans files contain the transition probabilities.  There's also a '#' which is 
used to represent the starting state.

The .emit files contain the probability of emitting a particular output from that state. 

The last pair of files is ambiguous_sents.obs and ambiguous_sents.tagged.obs. This is what you'll test your Viterbi implementation on.
(For the cat, you should create your own .obs files to test it as needed.)

**(10 points)**. Use the included code to implement load. Use cat as a sample file to work with. You should be able to do:

model = HMM()
model.load('cat')

You should store the transitions and emissions as dictionaries of dictionaries. e.g. 

    {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'}, 
    'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'}, 
    'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}


**(15 points)** Implement generate. It should take an integer n, and return a random observation of length n. 
To generate this, start in the initial state and repeatedly select successor states at random, 
using the probability as a weight, and then select an emission, again using the probability as a weight. You may find 
either numpy.random.choice or random.choices very helpful here.
Be sure that you are using the transition probabilities to determine the next state, and not a uniform distribution!

You should be able to run it with the pre-trained probabilities for the Brown corpus, like so:

python hmm.py partofspeech.browntags.trained --generate 20

which generates 20 random observations.

Here are two sample observations:

DET ADJ . ADV ADJ VERB ADP DET ADJ NOUN VERB ADJ NOUN 

the semi-catatonic , quite several must of an western bridge cannot spectacular analyses 

DET NOUN ADP NOUN CONJ DET VERB DET NOUN NOUN NOUN ADP DET NOUN 

whose light for wall and the learned the hull postmaster trash in his peters


**(10 points)** Next, implement the forward algorithm. This tells us, for a sequence of observations, the most likely
final state. You should be able to run this like so:

python hmm.py partofspeech.browntags.trained --forward ambiguous_sents.obs

or 

python hmm.py cat -- forward your_sample_cat.obs

**(15 points)** Next, implement Viterbi. This tells us, for a sequence of observations, the most likely sequence of states. 
You should be able to run this like so:

python hmm.py partofspeech.browntags.trained --viterbi ambiguous_sents.obs

This uses the HMM parameters in partofspeech.browntags.trained.{trans,emit} to compute the best sequence of part-of-speech 
tags for each sentence in ambiguous_sents.obs, and prints the results.

You might find it helpful to use a numpy array to hold the matrix.

You can compare your results to ambiguous_sents.tagged.obs.


**(Problem 3 - 15 points)**

[AINow](https://ainowinstitute.org/) is a research institute that produces policy analysis addressing the concentration of power in the tech industry.
They have recently published a landscape report assessing the state of the AI industry and making policy rcommendations.

Please read the [executive summary](https://ainowinstitute.org/wp-content/uploads/2023/04/Exec-Summary-AI-Now-2023-Landscape-Report-.pdf) and answer the following questions:

- What are the three dimensions along which Big Tech has an advantage in AI?
- Why does AI Now think it's important to focus on Big Tech?
- Priority 1 discusses Algorithmic Accountability. What does this mean? Why is it important to shift responsibility 
for detecting harm on companies themselves?
- What are the windows for action that are identified? Which do you personally think are the most effective or promising?

- The executive summary contains this quote:

"These are only a handful of examples, and what they make clear is that there is nothing about
artificial intelligence that is inevitable. Only once we stop seeing AI as synonymous with progress
can we establish popular control over the trajectory of these technologies and meaningfully confront
their serious social, economic, and political impacts—from exacerbating patterns of inequality in
housing, credit, healthcare, and education to inhibiting workers’ ability to organize and incentivizing
content production that is deleterious to young people’s mental and physical health."

Do you agree with this assessment? How might we rethink our relationship with AI and with technology in order to avoid these potential negative outcomes?

Now paste this last question into ChatGPT and include its response. How do you compare its output to your own? 

**(Problem 4 - grad students only)** 
- An alternative approach to tacking large AI problems and the challenges that come with centralized power is the development of *decentralized AI*. 
This is an approach that integrates blockchain technology to distribute computing and data across a large number of participants. 
Sometimes this is also referred to as *federated learning*.

[This paper](https://dl.acm.org/doi/pdf/10.1145/3614407.3643701) provides an overview of the challenges and benefits of 
federated learning. Please read it and answer the following questions:

- What problems do the authors believe decentralized AI will address?
- How can federated learning help to address privacy concerns?
- The authors talk about how federated learning might open the door for new 
types of horizontal collaborations. What do they mean by this?
- Originally, the thought was that decentralized AI would allow participants with fewer resources
to have a fair shot at contributing to AI development. The authors identify two reasons why this 
has not happened yet. What are they?


