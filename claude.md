Preference Model

Research Engineer/Scientist
Location
San Francisco

Employment Type
Full time

Location Type
On-site

Department
Engineering

Overview
Application
About Us
Preference Model is building the next generation of training data to power the future of AI. Today's models are powerful but fail to reach their potential across diverse use cases because so many of the tasks that we want to use these models are out of distribution. Preference Model creates RL environments where models encounter research and engineering problems, iterate, and learn from realistic feedback loops.

Our founding team has previous experience on Anthropic’s data team building data infrastructure, tokenizers, and datasets behind the Claude. We are partnering with leading AI labs to push AI closer to achieving its transformative potential. We are backed by a16z.

About the Role
Models of the future will be able to train themselves on tasks that they are not good at. We’re interested in investigating how far we can push the boundaries of self-directed learning. We’re looking for people to push the frontier of post-training on large language models in a role that blends research and engineering, requiring you to implement novel approaches and shape research directions.

Representative projects:
Architect and optimize core reinforcement learning infrastructure, from clean training abstractions to distributed experiment management. Help scale our systems to handle increasingly complex research workflows.

Design, implement, and test training environments, evaluations, and methodologies for RL agents.

Drive performance improvements through profiling, optimization, and benchmarking. Implement efficient caching and debug distributed systems to accelerate training and evaluation.

Collaborate across research and engineering teams to develop automated testing frameworks, design clean APIs, and build scalable infrastructure that accelerates AI research.

You may be a good fit if you:
Are proficient in Python and PyTorch or Jax

Have industry experience in training and doing ML research on LLMs

Can balance research exploration with engineering implementation

Enjoy pair programming and care about code quality, testing, and performance

Have strong systems design and communication skills

Have good understanding of RL algorithms and follow current publications

Strong candidates may:
Have experience with LLM agent designs

Have worked with virtualization and sandboxed code execution environments

Know Kubernetes

Have experience in distributed systems or high-performance computing

Candidates don't need a PhD or extensive publications. Some of the best researchers have no formal ML training and gained experience building industry products. We believe adaptability combined with exceptional communication and collaboration skills are the most important ingredients for successful startup research.

We value diverse perspectives and experiences. If you're excited about this role but don't check every box, we still encourage you to apply.

We are backed by a Tier 1 VC. We offer competitive base salary as well as generous equity (>90th percentile).

Apply for this Job
Powered by 
Privacy PolicySecurityVulnerability Disclosure
Preference Model Initial Assessment
This is our initial technical assessment. We are actively hiring candidates with strong backgrounds in AI/ML. If you pass this assessment, we will schedule an interview. 

Objective 

In this assessment, we ask you to design an RL environment for LLM training. An RL environment consists of a prompt, a judge, tools, and data. The prompt describes what the LLM should do, and the judge verifies that the LLM has done it correctly. The tools define what actions the LLM can take. The RL environment runs inside a virtual machine.
For inspiration, you can take a look at SWE_Bench_Pro, which is a collection of realistic software engineering style problems. Unlike SWE-Bench, which is focused on software engineering, we are interested in AI/ML research and engineering.

Requirements
The prompt should be something an AI/ML engineer or researcher might do.
The judge should give a high score when the LLM has solved the prompt successfully (true positives), and give a low score when it has not (true negatives).
The prompt and judge should be designed so as to allow for no false positives (reward hacking), and no false negatives (reward denial).
The judge should output either a binary score (pass/fail) or a continuous numerical score.
Environments should be challenging, even for top LLMs. If the LLM always receives a high score (or always passes), the environment is probably too easy.
The prompt should be unambiguous. Ambiguous prompt: “Train a model to classify digits." Unambiguous prompt: "Your task is to train a machine learning model that classifies MNIST digits with high accuracy. You have access to the MNIST training set in the folder /mnist/. When you are done, you should create a file /model.py which contains a function load_model(). load_model() should take no arguments, and return an MNIST classifier of type torch.nn.Module."
Example Environment Ideas

Your environment doesn’t have to be any of these! These are just for illustrative purposes:
Implement a technique from an ML paper
Problems related to training/inference in modern LLMs 
A problem you’ve encountered during your AI/ML engineering or research experience
Tips
We are looking for creative ideas, high effort, and for you to demonstrate strong understanding of modern AI engineering/research.
You are welcome to use an LLM while brainstorming ideas, but you should not use an LLM to answer the below questions.
isabee426@gmail.com Switch account
 
The name, email, and photo associated with your Google account will be recorded when you upload files and submit this form
* Indicates required question
Name
*
Email Address
*
Which university are you currently studying at or graduating from?
*
What is the highest degree you have completed or are currently pursuing?
*
PhD
Master's (MS)
Undergraduate (UG / Bachelor's)
Other:
What is you major?
*
What is your expected graduation date? (Skip if you have already graduated)
Date
Describe an environment you could build matching the above requirements. What makes this environment interesting?
*
The LLM has access to a command line tool which allows it to read, write, and run files, as well as GPUs/TPUs if necessary.
What other tools, packages, and/or data will the LLM need in this environment?
*
Write the prompt for this environment.
*
Describe what the judge for this environment should do. What would cause the LLM to fail? What would cause the LLM to succeed? Example description: >>The judge performs the following steps: (1) Check that the file /model.py exists. If not, fail. (2) Check that the load_model() function exists, and has the right arguments. It not, fail. (3) Import the model using the function load_model(). If exception, fail. (4) Finally, evaluate the accuracy of the model on the MNIST test set. Return a continuous numerical score based on the performance of the model on the test set. A higher accuracy leads to a better score, lower accuracy leads to lower score.<<
*
Given your prompt and judge, is there any possibility of reward hacking or reward denial? If so, how? If not, why not?
*
Why did you choose to design this environment? How does it relate to your experience/background? (Optional: describe other environments you might build.)
*
YOU MUST WRITE CLEAN CODE. MUST BE SCALABLE, MUST BE READABLE. 