# printed_characters_net

`Printed characters net` is a project that demonstrates use of neural networks for characters recognition.
The idea is to be able to detect in real time video characters like ones below:

![k](./readme/k.jpg) ![u](./readme/u.jpg) ![ba](./readme/ba.jpg) ![4](./readme/4.jpg) ![gaku](./readme/gaku.jpg)

Current network design is capable of learning to detect over 250 characters (Latin alphabet, digits, hiragana, katanaka and Jouyou Level 1 kanji) with about ~95% accuracy in ~10mins of training on MacBook Pro 2014. Another two hours of training can raise this to ~98%.

The core logic of the project is contained in scripts directory, which consists of following programs:
- `create_templates.py` - creates plain images of characters we want to recognize
- `create_templates_printout.py` - creates a pdf with template images that can be cut out to later use them in real-time detection
- `create_data.py` - using templates obtained from `create_templates_printout.py` and a camera, capture characters images that can be used for training
- `create_artificial_data.py` - given templates, create an artificially augmented dataset used for training
- `train_mnist.py` - a sanity check script to make sure our neural network can learn standard MNIST set
- `train_characters.py` - script for training neural network to detect templates
- `debug_characters_network.py` - a simple script that identifies most common classification mistakes performed by characters network
- `detection.py` - detect printed templates in real time video stream

Notes on data: 
1. You can control amount of artificial data created with constants defined in `create_artificial_data.py`. Given ~250 labels I recommend going for ~400 images per label - this should let you create all data and train the classifier to ~95% accuracy in under 15mins on an decent machine. You can get 90% results in 5mins or so when using 250 images per label.
2. Best resulst are of course obtained with real data, but even artificial data can work really well, especially if characters set is constrained to a small size, say only latin characters.

NOTE:
This project uses mostly plain numpy for neural networks code. I'm aware of frameworks like Theano and Tensorflow that could do a lot of heavy lifting for me, while at the same time providing faster execution on Nvidia GPUs, but my main goal for this project is to check my own understanding of neural networks concepts. Hence I strive to implement all steps of the learning algorithm from a scratch.
