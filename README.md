# mol3022-protein-secondary-structure-prediction

## Introduction

This is a sommand line software that can be used to predict the secondary structure of a protein sequence based on the sequence of amino acids.
It uses an ensamble of different neural network strategies to make these predictions.

### Interacting with the software

- The user inputs an amino acid sequence.
- The software returns a prediction of the secondary structure sequence.
- The user may choose to compare the prediction with the true secondary structure sequence, given that the user already have this sequence.
- The user input the true secondary structure sequence
- The software returns
  - The number of "misses" in the prediction
  - The location of these misses in the sequence, marked by "X"
  - The true secondary structure sequence
  - The predicted secondary structure sequence

## How to run

There are two ways to run the software:

- Running the executable file `main.exe` (wait around 10-20 sec for program to start)
- Running the python file `main.py`

**Note** that using the second option requires a python environment with all libraries in the `requirements.txt` file. The first option has no such prerequisites, and is the recommended method. The end of this document contains a list of all project files with a short description of their individual purpose. It is possible to run some of these files on their own, and that will also require a fitting python environment.

### Extra

It is also possible to run a version of the software which only uses a CNN to make predictions. Similar to the main version, the methods to run this are:

- Running the executable file `test_2d_cnn_only.exe` (wait around 10-20 sec for program to start)
- Running the python file `test_2d_cnn_only.py`

## Example input and target sequences

Here are some example inputs with corresponding target value for easy testing of the program.

**Note**: The examples are from the *test* set and have not been included in model training.

### Example 1

- ENLKLGFLVKQPEEPWFQTEWKFADKAGKDLGFEVIKIAVPDGEKTLNAIDSLAASGAKGFVICTPDPKLGSAIVAKARGYDMKVIAVDDQFVNAKGKPMDTVPLVMMAATKIGERQGQELYKEMQKRGWDVKESAVMAITANELDTARRRTTGSMDALKAAGFPEKQIYQVPTKSNDIPGAFDAANSMLVQHPEVKHWLIVGMNDSTVLGGVRATEGQGFKAADIIGIGINGVDAVSELSKAQATGFYGSLLPSPDVHGYKSSEMLYNWVAKDVEPPKFTEVTDVVLITRDNFKEELEKKGLGGK
- ______eee_______hhhhhhhhhhhhhh______eee___hhhhhhhhhhhhh_________________hhhhhhhhh_____________________________hhhhhhhhhhhhhhhhhh________eee_________hhhhhhhhhhhhhh_______ee________hhhhhhhhhhhhh_______eee_______hhhhhhhh____________ee_____hhhh__________ee_______hhhhhhhhhhhhh__________________________________

### Example 2

- APAFSVSPASGASDGQSVSVSVAAAGETYYIAQCAPVGGQDACNPATATSFTTDASGAASFSFTVRKSYAGQTPSGTPVGSVDCATDACNLGAGNSGLNLGHVALTFG
- __eeeee_________eeeeeee____eeeeeee_ee__ee________eee_______eeeee___eeeee_____eeeeee______eeeee______________

### Example 3

- GFPIPDPYCWDISFRTFYTIVDDEHKTLFNGILLLSQADNADHLNELRRCTGKHFLNEQQLMQASQYAGYAEHKKAHDDFIHKLDTWDGDVTYAKNWLVNHIKTIDFKYRGKI
- __________________hhhhhhhhhhhhhhhhhhh___hhhhhhhhhhhhhhhhhhhhhhhh_____hhhhhhhhhhhhhhhh_____hhhhhhhhhhhhhh_________

### Example 4

- RDFTPPTVKILQSSCDGGGHFPPTIQLLCLVSGYTPGTINITWLEDGQVMDVDLSTASTTQEGELASTQSELTLSQKHWLSDRTYTCQVTYQGHTFEDSTKKCADSNPRGVSAYLSRPSPFDLFIRKSPTITCLVVDLAPSKGTVNLTWSRASGKPVNHSTRKEEKQRNGTLTVTSTLPVGTRDWIEGETYQCRVTHPHLPRALMRSTTKTSGPRAAPEVYAFATPEWPGSRDKRTLACLIQNFMPEDISVQWLHNEVQLPDARHSTTQPRKTKGSGFFVFSRLEVTRAEWEQKDEFICRAVHEAASPSQTVQRAVSVNPGK
- ______eeeee____________eeeeeeeeeee_____eeeee________eee___ee_____eeeeeeeeeehhhhh_____eeeee________eee__________eeeee_____________eeeeeee____________eee_____________eeee__eeeeeeeee__hhhhhh___eeeee________eeeee__________eeeee__________eeeeeeeee_______eeeee________________ee_____ee_eeeeeeehhhhh_____eeeee_________eeee_______

### Example 5

- ANIVGGIEYSINNASLCSVGFSVTRGATKGFVTAGHCGTVNATARIGGAVVGTFAARVFPGNDRAWVSLTSAQTLLPRVANGSSFVTVRGSTEAAVGAAVCRSGRTTGYQCGTITAKNVTANYAEGAVRGLTQGNACMGRGDSGGSWITSAGQAQGVMSGGNVQSNGNNCGIPASQRSSLFERLQPILSQYGLSLVTG
- _eeee__eeee___eeee__eeeee__eeeeee_________eeee__eeeeeeeeee_____eeeeee____eeeeeeee__eeee___________eeeeee___eeeeeeeeeeeeeeee__eeeeeeeee___________eee_____eeeeeeee_________________eeeeehhhhhhhh__ee___

## Project files

``main.exe`` file is the main executable for this project. This is the easiest way to run the software

``main.py`` file controls the interaction between user and the prediction model. Run this for the main program functionality.

``models.py`` file contains the definition of the individual models used in the ensemble prediction.

``data`` folder contains training and test data. [This](https://archive.ics.uci.edu/dataset/68/molecular+biology+protein+secondary+structure) is a link to the data source.

``pretrained`` folder contains pretrained models that can be loaded and used without training.

``one_hot_encodings.py`` file contains mappings from amino acid symbols and secondary structure symbols to corresponding one-hot encodings.

``preprocessing.py`` file contains code for parsing data files and preparing the input to the different models.

``training.py`` file contains code for training the different models. Run this to try out the training.

``evaluation.py`` file contains functions for evaluating the performance of models. The main metrics are *specifisity* and *sensitivitiy*.

``inference.py`` file contains code for making predictions with trained models. Run this to make predictions on the test set and evaluate the results.

``test_2d_cnn_only.py`` file is a version of main.py that makes predictions only with the 2D CNN.

``test_2d_cnn_only.exe`` file is the easiest way to run the CNN version of the software.
