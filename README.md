# mol3022-protein-secondary-structure-prediction

## Introduction

Intro

## How to run

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

**main.py** file controls the interaction between user and the prediction model. Run this for the main program functionality.

**models.py** file contains the definition of the individual models used in the ensemble prediction.

**data** (include url to data source) folder contains training and test data.

**pretrained** folder contains pretrained models that can be loaded and used without training.

**one_hot_encodings.py** file contains mappings from amino acid symbols and secondary structure symbols to corresponding one-hot encodings.

**preprocessing.py** file contains code for parsing data files and preparing the input to the different models.

**training.py** file contains code for training the different models. Run this to try out the training.

**evaluation.py** file contains functions for evaluating the performance of models. The main metrics are *specifisity* and *sensitivitiy*.

**inference.py** file contains code for making predictions with trained models. Run this to make predictions on the test set and evaluate the results.

**test_2d_cnn_only.py** (should describe if i decide to include this)
