# PepMSND: Integrating Multi-level Feature Engineering and Comprehensive Databases to Enhance in vitro/in vivo Peptide Blood Stability Prediction

This is the code for "PepMSND: Integrating Multi-level Feature Engineering and Comprehensive Databases to Enhance in vitro/in vivo Peptide Blood Stability Prediction" paper.

## Code Hierarchy

```shell
bashCopy code/
├── Baseline models/            
├── Dataset/  
├── Models/ 
├── Peptide structure Dataset/  
├── Vocab.txt/   
├── LICENSE        
└── README.md       
```

## Instructions

### Install dependency

Running this  command for installing dependency in docker:

```shell
pip install requirments.txt
./replace.sh
```

### Dataset

```shell
The "./Dataset" directory contains a complete set of 635 data points regarding the stability of peptides in blood. For an easy-to-browse version of the dataset, you can visit http://model.highslab.com/static/Database.html.
```
### Training model

Running this  command for training the PepMSND model：

```sh
python ./Models/model.py
```

### Using PepMSND model

```sh
We have specially developed an online service platform for the PepMSND model to provide users with easier access and a trial experience of the model. If you are interested in exploring the capabilities of the PepMSND model, simply click on the following link: http://model.highslab.com/static/service, to access the usage interface.
```

## Support or Report Issues

If you encounter any issues or need support while using PepMSND, please report the issue in the [GitHub Issues](https://github.com/your_username/PepMSND/issues) .

## Copyright and License

This project is governed by the terms of the MIT License. Prior to utilization, kindly review the LICENSE document for comprehensive details and compliance instructions.

## Version History

- v1.0.0 (2025-01-14):



