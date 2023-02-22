# Ultrasound_Needle_Insertion_Software_MQP

## Project Description

Percutaneous Nephrolithotomy (PCNL) is the removal of a kidney stone through a surgical opening in the skin. This procedure is done by visualization of the kidney stone, needle insertion to create a tract into the target calyx of the kidney, and tract dilation to facilitate kidney stone removal. Common current techniques for needle insertion include the longitudinal approach, in which the needle is inserted in front of or behind the probe, and the transverse approach, in which the needle is inserted orthogonal to the probe and the probe must be swept back and forth to keep the needle visualized. In both cases, to keep the needle visualized from the skin surface to the target calyx, careful coordination must be kept between the probe, held in one hand, and the needle, held in the other (4). The goal of this project is to ultimately create a functional model of a needle insertion system that assists the user in visualizing the needle insertion process during ultrasound-guided PCNL.

This Github holds the code involved with trying to modify and enhance the needle visualization within the realtime ultrasound images

## Setting Up Environment
1. Download Anaconda
2. Download IDE platform of choice (VSCode, Pycharm, etc..)
3. Make sure the clarius environment is the active kernel use for running the code on the preferred IDE 
4. Open the Anaconda Prompt and write this line on the terminal: conda env create -f {Directory to this repository}\clarius.yaml
5. Once the environment is installed, activate the environment by writing: conda activate clarius

## Connecting and Running the Clarius Program 
1. disable all firewalls on PC
2. open mobile hotspot on PC
3. connect clarius mobile to hotspot
4. connect clarius probe to clarius app
5. check to make sure everything is connected on PC
6. Open anaconda prompt
7. enter "conda activate clarius"
8. cd into desired directory
9. enter: "python realtime_NeedleVizAlgorithm_demo.py
10. make sure to input address and port
