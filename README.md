# Astronomical Data Mining Competition
- Code for Astronomical Data Mining Competition [2018 Tianchi](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100150.711.9.2d80a073zHl4Mj&raceId=231646&_lang=en_US)
- Score of this code : 66.9%
- Team rank 35/843 at 1st round competition and 19/843 at 2nd round competition

## Folder Structure
- `src`: store the source code.
`src/3classfier.py`: model definition and training for a 3-classfier to classify star, galaxy and qso.
`src/unnkow_classfier.py`: model definition and training for a 2-classfier to classify an unknown or known.
`src/predict.py`: predict class for input spectrum.

## Requirements
- python (3.5.2)
- keras (2.1.5)
- numpy (1.14.2)
- opencv-python (3.4.0.12)
- pandas (0.22.0)
- scipy (1.0.1)
- tensorflow-gpu (1.2.0)