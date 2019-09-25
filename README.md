# RoPose Greenscreener - A Background Augmentation tool
This repository contains the Background Augmentation tool we developed for the RoPose-System. We used it to augment and create backgrounds for simulated RoPose-Datasets

### Prerequisites 

* Python 3.6 (for Typing etc.)
* Numpy
* OpenCV

## Installing (Source)
Clone the needed repositories to your virtual environment and install requirements:

```bash
git clone https://github.com/guthom/ropose_greenscreener
cd ropose_greenscreener
pip install -r requirements.txt
```

## Usage
```python
from greenscreener.Greenscreener import Greenscreener
import cv2

screener = Greenscreener(backgroundDir="path/to/background/directory",
                         backgroundScale=(1280, 720))

image = cv2.imread("path/to/image")
image = cv2.resize(src=image, dsize=(1280, 720))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = screener.ReplaceBackground(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

cv2.imshow("Greenscreener Demo", image)

cv2.waitKey(0)
```

## Open Source Acknowledgments
This work uses parts from:
* **numpy** https://www.numpy.org/
* **OpenCV** https://opencv.org/
* **

**Thanks to ALL the people who contributed to the projects!**

## Authors

* **Thomas Gulde**

Cognitive Systems Research Group, Reutlingen-University:
https://cogsys.reutlingen-university.de/

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Citation
This work is just indirectly involved in my research.
See other repositories for specific research project and citations.

https://github.com/guthom/ropose_dataset_tools  
https://github.com/guthom/ropose_datagrabber  
https://github.com/guthom/ropose  
https://github.com/guthom/ropose_ros
