# recognition_digit
Digit recognition using PyBrain Machine Learning library trained on MNIST images collection.

Autor: Nikita Ivanov (Gangashman) ganagsh@gmail.com

Dependencies
-----------

Install PyBrain:
```bash
git clone git://github.com/pybrain/pybrain.git
python setup.py install
```

Mnist library and matplotlib for visualization:
```bash
pip install mnist
pip install python-mnist
pip install matplotlib
```

Tools for working with data:
```bash
sudo apt install gfortran libblas-dev liblapack-dev libatlas-dev
pip install numpy scipy pillow

```

Test
-----------
Result: network result
Value: number on image
~42 seconds per epoch

epoch: 100:
<div align="center"><img src="https://github.com/gangashman/recognition_digit/blob/master/screenshots/Screenshot_20190313_012706.png"/></div>

epoch: 200:
<div align="center"><img src="https://github.com/gangashman/recognition_digit/blob/master/screenshots/Screenshot_20190313_025324.png"/></div>

epoch: 400:
<div align="center"><img src="https://github.com/gangashman/recognition_digit/blob/master/screenshots/Screenshot_20190313_053348.png"/></div>
