## Installing CloudComPy

The binary **CloudComPy*_-date-.7z** available [here](https://www.simulation.openfields.fr/index.php/download-binaries) is built in a Conda environment. (see [here](https://github.com/CloudCompare/CloudComPy/blob/master/doc/BuildWindowsConda.md) for the corresponding building instructions).

As CloudComPy is under development, these instructions and the link are subject to change from time to time...

This binary works only on Windows 10, and with a Conda environment as described below, not anywhere else!

You need a recent installation of Anaconda3 or miniconda3.

## Installing an environment for running CloudComPy and notebooks.

CloudComPy is an Python Wrapper for CloudCompare, which allows one user to call some algorithms directly from a python environment. 

You can navigate to the folder where you downloaded the CloudComPy exectuable and copy/paste the `environment.yml` file. 
Then you can run the following command to install a python environment compatible with CloudCompare and its python wrapper. 

```
conda env create -f environment.yml
```

This will create a new environment called `CloudComPy` in your conda installation. Don't forget to test CloudComPy before using it. 

You can do so by navigating to the CloudComPy310 directory containing the binaries and running the following script from an anaconda shell or similar. 

`````
conda activate CloudComPy310

envCloudComPy.bat
```

