<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />

<div align="center">
  <a href="https://github.com/alaaNfissi/WPT-Deep-Learning-SER-for-Suicide-Monitoring">
    <img src="figures/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Learnable Deep Wavelet Packet Transform for Speech Emotion Recognition in High-Risk Suicide Calls</h3>

  <p align="center">
    This paper has been accepted for publication in the 23rd IEEE International Conference on Machine Learning and Applications (ICMLA) 2024.
    <br />
   </p>
   <!-- <a href="https://github.com/alaaNfissi/WPT-Deep-Learning-SER-for-Suicide-Monitoring"><strong>Explore the docs »</strong></a> -->
</div>
   

  
<div align="center">

[![view - Documentation](https://img.shields.io/badge/view-Documentation-blue?style=for-the-badge)](https://github.com/alaaNfissi/SigWavNet-Learning-Multiresolution-Signal-Wavelet-Network-for-Speech-Emotion-Recognition/#readme "Go to project documentation")

</div>  


<div align="center">
    <p align="center">
    ·
    <a href="https://github.com/alaaNfissi/WPT-Deep-Learning-SER-for-Suicide-Monitoring/issues">Report Bug</a>
    ·
    <a href="https://github.com/alaaNfissi/WPT-Deep-Learning-SER-for-Suicide-Monitoring/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#abstract">Abstract</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#getting-the-code">Getting the code</a></li>
        <li><a href="#dependencies">Dependencies</a></li>
      </ul>
    </li>
    <li>
      <a href="#results">Results</a>
      <ul>
        <li><a href="#on-nspl-crise-dataset">On NSPL-CRISE dataset</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABSTRACT -->
## Abstract

<p align="justify"> In human-computer interaction and psychological evaluation, speech emotion recognition (SER) is crucial for interpreting emotional states from spoken language. Although there have been advancements, challenges such as system complexity, issues with feature distinctiveness, and noise interference continue to persist. This paper presents a novel end-to-end (E2E) deep learning multi-resolution framework for SER, which tackles these limitations by deriving significant representations directly from raw speech waveform signals. By leveraging the properties of wavelet packet transform (WPT), our approach introduces a learnable model for both wavelet bases and denoising through deep learning techniques. Unlike discrete wavelet transform (DWT), WPT offers a more detailed analysis by decomposing both approximation and detail coefficients, providing a finer resolution in the time-frequency domain. This capability enhances feature extraction by capturing more nuanced signal characteristics across different frequency bands. The framework incorporates a learnable activation function for asymmetric hard thresholding of wavelet packet coefficients. Our approach exploits the capabilities of wavelet packets for effective localization in both time and frequency domains. We then combine one-dimensional dilated convolutional neural networks (1D dilated CNN) with a spatial attention layer and bidirectional gated recurrent units (Bi-GRU) with a temporal attention layer to efficiently capture emotional features' nuanced spatial and temporal characteristics. By handling variable-length speech without segmentation and eliminating the need for pre or post-processing, the proposed model outperformed state-of-the-art methods on our NSPL-CRISE dataset and the public IEMOCAP dataset.</p>
<div align="center">
  
![model-architecture][model-architecture]
  
*Model's General Architecture*
  
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
* ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
* ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
* ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
* ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started
<p align="justify">
To ensure consistency and compatibility across our datasets, we first convert all audio signals to a uniform 16 KHz sampling rate and mono-channel format. We then divide each dataset into two primary subsets: 90% for training and validation purposes, and the remaining 10% designated for testing as unseen data. For the training and validation segments, we implement a 10-fold cross-validation method. This partitioning and the allocation within the cross-validation folds leverage stratified random sampling, a method that organizes the dataset into homogenous strata based on emotional categories. Unlike basic random sampling, this approach guarantees a proportional representation of each class, leading to a more equitable and representative dataset division.</p>

<p align="justify">
In the quest to identify optimal hyperparameters for our model, we utilize a grid search strategy. Hyperparameter tuning can be approached in several ways, including the use of scheduling algorithms. These schedulers can efficiently manage trials by early termination of less promising ones, as well as pausing, duplicating, or modifying the hyperparameters of ongoing trials. For its effectiveness and performance, we have selected the Asynchronous Successive Halving Algorithm (ASHA) as our optimization technique. The data preprocessing used in this study is provided in the `Data_exploration` folder.  
</p>

### Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/alaaNfissi/WPT-Deep-Learning-SER-for-Suicide-Monitoring.git

or [download a zip archive](https://github.com/alaaNfissi/WPT-Deep-Learning-SER-for-Suicide-Monitoring/archive/refs/heads/main.zip).

### Dependencies

<p align="justify">
You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `requirements.txt`.
We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).
Run the following command to create an `ser-env` environment to create a separate environment:
  
```sh 
    conda create --name ser-env
```

Activate the environment, this will enable it for your current terminal session. Any subsequent commands will use software that is installed in the environment:

```sh 
    conda activate ser-env
 ```

Use Pip to install packages to the Anaconda Environment:

```sh 
    conda install pip
```

Install all required dependencies in it:

```sh
    pip install -r requirements.txt
```
  
</p>

## Results

### On NSPL-CRISE dataset
<p align="justify"> 
The trials showcase the model's proficiency in recognizing diverse emotional expressions from the NSPL-CRISE dataset. The confusion matrix highlights our model's strong performance and areas for improvement. It accurately classifies "Angry" at 75.68%, with some confusion with "Neutral" and "Sad" at 12.16% each. "FCW" shows 68.18% accuracy, often misclassified as "Sad" (18.18%) and "Neutral" (9.09%). "Happy" is recognized with 76.32% accuracy, but is sometimes confused with "Angry" (10.53%) and "Neutral" (9.21%). "Neutral" has a high accuracy of 78.02%, with misclassifications involving "Sad" (9.89%), "Angry" (6.59%), and "Happy" (5.49%). "Sad" is correctly identified 74.67% of the time, with errors in "FCW" (13.33%) and "Neutral" (9.33%). Our model stands out with a test accuracy of 75.3\% and an F1-score of 75.5\% on NSPL-CRISE and a test accuracy of 85.9\% and an F1-score of 86.2\% on IEMOCAP, significantly outperforming state-of-the-art methods.
</p>

Confusion matrix on NSPL-CRISE            | 
:-----------------------------------------------------------------:|
![nspl_crise_cfm](figures/nspl_crise_cfm_FWPT.png)  |


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<p align="center">
  
_For more detailed experiments and results you can read the paper._
</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Github Link: [https://github.com/alaaNfissi/WPT-Deep-Learning-SER-for-Suicide-Monitoring](https://github.com/alaaNfissi/WPT-Deep-Learning-SER-for-Suicide-Monitoring)

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[model-architecture]: figures/WPT_SigWavNet_Architecture.png


[anaconda.com]: https://anaconda.org/conda-forge/mlconjug/badges/version.svg
[anaconda-url]: https://anaconda.org/conda-forge/mlconjug

[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
