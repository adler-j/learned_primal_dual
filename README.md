Learned Primal-Dual Reconstruction
==================================

This repository contains the code for the article "[Learned Primal-Dual Reconstruction](https://arxiv.org/abs/1707.06474)".

Contents
--------
The code contains the following

* Training using ellipse phantoms
* Evaluation on ellipse phantoms
* Training using anthropomorphic data from Mayo Clinic.
* Evaluation on example slice
* Reference reconstructions of the above using [ODL](https://github.com/odlgroup/odl).

Pre-trained networks
--------------------
The pre-trained networks are currently under finalization and will be released soon.

Dependencies
------------
The code is currently based on the latest version of [ODL](https://github.com/odlgroup/odl/pull/972). It can be most easily installed by running 

```bash
$ pip install https://github.com/odlgroup/odl/archive/master.zip
```

The code also requires the utility library [adler](https://github.com/adler-j/adler) which can be installed via

```bash
$ pip install https://github.com/adler-j/adler/archive/master.zip
```

Contact
-------
[Jonas Adler](https://www.kth.se/profile/jonasadl), PhD student  
KTH, Royal Institute of Technology  
Elekta Instrument AB  
jonasadl@kth.se

[Ozan Öktem](https://www.kth.se/profile/ozan), Associate Professor  
KTH, Royal Institute of Technology  
ozan@kth.se

Funding
-------
Development is financially supported by the Swedish Foundation for Strategic Research as part of the project "Low complexity image reconstruction in medical imaging" and "3D reconstruction with simulated forward models".

Development has also been financed by [Elekta](https://www.elekta.com/).
